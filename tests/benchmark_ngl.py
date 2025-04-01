# benchmark_ngl.py
import argparse
import subprocess
import time
import requests
import psutil  # Requires: pip install psutil
import os
import signal
from statistics import mean, stdev
import sys

DEFAULT_TEST_PROMPT = "Explique o conceito de 'type hinting' em Python em pelo menos 50 palavras, fornecendo um exemplo curto."
DEFAULT_SERVER_EXE = "./llama-server"  # Ajuste se necessário
DEFAULT_MODEL_PATH = (
    "../models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Ajuste se necessário
)


def find_and_kill_server(server_exe_name="llama-server", sig=signal.SIGTERM):
    """Encontra processos pelo nome do executável e tenta terminá-los."""
    killed_count = 0
    force_killed = 0  # Inicializa a variável aqui
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            # Checa se o nome base do executável no cmdline corresponde
            proc_name = ""
            if proc.info["cmdline"] and len(proc.info["cmdline"]) > 0:
                # Pega o nome base do primeiro argumento (o executável)
                proc_name = os.path.basename(proc.info["cmdline"][0])

            # Compara nome base ou nome do processo psutil
            if proc_name == server_exe_name or proc.info["name"] == server_exe_name:
                print(
                    f"  Encontrado processo '{server_exe_name}' (PID: {proc.pid}). Tentando terminar com sinal {sig.name}..."
                )
                proc.send_signal(sig)
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Processo já pode ter morrido ou não temos permissão
            pass
        except Exception as e:
            print(f"  Erro inesperado ao tentar terminar PID {proc.pid}: {e}")

    if killed_count > 0:
        print(
            f"  {killed_count} processo(s) '{server_exe_name}' sinalizado(s) para terminar."
        )
        time.sleep(2)  # Dá um tempo para os processos terminarem
        # Verifica se ainda existem e força o kill se necessário
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                proc_name = ""
                if proc.info["cmdline"] and len(proc.info["cmdline"]) > 0:
                    proc_name = os.path.basename(proc.info["cmdline"][0])

                if proc_name == server_exe_name or proc.info["name"] == server_exe_name:
                    print(
                        f"  Processo '{server_exe_name}' (PID: {proc.pid}) ainda vivo. Forçando KILL."
                    )
                    proc.kill()
                    force_killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if force_killed > 0:
            print(
                f"  {force_killed} processo(s) '{server_exe_name}' forçados a terminar (KILL)."
            )
            time.sleep(1)

    return (
        killed_count + force_killed > 0
    )  # Retorna True se algum processo foi encontrado/morto


def start_server(server_path, model_path, ngl, host, port, context_size=2048):
    """Inicia o llama-server como um subprocesso."""
    command = [
        server_path,
        "-m",
        model_path,
        "-ngl",
        str(ngl),
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(context_size),  # Tamanho do contexto
        # Adicione outras flags se necessário, ex: --threads
    ]
    print(f"  Iniciando servidor com comando: {' '.join(command)}")
    try:
        # Inicia sem capturar stdout/stderr para vermos a saída do llama-server
        server_process = subprocess.Popen(command)
        print(f"  Servidor iniciado (PID: {server_process.pid}).")
        return server_process
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Executável do servidor não encontrado em '{server_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO CRÍTICO: Falha ao iniciar o servidor: {e}")
        sys.exit(1)


def check_server_ready(url, timeout=15, attempts=5):
    """Verifica se o servidor está respondendo na URL base."""
    check_url = f"{url}/"  # Tenta acessar a raiz
    print(f"  Verificando se o servidor está pronto em {check_url}...")
    for i in range(attempts):
        try:
            response = requests.get(check_url, timeout=timeout)
            # llama.cpp server pode retornar 404 na raiz, mas se respondeu, está ok
            if response.status_code == 200 or response.status_code == 404:
                print(
                    f"  Servidor respondeu (Status: {response.status_code}). Pronto para testes."
                )
                return True
            else:
                print(
                    f"  Tentativa {i + 1}/{attempts}: Servidor respondeu com status inesperado {response.status_code}."
                )
        except requests.exceptions.ConnectionError:
            print(f"  Tentativa {i + 1}/{attempts}: Conexão recusada. Aguardando...")
        except requests.exceptions.Timeout:
            print(f"  Tentativa {i + 1}/{attempts}: Timeout na conexão.")
        except requests.exceptions.RequestException as e:
            print(f"  Tentativa {i + 1}/{attempts}: Erro na requisição: {e}")

        if i < attempts - 1:
            time.sleep(
                timeout // attempts + 1
            )  # Espera um pouco antes de tentar novamente

    print(f"  ERRO: Servidor não ficou pronto após {attempts} tentativas.")
    return False


def run_benchmark(chat_url, prompt, num_runs=3, api_timeout=180):
    """Executa o benchmark enviando prompts para a API de chat."""
    times = []
    payload = {
        "model": "mistral-7b",  # O nome pode não importar tanto se só há um modelo carregado
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,  # Baixa temperatura para consistência
        "max_tokens": 150,  # Limita o tamanho da resposta
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    print(f"  Iniciando benchmark ({num_runs} execuções)...")
    for i in range(num_runs):
        print(f"    Execução {i + 1}/{num_runs}...")
        start_time = time.perf_counter()
        try:
            response = requests.post(
                chat_url, headers=headers, json=payload, timeout=api_timeout
            )
            response.raise_for_status()  # Levanta exceção para status 4xx/5xx

            end_time = time.perf_counter()
            duration = end_time - start_time
            times.append(duration)
            print(f"      Sucesso! Duração: {duration:.3f}s")

            # Opcional: Imprimir a resposta para depuração
            # try:
            #     response_data = response.json()
            #     print(f"      Resposta (trecho): {response_data['choices'][0]['message']['content'][:50]}...")
            # except Exception:
            #     print("      Não foi possível decodificar a resposta JSON.")

        except requests.exceptions.Timeout:
            print(f"      ERRO: Timeout ({api_timeout}s) na chamada da API.")
            return None, None  # Indica falha
        except requests.exceptions.ConnectionError:
            print("      ERRO: Falha na conexão com a API (Servidor caiu?).")
            return None, None  # Indica falha
        except requests.exceptions.RequestException as e:
            print(f"      ERRO: Falha na requisição API: {e}")
            # Erros 5xx podem indicar OOM no servidor
            if response is not None and 500 <= response.status_code < 600:
                print(
                    "      ERRO 5xx: Possível erro de Out-of-Memory (OOM) no servidor."
                )
            return None, None  # Indica falha
        time.sleep(1)  # Pequena pausa entre as execuções

    if not times:
        return None, None

    avg_time = mean(times)
    std_dev_time = stdev(times) if len(times) > 1 else 0.0
    print("  Benchmark concluído para este ngl.")
    return avg_time, std_dev_time


def parse_ngl_values(ngl_input):
    """Processa a string de entrada para ngl_values."""
    values = set()  # Usar set para evitar duplicatas
    parts = ngl_input.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                if start > end:
                    start, end = end, start  # Garante ordem correta
                values.update(range(start, end + 1))
            except ValueError:
                print(f"AVISO: Ignorando range inválido '{part}' em --ngl-values.")
        else:
            try:
                values.add(int(part))
            except ValueError:
                print(f"AVISO: Ignorando valor inválido '{part}' em --ngl-values.")

    if not values:
        print("ERRO CRÍTICO: Nenhum valor de NGL válido foi fornecido ou parseado.")
        sys.exit(1)

    return sorted(list(values))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark llama-server com diferentes valores de -ngl."
    )
    parser.add_argument(
        "--server-path",
        type=str,
        default=DEFAULT_SERVER_EXE,
        help=f"Caminho para o executável llama-server (default: {DEFAULT_SERVER_EXE})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Caminho para o arquivo .gguf do modelo (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--ngl-values",
        type=str,
        required=True,
        help='Valores de NGL a testar. Ex: "0,8,16,24,33" ou "16-28"',
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host do servidor (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Porta do servidor (default: 8080)"
    )
    parser.add_argument(
        "--warmup-seconds",
        type=int,
        default=25,
        help="Tempo (s) para esperar o servidor carregar (default: 25)",
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        default=DEFAULT_TEST_PROMPT,
        help="Prompt a ser usado no benchmark.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Número de execuções por valor de NGL para calcular a média (default: 3)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=2048,
        help="Tamanho do contexto (-c) para o servidor (default: 2048)",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=180,
        help="Timeout (s) para cada chamada da API de chat (default: 180)",
    )

    args = parser.parse_args()

    # Verifica se o executável e o modelo existem
    if not os.path.isfile(args.server_path):
        print(
            f"ERRO CRÍTICO: Executável do servidor não encontrado em '{args.server_path}'"
        )
        sys.exit(1)
    if not os.path.isfile(args.model_path):
        print(f"ERRO CRÍTICO: Arquivo do modelo não encontrado em '{args.model_path}'")
        sys.exit(1)

    ngl_values_to_test = parse_ngl_values(args.ngl_values)
    print(f"Valores de NGL a serem testados: {ngl_values_to_test}")

    results = {}
    server_exe_name = os.path.basename(args.server_path)

    # Garante que nenhum servidor esteja rodando antes de começar
    print("\n--- Limpeza Inicial ---")
    find_and_kill_server(server_exe_name)
    time.sleep(2)

    for ngl in ngl_values_to_test:
        print(f"\n--- Testando com ngl = {ngl} ---")
        server_process = None  # Reseta a variável
        try:
            server_process = start_server(
                args.server_path,
                args.model_path,
                ngl,
                args.host,
                args.port,
                args.context_size,
            )

            print(
                f"  Aguardando {args.warmup_seconds}s para o servidor iniciar e carregar o modelo..."
            )
            time.sleep(args.warmup_seconds)

            server_url = f"http://{args.host}:{args.port}"
            chat_url = f"{server_url}/v1/chat/completions"

            # Verifica se o processo ainda está vivo antes de checar a prontidão
            try:
                if (
                    server_process.poll() is not None
                ):  # Se poll() não for None, o processo terminou
                    print(
                        f"  ERRO: Processo do servidor (PID: {server_process.pid}) terminou inesperadamente durante o warmup (Código: {server_process.poll()}). Provável OOM."
                    )
                    results[ngl] = "Falha (Crash no Warmup - OOM?)"
                    continue  # Pula para o próximo NGL
            except Exception as e:
                print(
                    f"  AVISO: Não foi possível verificar o status do processo do servidor: {e}"
                )

            if not check_server_ready(server_url, timeout=10, attempts=3):
                print(f"  ERRO: Servidor não respondeu em {server_url} após warmup.")
                results[ngl] = "Falha (Não Respondeu)"
                # Tenta matar o processo zumbi se ele ainda existir
                if server_process and server_process.poll() is None:
                    print("  Tentando matar processo do servidor que não respondeu...")
                    find_and_kill_server(server_exe_name, sig=signal.SIGKILL)
                continue  # Pula para o próximo NGL

            avg_time, std_dev_time = run_benchmark(
                chat_url, args.test_prompt, args.num_runs, args.api_timeout
            )

            if avg_time is None:
                # A função run_benchmark já imprime a causa do erro
                results[ngl] = "Falha (Erro API/Timeout/OOM?)"
            else:
                results[ngl] = f"{avg_time:.3f}s (±{std_dev_time:.3f}s)"
                print(f"  Resultado para ngl={ngl}: Tempo médio = {results[ngl]}")

        except Exception as e:
            print(f"ERRO INESPERADO durante o teste de ngl={ngl}: {e}")
            results[ngl] = "Falha (Erro Script)"
        finally:
            # Garante que o servidor seja morto após cada teste de NGL
            print(f"  Finalizando servidor para ngl={ngl}...")
            if server_process and server_process.poll() is None:
                find_and_kill_server(
                    server_exe_name, sig=signal.SIGKILL
                )  # Força kill para garantir limpeza rápida
            else:
                # Se o processo já morreu ou não foi iniciado, apenas garante que não há outros
                find_and_kill_server(server_exe_name, sig=signal.SIGKILL)
            time.sleep(2)  # Pausa antes do próximo NGL

    # Imprimir resultados finais
    print("\n--- Resultados Finais do Benchmark ---")
    print(f"Modelo: {os.path.basename(args.model_path)}")
    print(f'Prompt: "{args.test_prompt[:50]}..."')
    print(f"Execuções por NGL: {args.num_runs}")
    print("-" * 40)
    print(f"{'NGL':<5} | {'Tempo Médio (Desvio Padrão)':<30}")
    print("-" * 40)
    for ngl, result in sorted(results.items()):
        print(f"{ngl:<5} | {result:<30}")
    print("-" * 40)


if __name__ == "__main__":
    main()
