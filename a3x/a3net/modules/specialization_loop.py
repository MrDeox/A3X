import torch
import torch.optim as optim
import torch.nn as nn
from typing import Optional, List, Dict, Any
import logging

# Updated import paths relative to project structure
from a3x.a3net.integration.a3x_bridge import handle_directive, MEMORY_BANK
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
from a3x.a3net.core.reflective_language_fragment import ReflectiveLanguageFragment

# Import from sibling module
from .utils import append_to_log, _log_result, analisar_reflexao_e_sugerir_criacao

logger = logging.getLogger(__name__)

def iniciar_ciclo_especializacao(
    new_id: str,
    base_id: str,
    results_summary: dict,
    input_dim: int = 128,
    low_confidence_threshold: float = 0.6,
    focused_epochs: int = 3,
    depth: int = 0, # Adicionado parâmetro de profundidade
    max_depth: int = 5, # Adicionado limite máximo de profundidade
    problematic_input: Optional[List[float]] = None # Entrada que causou a criação
):
    """Inicia um ciclo de especialização para um fragmento recém-criado."""
    print(f"\n--- [Ciclo Especialização {new_id} | Profundidade {depth}] Iniciando ---")
    append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Iniciando ciclo baseado em {base_id}")

    # 1. Treinamento Focado
    focused_training_success = False
    if problematic_input:
        print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Iniciando Treinamento Focado com base na entrada problemática...")
        append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Treinamento focado ({focused_epochs} épocas) com base na entrada anterior.")

        try:
            # Obter o fragmento especialista
            # --- Re-instanciar fragmento do arquivo para garantir objeto completo --- 
            state_file_path = MEMORY_BANK.save_dir / f"{new_id}.pt"

            saved_data = torch.load(state_file_path)
            state_dict = saved_data.get('state_dict')
            fragment_args = saved_data.get('init_args')
            fragment_class_name = saved_data.get('class_name')

            if not state_dict or not fragment_args or not fragment_class_name or fragment_class_name != 'ReflectiveLanguageFragment':
                 # Log mais detalhado do erro
                 missing_keys = []
                 if not state_dict: missing_keys.append('state_dict')
                 if not fragment_args: missing_keys.append('init_args') # Informar a chave correta esperada
                 if not fragment_class_name: missing_keys.append('class_name')
                 error_detail = f"Chaves faltando ou inválidas: {', '.join(missing_keys)}" if missing_keys else f"class_name esperado 'ReflectiveLanguageFragment', mas obtido '{fragment_class_name}'"
                 raise ValueError(f"Dados salvos inválidos para {new_id} em {state_file_path}. Detalhe: {error_detail}")

            fragment = ReflectiveLanguageFragment(**fragment_args)
            fragment.load_state_dict(state_dict)
            fragment.eval() # Garantir modo de avaliação inicial
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Fragmento re-instanciado e state_dict carregado de {state_file_path}")
            # --- Fim Re-instanciação ---

            # Obter o rótulo alvo do fragmento BASE usando a entrada problemática
            base_ask_directive = {"type": "ask", "fragment_id": base_id, "input": problematic_input}
            base_ask_result = handle_directive(base_ask_directive)
            if not base_ask_result or base_ask_result.get("status") != "success":
                 raise ValueError(f"Falha ao obter rótulo base para treino focado: {base_ask_result}")
            target_label_str = base_ask_result.get("output")
            # --- Corrigir acesso para id_to_label e inverter mapeamento ---
            if not hasattr(fragment, 'id_to_label') or not fragment.id_to_label:
                raise ValueError(f"Atributo 'id_to_label' ausente ou vazio no especialista {new_id}.")
            # Inverter o dicionário para obter Label -> ID
            label_to_id_map = {v: k for k, v in fragment.id_to_label.items()}
            if target_label_str not in label_to_id_map:
                # Melhorar mensagem de erro mostrando o mapeamento real
                raise ValueError(f"Rótulo base '{target_label_str}' não encontrado no mapeamento do especialista {new_id}: {fragment.id_to_label}")
            target_label_idx = label_to_id_map[target_label_str]
            target_tensor = torch.tensor([target_label_idx], dtype=torch.long)
            # --- Fim da correção ---

            # Gerar dados de treino (variações da entrada com ruído)
            num_variations = 10 # Quantidade de variações para o mini-batch
            noise_level = 0.05  # Nível do ruído gaussiano
            input_tensor = torch.tensor(problematic_input, dtype=torch.float32).unsqueeze(0)
            noise = torch.randn_like(input_tensor) * noise_level
            training_inputs = input_tensor + noise
            for _ in range(num_variations - 1):
                noise = torch.randn_like(input_tensor) * noise_level
                training_inputs = torch.cat((training_inputs, input_tensor + noise), dim=0)

            # Configurar otimizador e critério
            # --- Corrigir acesso aos parâmetros ---
            optimizer = optim.Adam(fragment.parameters(), lr=0.001) # <<< Usar fragment.parameters()
            criterion = nn.CrossEntropyLoss()

            # Loop de treinamento direto
            # --- Corrigir chamada de modo ---
            fragment.train() # <<< Chamar train() no fragmento inteiro
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Executando {focused_epochs} épocas de treino direto...")
            for epoch in range(focused_epochs):
                epoch_loss = 0.0
                optimizer.zero_grad()
                # Mini-batch com as variações + label alvo repetido
                # --- Corrigir forward pass ---
                outputs = fragment(training_inputs) # <<< Chamar o fragmento diretamente (invoca forward)
                loss = criterion(outputs, target_tensor.repeat(num_variations))
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item() # Loss do batch único
                print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Época [{epoch+1}/{focused_epochs}], Loss: {epoch_loss:.4f}")
                append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Treino Focado Época {epoch+1}, Loss: {epoch_loss:.4f}")
            # --- Corrigir chamada de modo ---
            fragment.eval() # <<< Chamar eval() no fragmento inteiro

            # Salvar o fragmento treinado
            MEMORY_BANK.save(new_id, fragment) # <<< Passar ID e o objeto fragmento
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Fragmento especialista salvo após treino focado.")
            focused_training_success = True
            results_summary["success"] += 1 # Conta treino focado como sucesso

        except Exception as e:
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Erro durante treino focado: {e}")
            logger.exception(f"Detailed error during focused training for {new_id}") # Log with traceback
            append_to_log(f"# [FALHA Ciclo Especialização {new_id} | Profundidade {depth}] Treino focado: {e}")
            results_summary["failed"] += 1
            # Não retorna aqui, permite que o ciclo continue para reavaliação mesmo se o treino falhar

    else:
        print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Treinamento focado pulado: Nenhuma entrada problemática fornecida.")
        append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Treino focado pulado (sem entrada problemática).")

    # 2. Reavaliação Pós-Treino
    print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Reavaliando confiança pós-treino...")
    append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Reavaliando confiança pós-treino.")
    test_input = [0.5] * input_dim # Usar input neutro padrão
    ask_directive = {
        "type": "ask",
        "fragment_id": new_id,
        "input": test_input
    }
    ask_result = handle_directive(ask_directive)
    print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Resultado Reavaliação: {ask_result}")

    if not ask_result or ask_result.get("status") != "success":
        error_msg = ask_result.get('explanation', 'Erro desconhecido') if ask_result else 'Resultado Nulo'
        print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Erro: Falha na reavaliação pós-treino: {error_msg}")
        append_to_log(f"# [FALHA Ciclo Especialização {new_id} | Profundidade {depth}] Reavaliação pós-treino: {error_msg}")
        results_summary["failed"] += 1
        return # Aborta o ciclo

    _log_result(ask_directive, ask_result, log_prefix=f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Reavaliação: ")
    results_summary["success"] += 1 # Conta reavaliação como sucesso
    new_confidence = ask_result.get("confidence", 0.0)

    # 3. Verificação Recursiva de Especialização
    if new_confidence < low_confidence_threshold:
        # Verificar limite de profundidade ANTES de prosseguir
        if depth >= max_depth:
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Limite máximo de recursão ({max_depth}) atingido. Interrompendo especialização.")
            append_to_log(f"# [Ciclo Especialização {new_id}] Limite máximo de recursão ({max_depth}) atingido.")
            return # Interrompe a recursão
        
        print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Confiança ainda baixa ({new_confidence:.2f} < {low_confidence_threshold}). Tentando refletir e especializar novamente.")
        append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Confiança ainda baixa ({new_confidence:.2f}). Iniciando reflexão.")

        # 3a. Refletir sobre o especialista
        reflect_cmd_str = f"refletir sobre fragmento '{new_id}' como a3l"
        reflect_directive = interpret_a3l_line(reflect_cmd_str)
        if not reflect_directive:
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Erro: Falha ao interpretar comando de reflexão.")
            append_to_log(f"# [FALHA Ciclo Especialização {new_id} | Profundidade {depth}] Erro ao interpretar reflexão recursiva.")
            results_summary["failed"] += 1
            return

        reflect_result = handle_directive(reflect_directive)
        print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Resultado Reflexão Recursiva: {reflect_result}")

        if not reflect_result or reflect_result.get("status") != "success" or "reflection_a3l" not in reflect_result:
            error_msg = reflect_result.get('error', 'Texto de reflexão A3L não encontrado') if reflect_result else 'Resultado Nulo'
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Erro: Falha na reflexão recursiva: {error_msg}")
            append_to_log(f"# [FALHA Ciclo Especialização {new_id} | Profundidade {depth}] Reflexão recursiva: {error_msg}")
            results_summary["failed"] += 1
            return

        _log_result(reflect_directive, reflect_result, log_prefix=f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Reflexão Recursiva: ")
        results_summary["success"] += 1 # Conta reflexão como sucesso
        reflection_text = reflect_result["reflection_a3l"]

        # 3b. Analisar reflexão e sugerir nova criação
        next_creation_suggestion = analisar_reflexao_e_sugerir_criacao(new_id, reflection_text)
        if next_creation_suggestion:
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Reflexão sugere nova especialização: {next_creation_suggestion}")
            append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Sugestão Recursiva: {next_creation_suggestion}")

            # 3c. Executar nova criação
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Interpretando e executando criação recursiva...")
            creation_directive = interpret_a3l_line(next_creation_suggestion)
            if not creation_directive:
                print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Erro: Falha ao interpretar comando de criação recursiva.")
                append_to_log(f"# [FALHA Ciclo Especialização {new_id} | Profundidade {depth}] Erro ao interpretar criação recursiva.")
                results_summary["failed"] += 1
                return

            creation_result = handle_directive(creation_directive)
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Resultado Criação Recursiva: {creation_result}")

            if not creation_result or creation_result.get("status") != "success":
                error_msg = creation_result.get('message', 'Erro desconhecido') if creation_result else 'Resultado Nulo'
                print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Erro: Falha na criação recursiva: {error_msg}")
                append_to_log(f"# [FALHA Ciclo Especialização {new_id} | Profundidade {depth}] Criação recursiva: {error_msg}")
                results_summary["failed"] += 1
                return

            _log_result(creation_directive, creation_result, log_prefix=f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Criação Recursiva: ")
            results_summary["success"] += 1 # Conta criação recursiva como sucesso
            new_id_v2 = creation_result.get("new_fragment_id")
            base_id_v2 = creation_result.get("base_fragment_id") # Deveria ser o new_id original

            if new_id_v2 and base_id_v2 == new_id:
                # CHAMADA RECURSIVA!
                iniciar_ciclo_especializacao(
                    new_id=new_id_v2,
                    base_id=base_id_v2, # O especialista anterior é a nova base
                    results_summary=results_summary,
                    input_dim=input_dim,
                    low_confidence_threshold=low_confidence_threshold,
                    focused_epochs=focused_epochs,
                    problematic_input=problematic_input, # Propagar a entrada original
                    depth=depth + 1, # Incrementar profundidade
                    max_depth=max_depth # Propagar o limite
                )
            else:
                print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Erro: IDs inconsistentes após criação recursiva.")
                append_to_log(f"# [FALHA Ciclo Especialização {new_id} | Profundidade {depth}] IDs inconsistentes após criação recursiva.")
                results_summary["failed"] += 1

        else:
            print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Reflexão recursiva não gerou nova sugestão. Especialização estabilizada (ou falhou em identificar melhoria).")
            append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Reflexão recursiva não gerou nova sugestão. Ciclo interrompido.")
            # (Opcional: Avaliar new_id vs base_id aqui para um relatório final)
            # avaliar_fragmento_criado(new_id, base_id, results_summary, input_dim)

    else:
        # Confiança satisfatória após treino focado
        print(f"[Ciclo Especialização {new_id} | Profundidade {depth}] Confiança satisfatória ({new_confidence:.2f} >= {low_confidence_threshold}) após treino focado. Ciclo concluído.")
        append_to_log(f"# [Ciclo Especialização {new_id} | Profundidade {depth}] Confiança satisfatória ({new_confidence:.2f}). Ciclo concluído.")
        # (Opcional: Avaliar new_id vs base_id aqui)
        # avaliar_fragmento_criado(new_id, base_id, results_summary, input_dim)

    print(f"--- [Ciclo Especialização {new_id}] Finalizado ---") 

async def reflect_fragment(fragment_id: str, output_format: str = "dict"):
    """Loads a fragment and calls its reflect method."""
    logger.info(f"[Reflect] Attempting reflection for fragment '{fragment_id}'...")
    fragment = MEMORY_BANK.load(fragment_id)
    if not fragment:
        logger.warning(f"[Reflect] Fragment '{fragment_id}' not found in MemoryBank.")
        # Return an error structure or specific value?
        return {"status": "error", "message": f"Fragment '{fragment_id}' not found for reflection."} 
    
    if not hasattr(fragment, "reflect") or not callable(getattr(fragment, "reflect")) :
        logger.warning(f"[Reflect] Fragment '{fragment_id}' (type: {type(fragment).__name__}) does not implement an async 'reflect' method.")
        return {"status": "error", "message": f"Fragment '{fragment_id}' does not support reflection."}
    
    logger.info(f"[Reflect] Calling reflect() on '{fragment_id}' (type: {type(fragment).__name__}) with format '{output_format}'.")
    try:
        # Assuming reflect is an async method
        result = await fragment.reflect(output_format=output_format) 
        
        # Log result snippet for debugging
        result_repr = f"\n{result}" if isinstance(result, str) else str(result)[:400]
        logger.info(f"[Reflect] Reflection result for '{fragment_id}': {result_repr}...")
        
        # Return a consistent dictionary structure
        return {"status": "success", "reflection_result": result} 

    except Exception as e:
        logger.error(f"[Reflect] Error during reflect() call for fragment '{fragment_id}': {e}", exc_info=True)
        return {"status": "error", "message": f"Error during reflection: {e}"} 