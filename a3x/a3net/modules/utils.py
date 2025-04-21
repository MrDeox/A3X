import re
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
import os

# Updated import paths relative to potential usage location or absolute
from a3x.a3net.integration.a3lang_interpreter import interpret_a3l_line
from a3x.a3net.integration.a3x_bridge import handle_directive, MEMORY_BANK

logger = logging.getLogger(__name__)

# --- Output Log File ---
OUTPUT_LOG_FILE = "a3x/a3net/examples/session_output.a3l" # Keep path relative to project root for now

# --- Helper Functions ---

def get_cumulative_epochs(fragment_id: str, log_file: str) -> int:
    """Reads the log file and sums the training epochs for a given fragment_id."""
    total_epochs = 0
    log_path = Path(log_file)
    if not log_path.exists():
        logger.warning(f"Log file {log_file} not found for cumulative epoch check.")
        return 0

    # Use raw string for regex pattern
    log_pattern = re.compile(r"^fragmento\s+'" + re.escape(fragment_id) + r"'\s+foi\s+treinado\s+por\s+(\d+)\s+épocas?", re.IGNORECASE)

    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = log_pattern.match(line.strip())
                if match:
                    try:
                        epochs = int(match.group(1))
                        total_epochs += epochs
                    except ValueError:
                        logger.warning(f"Could not parse epoch number in log line: {line.strip()}")
        # Don't print here, too verbose
        # print(f"[Epoch Check] Found {total_epochs} cumulative epochs for '{fragment_id}' in {log_file}")
    except Exception as e:
        logger.error(f"Error reading log file {log_file} for cumulative epoch check: {e}")
        return 0

    return total_epochs

def generate_symbolic_suggestion(fragment_id: str, confidence: float) -> Optional[str]:
    """Generates a symbolic A3L suggestion based on confidence."""
    if confidence < 0.6:
        return f"treinar fragmento '{fragment_id}' por 2 épocas"
    elif confidence < 0.9:
        return f"refletir sobre fragmento '{fragment_id}' como a3l"
    return None

def analisar_reflexao_e_sugerir_criacao(fragment_id: str, reflection_a3l: str) -> Optional[str]:
    """Analisa uma string de reflexão A3L e sugere a criação de um novo fragmento se necessário."""
    triggers = ["confiança baixa", "ambíguo", "conflito", "turbido", "default reflective"]
    reflection_lower = reflection_a3l.lower()
    found_trigger = None
    for trigger in triggers:
        if trigger in reflection_lower:
            found_trigger = trigger
            break

    if found_trigger:
        # Gera um novo ID baseado no gatilho (simplificado)
        trigger_suffix = re.sub(r'[^a-z0-9_]', '', found_trigger.replace(' ', '_'))
        # --- Limitar comprimento do ID base antes de adicionar sufixo --- 
        max_base_len = 64 - len('_especialista_') - len(trigger_suffix) - 5 # -5 para margem e contador
        base_fragment_id_truncated = fragment_id[:max_base_len]
        
        base_new_id = f"{base_fragment_id_truncated}_especialista_{trigger_suffix}"

        # --- Verificar se ID já existe e adicionar sufixo numérico --- 
        new_fragment_id = base_new_id
        counter = 1
        # Verificar a existência do arquivo .pt correspondente
        # Ensure MEMORY_BANK is initialized before this is called or pass save_dir
        if MEMORY_BANK.save_dir: 
            while os.path.exists(os.path.join(MEMORY_BANK.save_dir, f"{new_fragment_id}.pt")):
                new_fragment_id = f"{base_new_id}_{counter}"
                counter += 1
                if counter > 100: # Limite para evitar loop infinito acidental
                     logger.error(f"Exceeded counter limit trying to generate unique ID based on {base_new_id}")
                     return None # Falha ao gerar ID único
                     
            if new_fragment_id != base_new_id:
                 print(f"[Análise Reflexão] ID base '{base_new_id}' já existe, usando '{new_fragment_id}'.")
                 
            print(f"[Análise Reflexão] Gatilho '{found_trigger}' encontrado para '{fragment_id}'. Sugerindo criação de '{new_fragment_id}'.")
            return f"criar fragmento '{new_fragment_id}' com base em '{fragment_id}'"
        else:
            logger.error("MEMORY_BANK.save_dir not set. Cannot check for existing fragment files.")
            return None
    
    return None # Nenhuma sugestão de criação

def append_to_log(a3l_line: str):
    """Appends a line to the symbolic session log file."""
    try:
        with open(OUTPUT_LOG_FILE, "a") as f:
            f.write(a3l_line + "\n")
    except Exception as e:
        logger.error(f"Failed to write to output log file {OUTPUT_LOG_FILE}: {e}")

def _log_result(directive: Dict[str, Any], result: Dict[str, Any], log_prefix: str = ""):
    """Generates and logs the symbolic A3L string for a successful directive execution."""
    log_line = None
    directive_type = directive.get("type")

    if directive_type == "ask":
        frag_id = result.get("fragment_id")
        output = result.get("output")
        confidence = result.get("confidence")
        if frag_id and output is not None and confidence is not None:
             log_line = f"fragmento '{frag_id}' respondeu '{output}' com confiança {confidence:.2f}"

    elif directive_type == "train_fragment":
         frag_id = result.get("fragment_id")
         epochs = directive.get("epochs")
         if frag_id and epochs:
             log_line = f"fragmento '{frag_id}' foi treinado por {epochs} épocas"

    elif directive_type == "reflect_fragment":
        if directive.get("format") == "a3l" and "reflection_a3l" in result:
             log_line = result["reflection_a3l"]

    elif directive_type == "import_fragment":
         path = result.get("path")
         if path:
            log_line = f"fragmento importado de '{path}'"

    elif directive_type == "export_fragment":
        frag_id = result.get("fragment_id")
        path = result.get("path")
        if frag_id and path:
            log_line = f"fragmento '{frag_id}' exportado para '{path}'"

    elif directive_type == "create_fragment_from_base":
        new_frag_id = result.get("new_fragment_id")
        base_frag_id = result.get("base_fragment_id")
        if new_frag_id and base_frag_id:
            log_line = f"fragmento '{new_frag_id}' criado com base em '{base_frag_id}'"

    # --- Handle logging for actions within conditionals ---
    elif directive_type == "conditional_directive" or directive_type == "cumulative_epochs_conditional":
        # Log based on the nested action type, assuming result contains necessary info if success
        action_directive = directive.get("action", {})
        nested_directive_type = action_directive.get("type")

        if nested_directive_type == "ask":
            frag_id = result.get("fragment_id") or action_directive.get("fragment_id")
            output = result.get("output")
            confidence = result.get("confidence")
            if frag_id and output is not None and confidence is not None:
                log_line = f"fragmento '{frag_id}' respondeu '{output}' com confiança {confidence:.2f}"
        elif nested_directive_type == "train_fragment":
            frag_id = result.get("fragment_id") or action_directive.get("fragment_id")
            epochs = action_directive.get("epochs")
            if frag_id and epochs:
                log_line = f"fragmento '{frag_id}' foi treinado por {epochs} épocas"
        elif nested_directive_type == "reflect_fragment" and action_directive.get("format") == "a3l" and "reflection_a3l" in result:
             log_line = result["reflection_a3l"]
        elif nested_directive_type == "import_fragment":
             path = result.get("path") or action_directive.get("path")
             if path:
                 log_line = f"fragmento importado de '{path}'"
        elif nested_directive_type == "export_fragment":
            frag_id = result.get("fragment_id") or action_directive.get("fragment_id")
            path = result.get("path")
            if frag_id and path:
                 log_line = f"fragmento '{frag_id}' exportado para '{path}'"

    # Append to log file if a symbolic line was generated
    if log_line:
        append_to_log(log_prefix + log_line)

# --- Avaliação Pós-Criação ---
def avaliar_fragmento_criado(
    new_fragment_id: str,
    base_fragment_id: str,
    results_summary: dict,
    input_dim: int = 128,
    threshold: float = 0.05 # Margem mínima de melhora para considerar especializado
):
    """Avalia um fragmento recém-criado comparando-o com o fragmento base."""
    print(f"[Avaliação Pós-Criação] Avaliando '{new_fragment_id}' vs '{base_fragment_id}'")
    append_to_log(f"# [Avaliação Pós-Criação] Comparando {new_fragment_id} com {base_fragment_id}")

    # Usar um vetor de entrada neutro ou baseado em algum contexto relevante (a ser definido)
    # Por enquanto, um vetor neutro simples
    test_input = [0.5] * input_dim

    base_result = handle_directive({
        "type": "ask",
        "fragment_id": base_fragment_id,
        "input": test_input
    })
    new_result = handle_directive({
        "type": "ask",
        "fragment_id": new_fragment_id,
        "input": test_input
    })

    base_conf = base_result.get("confidence", 0.0) if base_result and base_result.get("status") == "success" else 0.0
    new_conf = new_result.get("confidence", 0.0) if new_result and new_result.get("status") == "success" else 0.0

    # Log mais detalhado dos resultados da avaliação
    base_output = base_result.get("output", "N/A") if base_result and base_result.get("status") == "success" else "ERRO"
    new_output = new_result.get("output", "N/A") if new_result and new_result.get("status") == "success" else "ERRO"

    log_msg = (
        f"Resultado: Novo ('{new_fragment_id}') -> {new_output} ({new_conf:.2f}) "
        f"vs Base ('{base_fragment_id}') -> {base_output} ({base_conf:.2f}). "
    )

    if new_conf > base_conf + threshold:
        log_msg += f"Especialização bem-sucedida (melhora > {threshold:.2f})."
        results_summary["success"] += 1 # Contabiliza como sucesso
    elif base_conf > new_conf + threshold:
        log_msg += f"Regressão detectada (piora > {threshold:.2f})."
        results_summary["failed"] += 1 # Contabiliza como falha
    else:
        log_msg += f"Especialização insuficiente ou performance similar (|diff| <= {threshold:.2f})."
        # Não contabiliza como sucesso ou falha, apenas neutro por enquanto
        # Poderia ser um 'warning' ou outro status

    print(f"[Avaliação Pós-Criação] {log_msg}")
    append_to_log(f"# [Avaliação Pós-Criação] {log_msg}")

    return new_conf > base_conf + threshold 