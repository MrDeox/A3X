from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def generate_code_patch(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Gera trechos de código ou patches para modificações em arquivos com base em instruções específicas.
    
    Args:
        context: Objeto de contexto contendo memória e informações do agente. Deve incluir
                 'target_file' (arquivo alvo), 'instructions' (instruções para o patch),
                 e opcionalmente 'existing_code' (código existente para contexto).
    
    Returns:
        Dict[str, Any]: Resultado contendo o código gerado ou patch, e uma mensagem de status.
    """
    if context is None:
        context = {}
    
    # Extrair informações do contexto
    target_file = context.get('target_file', 'não especificado')
    instructions = context.get('instructions', 'Gerar código genérico.')
    existing_code = context.get('existing_code', '')
    
    logger.info(f"[GenerateCodePatch] Gerando patch para {target_file} com instruções: {instructions}")
    
    # Lógica para gerar o código ou patch baseado nas instruções
    if 'adjust_llm_parameters' in instructions.lower():
        patch = """
# Se a resposta do LLM estiver vazia, tentar ajustar os parâmetros antes de falhar
if not response_content:
    agent_logger.warning(\"[Planner] Resposta vazia do LLM detectada. Chamando skill adjust_llm_parameters para ajustar configurações.\")
    from a3x.skills.adjust_llm_parameters import adjust_llm_parameters
    adjustment_result = adjust_llm_parameters(context={'mem': {}})
    agent_logger.info(f\"[Planner] Resultado do ajuste do LLM: {adjustment_result.get('message', 'Ajuste falhou')}\")
    # Tentar novamente com os novos parâmetros
    async for chunk in call_llm(messages, llm_url=llm_url, stream=False):
        response_content += chunk
"""
        message = f"Patch gerado para integrar adjust_llm_parameters em {target_file}."
    elif 'json_find_gpt fix' in instructions.lower():
        patch = """
def json_find_gpt(input_str: str):
    \"""
    Finds the first JSON object demarcated by ```json ... ``` or ``` ... ```.
    Helper based on AutoGPT's parsing. Also handles if the whole string is JSON.
    \"""
    # Try finding ```json ``` block
    im_json = re.search(
        r\"```(?:json)?\s*\n(.*?)\n```\", input_str, re.DOTALL | re.IGNORECASE
    )
    
    if im_json:
        return im_json
    else:
        # Fallback: Check if the entire string is valid JSON
        try:
            json.loads(input_str)
            # If parsing succeeds, wrap it to mimic the regex group structure
            class MockMatch:
                _content = input_str
                def group(self, num):
                    if num == 1:
                        return self._content
                    return None
            return MockMatch()
        except json.JSONDecodeError:
            return None
"""
        message = f"Patch gerado para corrigir o docstring da função json_find_gpt em {target_file}."
    else:
        patch = "# Código ou patch genérico gerado. Por favor, refine as instruções para um resultado mais específico."
        message = f"Instruções genéricas recebidas. Patch básico gerado para {target_file}."
    
    logger.info(f"[GenerateCodePatch] {message}")
    
    return {
        'status': 'success',
        'message': message,
        'patch': patch,
        'target_file': target_file
    } 