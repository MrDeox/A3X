import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List
import asyncio

# LLM desativado. Este fragmento só deve ser usado como fonte passiva de conhecimento
# via comando A3L explícito ('learn_from_professor'). 
# O A³Net é o único responsável por planejamento e decisão.

# --- Controle Global --- 
# Defina como False para desativar completamente o LLM, mesmo para 'aprender com'
LLM_ENABLED = True 


try:
    from a3x.core.llm_interface import LLMInterface
except ImportError:
    LLMInterface = None # Ou adicione um pass

logger = logging.getLogger(__name__)

# Classe comentada conforme solicitado, pois A³Net assumiu planejamento.
# ATENÇÃO: Isso quebrará a funcionalidade do comando A3L 'learn_from_professor'.
# Descomente se precisar usar o LLM como fonte de conhecimento passiva via A3L.
class ProfessorLLMFragment(nn.Module):
    """
    Fragmento especializado em interagir com um LLM para obter conhecimento textual.
    Atua como um 'oráculo' passivo, respondendo a perguntas específicas 
    originadas por comandos A3L.
    """

    def __init__(self, fragment_id: str, description: str, llm_url: Optional[str] = None, mock_response: Optional[str] = None):
        """Inicializa o ProfessorLLMFragment.

        Args:
            fragment_id: ID único.
            description: Descrição funcional.
            llm_url: URL do servidor LLM (se não usar mock).
            mock_response: Resposta fixa para testes (se llm_url não for fornecido).
        """
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self.llm_url = llm_url
        self.mock_response = mock_response
        self.llm_interface = None
        self.is_active = False
        status_msg = ""

        if not LLM_ENABLED:
            status_msg = "LLM_ENABLED=False. Fragmento desativado globalmente."
            logger.warning(f"ProfessorLLMFragment '{self.fragment_id}' está globalmente desativado (LLM_ENABLED=False).")
            self.is_active = False
        elif self.llm_url and LLMInterface:
            try:
                self.llm_interface = LLMInterface(llm_url=self.llm_url)
                status_msg = f"LLM Interface initialized with URL: {self.llm_url}"
                logger.info(f"ProfessorLLMFragment '{self.fragment_id}' initialized LLMInterface with URL: {self.llm_interface.llm_url}")
                self.is_active = True
            except Exception as e:
                logger.error(f"ProfessorLLMFragment '{self.fragment_id}' failed to initialize LLMInterface: {e}", exc_info=True)
                status_msg = f"LLM Interface FAILED: {e}. Using mock if available."
                self.llm_interface = None # Ensure it's None on failure
                self.is_active = bool(self.mock_response) # Active only if mock exists
        elif self.mock_response:
            status_msg = "Using mock response."
            logger.warning(f"ProfessorLLMFragment '{self.fragment_id}' will use ONLY mock responses.")
            self.is_active = True
        else:
             status_msg = "No LLM URL or mock response provided. Fragment is inactive."
             logger.error(f"ProfessorLLMFragment '{self.fragment_id}' created without LLM URL or mock response.")
             self.is_active = False

        logger.info(f"ProfessorLLMFragment '{self.fragment_id}' initialized ({status_msg}). Active: {self.is_active}")

    async def ask_llm(self, question: str) -> str:
        """Envia uma pergunta para o LLM e retorna a resposta textual."""
        logger.info(f"Professor '{self.fragment_id}' received question: '{question[:100]}...'")
        
        if not self.is_active:
            logger.error(f"Professor '{self.fragment_id}' is inactive. Cannot answer.")
            return "<Professor fragment inactive or disabled>"

        if self.llm_interface:
            try:
                # --- Montar o prompt A3L (v6 - Foco no Estilo) --- 
                system_prompt = ("Você está se comunicando com um sistema simbólico que entende uma linguagem chamada A3L. "
                               "Essa linguagem é **quase igual à linguagem natural**, mas com uma estrutura mais direta e explícita.\n\n"
                               "Sua tarefa é **reescrever a intenção dada usando frases curtas, uma por linha, começando sempre com verbos como 'criar', 'treinar', 'refletir', 'interpretar', 'aprender' etc.**\n\n"
                               "Não explique. Não use texto solto. Apenas traduza para instruções diretas, como nos exemplos abaixo.\n\n"
                               "### Exemplos:\n\n"
                               "Entrada: \"Quero treinar um agente que gere ideias de marketing\"\n"
                               "Saída:\n"
                               "criar fragmento 'gerador_marketing' tipo 'neural' input_dim 128 hidden_dim 64 output_dim 1\n"
                               "treinar fragmento 'gerador_marketing' por 10 épocas\n"
                               "refletir sobre fragmento 'gerador_marketing'\n\n"
                               "Entrada: \"Me ensine como ganhar dinheiro com IA\"\n"
                               "Saída:\n"
                               "aprender com 'prof_geral' question \"Como ganhar dinheiro com IA?\"\n\n"
                               "Entrada: \"Crie um plano para automatizar vendas com IA\"\n"
                               "Saída:\n"
                               "criar fragmento 'autom_vendas' tipo 'neural' input_dim 128 hidden_dim 64 output_dim 1\n"
                               "treinar fragmento 'autom_vendas' por 10 épocas\n"
                               "interpretar texto 'Quero automatizar vendas usando IA'\n\n"
                               "Agora reescreva a seguinte frase nesse formato A3L:\n")
                               
                # Combine system prompt and user question (objetivo)
                messages = [
                    # System prompt sets the stage and provides examples
                    {"role": "system", "content": system_prompt},
                    # The user question is the phrase to be rewritten
                    {"role": "user", "content": question} 
                ]
                # ---------------------------------------
                
                # Chamar o LLM de forma não-streaming para obter a resposta completa
                logger.info(f"# [ProfessorLLM] Enviando prompt A3L (v6 Estilo) para LLM...") # Updated log
                # --- Log messages being sent ---
                logger.debug(f"# [ProfessorLLM] Messages structure: {messages}")
                # -------------------------------
                full_response = ""
                async for chunk in self.llm_interface.call_llm(messages=messages, stream=False, temperature=0.2): # Lower temp for structured output
                    # Optional: Log chunk info if needed for streaming issues
                    # logger.debug(f"# [ProfessorLLM] Received chunk (len={len(chunk)}): {chunk[:50]}...")
                    full_response += chunk
                
                # --- Log raw response length and content ---
                logger.info(f"# [ProfessorLLM] Raw full_response received (length: {len(full_response)}).")
                logger.debug(f"# [ProfessorLLM] Raw full_response content: {full_response}") 
                # -------------------------------------------
                
                # Cleanup potentially unwanted markdown code blocks often added by models
                cleaned_response = full_response.strip()
                if cleaned_response.startswith("```a3l"):
                    cleaned_response = cleaned_response[len("```a3l"):].strip()
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[len("```"):].strip()
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-len("```")].strip()
                    
                # --- Log cleaned response ---
                logger.info(f"# [ProfessorLLM] Returning cleaned response (length: {len(cleaned_response)}). Content: {cleaned_response}")
                # ----------------------------
                return cleaned_response
            except Exception as e:
                logger.error(f"Professor '{self.fragment_id}' failed to get response from LLM: {e}", exc_info=True)
                # Fallback para mock se erro ocorrer?
                if self.mock_response:
                    logger.warning(f"Professor '{self.fragment_id}' failed LLM call, returning mock response.")
                    return self.mock_response.replace("{question}", question)
                else:
                    return f"<Error communicating with LLM: {e}>"
        elif self.mock_response:
            logger.warning(f"Professor '{self.fragment_id}' returning mock response.")
            # Substituir placeholders na mock response se necessário (exemplo)
            processed_mock = self.mock_response.replace("{question}", question)
            return processed_mock
        else:
             # Should not happen if self.is_active is True, but as safeguard:
             logger.error(f"Professor '{self.fragment_id}' is marked active but has no llm_interface or mock.")
             return "<Professor fragment configuration error>"

    def generate_reflection_a3l(self) -> str:
        """Gera descrição A3L do fragmento."""
        status = "inactive"
        if self.is_active:
            if self.llm_interface:
                 status = f"active (LLM: {self.llm_url})"
            elif self.mock_response:
                 status = "active (mock)"
        elif not LLM_ENABLED:
             status = "disabled (LLM_ENABLED=False)"
             
        return (f"fragmento '{self.fragment_id}' ({self.__class__.__name__}), "
                f"consulta LLM como fonte passiva de conhecimento via A3L. "
                f"Status: {status}. Descrito como: '{self.description}'.")

# Exemplo de como seria instanciado (se não comentado)
# professor = ProfessorLLMFragment(
#     fragment_id="prof_geral", 
#     description="Oráculo LLM geral", 
#     llm_url="http://localhost:8080/completion" # Exemplo
# )