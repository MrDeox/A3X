import torch.nn as nn
import logging
from typing import Dict, Any, Optional
import asyncio

LLM_ENABLED = True

try:
    from a3x.core.llm_interface import LLMInterface
except ImportError:
    LLMInterface = None
    logging.warning("LLMInterface could not be imported. ProfessorLLMFragment will only work with mocks or if LLM_ENABLED is False.")

logger = logging.getLogger(__name__)


class ProfessorLLMFragment(nn.Module):
    def __init__(self, fragment_id: str, description: str, llm_url: Optional[str] = None, mock_response: Optional[str] = None):
        super().__init__()
        self.fragment_id = fragment_id
        self.description = description
        self.llm_url = llm_url
        self.mock_response = mock_response
        self.llm_interface = None
        self.is_active = False

        if not LLM_ENABLED:
            logger.warning(f"ProfessorLLMFragment '{self.fragment_id}' está globalmente desativado (LLM_ENABLED=False).")
        elif self.llm_url and LLMInterface:
            try:
                self.llm_interface = LLMInterface(llm_url=self.llm_url)
                self.is_active = True
                logger.info(f"ProfessorLLMFragment '{self.fragment_id}' initialized LLMInterface with URL: {self.llm_interface.llm_url}")
            except Exception as e:
                logger.error(f"ProfessorLLMFragment '{self.fragment_id}' failed to initialize LLMInterface: {e}", exc_info=True)
                self.llm_interface = None
                self.is_active = bool(self.mock_response)
        elif self.mock_response:
            self.is_active = True
            logger.warning(f"ProfessorLLMFragment '{self.fragment_id}' will use ONLY mock responses.")
        else:
            logger.error(f"ProfessorLLMFragment '{self.fragment_id}' created without LLM URL or mock response.")

    async def ask_llm(self, question: str) -> str:
        logger.info(f"Professor '{self.fragment_id}' received question: '{question[:100]}...'")
        if not self.is_active:
            return "<Professor fragment inactive or disabled>"

        if self.llm_interface:
            try:
                system_prompt = (
                    "Você está se comunicando com um sistema simbólico que entende uma linguagem chamada A3L. "
                    "Essa linguagem é quase igual à linguagem natural, mas com uma estrutura mais direta e explícita.\n\n"
                    "Sua tarefa é reescrever a intenção dada usando frases curtas, começando com verbos como 'criar', 'treinar', etc.\n"
                    "Não explique. Não use texto solto.\n\n"
                    "**IMPORTANTE:** Se for um erro indicando que o fragmento não existe, sugira:\n"
                    "1. listar fragmentos\n2. criar fragmento\n\n"
                    "Entrada: \"refletir sobre erro ao executar comando '{...}': Fragment 'aprendiz' not found.\"\n"
                    "Saída:\nlistar fragmentos\ncriar fragmento 'aprendiz' tipo '...' \n\n"
                    "Agora reescreva a seguinte frase nesse formato A3L:\n"
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
                full_response = ""
                async for chunk in self.llm_interface.call_llm(messages=messages, stream=False, temperature=0.2):
                    full_response += chunk
                cleaned_response = full_response.strip().strip("```a3l").strip("```").strip()
                logger.info(f"[{self.fragment_id}] LLM response: {cleaned_response}")
                return cleaned_response
            except Exception as e:
                logger.error(f"LLM call failed: {e}", exc_info=True)
                if self.mock_response:
                    return self.mock_response.replace("{question}", question)
                return f"<Error communicating with LLM: {e}>"
        elif self.mock_response:
            return self.mock_response.replace("{question}", question)
        else:
            return "<Professor fragment configuration error>"

    def generate_reflection_a3l(self) -> str:
        status = "inactive"
        if self.is_active:
            status = f"active (LLM: {self.llm_url})" if self.llm_interface else "active (mock)"
        elif not LLM_ENABLED:
            status = "disabled (LLM_ENABLED=False)"
        return (f"fragmento '{self.fragment_id}' ({self.__class__.__name__}), "
                f"consulta LLM como fonte passiva de conhecimento via A3L. "
                f"Status: {status}. Descrito como: '{self.description}'.")

    async def handle_message(self, **message: Dict[str, Any]) -> None:
        msg_type = message.get("message_type")
        content = message.get("content", {})
        sender_info = content.get("sender", "unknown_sender")

        logger.debug(f"[{self.fragment_id}] Received message: Type='{msg_type}', Sender='{sender_info}'")

        if msg_type == "ajuda_requerida":
            question = content.get("question") or content.get("original_text", "")
            if question:
                try:
                    llm_response = await self.ask_llm(question)
                    logger.info(f"[{self.fragment_id}] LLM Response: '{llm_response[:200]}...'")
                except Exception as e:
                    logger.error(f"Error processing 'ajuda_requerida': {e}", exc_info=True)
            else:
                logger.warning(f"[{self.fragment_id}] 'ajuda_requerida' sem pergunta válida.")
        else:
            await super().handle_message(message)
