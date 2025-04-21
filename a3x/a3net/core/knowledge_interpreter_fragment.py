import torch
import torch.nn as nn
import re
import logging
from typing import List, Dict, Optional, Any, Tuple

# --- Import ProfessorLLMFragment conditionally for type hinting --- 
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .professor_llm_fragment import ProfessorLLMFragment

logger = logging.getLogger(__name__)

class KnowledgeInterpreterFragment(nn.Module):
    """
    Interpreta texto bruto (ex: respostas de LLM, logs) para extrair 
    sugestões simbólicas e convertê-las em comandos A3L executáveis.
    
    Este fragmento atua como um tradutor de conhecimento em linguagem natural 
    para a linguagem estruturada A3L, facilitando o aprendizado e a 
    adaptação do sistema com base em insights textuais.
    
    Pode consultar um ProfessorLLMFragment se a interpretação direta falhar.
    """
    
    # Regex patterns for A3L commands - Case-insensitive matching enabled later
    # Note: Using non-capturing groups (?:...) where appropriate.
    # CREATE: [DEPRECATED - agora checamos prefixo] criar [um/o] fragmento 'nome' [params]
    # CREATE_PATTERN = r"criar\s+(?:um\s+|o\s+)?fragmento\s+'([^']+)'(?:.*)?" # Old, less specific
    # TRAIN: (verb like treinar/ajustar/etc) fragmento 'nome' por N épocas
    TRAIN_PATTERN = r"(?:treinar|treina-lo|treinado|ajustar|refinar)\b\s+fragmento\s+'([^']+)'\s+por\s+(\d+)\s+épocas?"
    # REFLECT: refletir [sobre [o]] fragmento 'nome' [como a3l|json]
    REFLECT_PATTERN = r"(?:refletir)\s+(?:sobre)?\s*(?:o)?\s*fragmento\s+'([^']+)'(?:\s*como\s+(a3l|json))?" # Default format is a3l
    # ASK: perguntar/consultar [ao/o] fragmento 'nome' com/usando <input>
    # Input capture remains non-greedy for general text, explicit for lists.
    ASK_PATTERN = r"(?:perguntar|consultar)\s+(?:ao|o)?\s*fragmento\s+'([^']+)'\s+(?:com|usando)\s+(\[.*\]|.+?)"
    
    def __init__(self, 
                 fragment_id: str, 
                 description: str,
                 custom_patterns: Optional[Dict[str, re.Pattern]] = None):
        """Inicializa o KnowledgeInterpreterFragment.

        Args:
            fragment_id: Identificador único do fragmento.
            description: Descrição textual da sua função.
            custom_patterns: Opcional. Dicionário para substituir ou adicionar padrões regex.
                               Chaves devem ser como 'CREATE', 'TRAIN', etc.
        """
        super().__init__() # Herda de nn.Module por consistência, mas não usa camadas
        self.fragment_id = fragment_id
        self.description = description
        
        # Definir padrões regex a serem usados (CREATE agora é tratado por prefixo)
        self.patterns = {
            # "CREATE": re.compile(self.CREATE_PATTERN, re.IGNORECASE),
            "TRAIN": re.compile(self.TRAIN_PATTERN, re.IGNORECASE),
            "REFLECT": re.compile(self.REFLECT_PATTERN, re.IGNORECASE),
            "ASK": re.compile(self.ASK_PATTERN, re.IGNORECASE),
        }
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # --- Adiciona professor_fragment --- 
        self.professor_fragment: Optional["ProfessorLLMFragment"] = None 

        logger.info(f"KnowledgeInterpreterFragment '{self.fragment_id}' initialized.")
        print(f"[KnowledgeInterpreter '{self.fragment_id}'] Initialized.")

    def set_professor(self, professor: Optional["ProfessorLLMFragment"]):
        """Injeta a instância do Professor LLM Fragment."""
        self.professor_fragment = professor
        if professor:
            logger.info(f"[{self.fragment_id}] Professor fragment '{professor.fragment_id}' injected.")
        else:
            logger.info(f"[{self.fragment_id}] Professor fragment removed/not set.")

    def _extract_a3l_commands(self, text: str, context_fragment_id: Optional[str] = None) -> List[str]:
        """Helper interno para extrair comandos A3L de um bloco de texto.
           Reutiliza a lógica de padrões regex.
        """
        extracted_commands = []
        # Dividir texto em frases/linhas
        sentences = [s.strip() for s in re.split(r'\s*[.\n]+\s*', text) if s.strip()]

        for sentence in sentences:
            # --- Tratamento especial para CREATE (checa prefixo) ---
            # Normaliza espaços e ignora caso para checagem de prefixo
            normalized_sentence = ' '.join(sentence.lower().split())
            if normalized_sentence.startswith("criar fragmento "):
                 # Considera a frase inteira como um comando create candidato
                 extracted_commands.append(sentence.strip()) 
                 continue # Assume que uma linha CREATE não terá outros comandos
            # -----------------------------------------------------

            # Outros comandos (TRAIN, REFLECT, ASK) usam regex finditer
            # Treinar fragmento
            for match in self.patterns["TRAIN"].finditer(sentence):
                frag_id = match.group(1)
                epochs = match.group(2)
                # Código de validação e formatação do comando... (existente)
                if frag_id:
                    try:
                        epochs_int = int(epochs)
                        if epochs_int > 0:
                            command = f"treinar fragmento '{frag_id}' por {epochs_int} épocas"
                            extracted_commands.append(command)
                        else:
                            logger.warning(f"Invalid epoch number ({epochs}) ignored for training '{frag_id}'.")
                    except ValueError:
                        logger.warning(f"Could not parse epochs '{epochs}' for training '{frag_id}'.")

            # Refletir sobre fragmento
            for match in self.patterns["REFLECT"].finditer(sentence):
                frag_id = match.group(1)
                command = f"refletir sobre fragmento '{frag_id}' como a3l"
                extracted_commands.append(command)
             
            # Perguntar ao fragmento (simplificado)
            for match in self.patterns["ASK"].finditer(sentence):
                frag_id = match.group(1)
                input_str = match.group(2).strip()
                # Basic cleaning
                if input_str.endswith('.'): input_str = input_str[:-1] 
                if input_str.endswith('].'): input_str = input_str[:-1]
                
                # Mantenha a entrada como string para a diretiva A3L
                command = f"perguntar ao fragmento '{frag_id}' com {input_str}"
                extracted_commands.append(command)

        # Remover duplicatas mantendo a ordem
        unique_commands = []
        seen = set()
        for cmd in extracted_commands:
            if cmd not in seen:
                unique_commands.append(cmd)
                seen.add(cmd)
        return unique_commands

    async def interpret_knowledge(self, raw_text: str, context_fragment_id: Optional[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Analisa o texto bruto, extrai sugestões e as converte para comandos A3L.
        Tenta extrair diretamente, e se falhar, consulta o Professor LLM.
        
        Args:
            raw_text: O texto a ser interpretado.
            context_fragment_id: Opcional. ID do fragmento que é o contexto principal
                                 da conversa/reflexão, usado como fallback para 
                                 pronomes como "ele" ou "treiná-lo".

        Returns:
            Uma tupla: (lista de comandos A3L extraídos, metadados da interpretação).
        """
        metadata = {"source": "Unknown", "original_query": raw_text, "professor_consulted": False}
        
        # 1. Tentar extrair comandos diretamente usando regex
        direct_commands = self._extract_a3l_commands(raw_text, context_fragment_id)
        
        if direct_commands:
            logger.info(f"[{self.fragment_id}] Extraiu {len(direct_commands)} comandos diretamente.")
            metadata["source"] = "Direct Regex"
            return direct_commands, metadata

        # 2. Se não extraiu diretamente E um professor está disponível, consultar o professor
        elif self.professor_fragment:
            logger.info(f"[{self.fragment_id}] Nenhum comando direto encontrado. Consultando Professor '{self.professor_fragment.fragment_id}'...")
            metadata["professor_consulted"] = True
            try:
                # Formular prompt para o LLM gerar comandos A3L (v4, estilo simbiótico - F-STRINGS CORRIGIDAS)
                prompt = (
                    f"Você é um especialista na linguagem `.a3l`, que é praticamente idêntica à linguagem natural, mas com algumas convenções formais específicas para representar ações de forma estruturada.\n"
                    f"A linguagem `.a3l` é usada para controlar agentes autônomos e descrever intenções de forma executável.\n\n"
                    f"Seu trabalho é converter qualquer intenção dada em linguagem natural em uma sequência de comandos `.a3l`.\n\n"
                    f"### Convenções da linguagem `.a3l`:\n"
                    f"- Ações são verbos no infinitivo: `criar fragmento`, `treinar fragmento`, `refletir sobre`, etc.\n"
                    f"- Fragmentos são referenciados com nomes entre aspas simples: `'nome_fragmento'`\n"
                    f"- Tipos, parâmetros e valores são declarados de forma explícita, como: `tipo \'neural\'`, `input_dim 128`, `por 10 épocas`\n"
                    f"- Comentários iniciam com `#`\n"
                    f"- Não há explicações, apenas os comandos\n\n"
                    f"### Exemplo:\n"
                    f"**Entrada:** \"Quero que você me ajude a criar um agente que use IA para vender produtos online\"\n\n"
                    f"**Saída:**\n"
                    f"criar fragmento 'vendedor_ia' tipo 'neural' input_dim 128 output_dim 1\n"
                    f"treinar fragmento 'vendedor_ia' por 10 épocas\n"
                    f"refletir sobre fragmento 'vendedor_ia'\n"
                    f"\n---\n\n"
                    f"A partir de agora, sempre que você receber uma tarefa em linguagem natural, responda apenas com comandos `.a3l`, sem explicações. Se não for possível executar, diga:\n"
                    f"# Nenhum comando aplicável\n"
                    f"\n---\n\n"
                    f"### Tarefa Atual (Converter a seguinte intenção):\n"
                    f"{raw_text}\n"
                    f"\n### Saída `.a3l`:\n"
                ) # <<< Parêntese fechando a definição de prompt
                
                # Chamar o professor de forma assíncrona
                professor_response = await self.professor_fragment.ask_llm(prompt)
                logger.info(f"[{self.fragment_id}] Resposta do Professor: \n{professor_response}")

                # 3. Tentar extrair comandos A3L da RESPOSTA do professor
                # Garantir que a resposta seja uma string antes de processar
                if isinstance(professor_response, str):
                    commands_from_professor = self._extract_a3l_commands(professor_response, context_fragment_id)
                else:
                    logger.warning(f"[{self.fragment_id}] Resposta do professor não é uma string: {type(professor_response)}. Não é possível extrair comandos.")
                    commands_from_professor = []
                
                if commands_from_professor:
                    logger.info(f"[{self.fragment_id}] Extraiu {len(commands_from_professor)} comandos da resposta do Professor.")
                    metadata["source"] = f"Professor {self.professor_fragment.fragment_id}"
                    return commands_from_professor, metadata
                else:
                    logger.info(f"[{self.fragment_id}] Nenhum comando A3L extraído da resposta do Professor.")
                    metadata["source"] = "Professor Consulted (No Commands Found)"
                    return [], metadata

            except Exception as prof_err: # <<< Bloco except corretamente indentado
                logger.error(f"[{self.fragment_id}] Erro ao consultar Professor '{self.professor_fragment.fragment_id}': {prof_err}", exc_info=True)
                metadata["source"] = "Direct (Professor Consult Failed)"
                return [], metadata
        
        # 4. Se não extraiu diretamente E não há professor
        else:
            logger.info(f"[{self.fragment_id}] Nenhum comando A3L reconhecido diretamente e nenhum Professor disponível.")
            metadata["source"] = "Direct (No Commands Found)"
            return [], metadata

    def generate_reflection_a3l(self) -> str:
        """Gera uma descrição A3L do fragmento interpretador."""
        return (f"fragmento '{self.fragment_id}' ({self.__class__.__name__}), "
                f"interpreta texto para extrair comandos A3L. "
                f"Descrito como: '{self.description}'.")

    # --- Métodos Padrão nn.Module (não utilizados ativamente) ---
    def forward(self, *args, **kwargs):
        # Este fragmento não usa o forward pass padrão para tensores.
        # O método principal é interpret_knowledge.
        raise NotImplementedError("KnowledgeInterpreterFragment usa 'interpret_knowledge', não 'forward'.")

# Exemplo de uso (se executado diretamente)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Para ver os logs de debug
    
    interpreter = KnowledgeInterpreterFragment(
        fragment_id="interpretador_mestre",
        description="Interpreta logs e sugestões para A3L"
    )
    
    # Simular injeção do professor (requer ProfessorLLMFragment)
    # try:
    #     from professor_llm_fragment import ProfessorLLMFragment, LLM_ENABLED
    #     if LLM_ENABLED:
    #         mock_professor = ProfessorLLMFragment("prof_mock", "Mock Professor", llm_url="http://localhost:8080")
    #         interpreter.set_professor(mock_professor)
    # except ImportError:
    #     print("\n[AVISO] ProfessorLLMFragment não encontrado para teste completo.")
        
    print(interpreter.generate_reflection_a3l())
    
    test_text_1 = """
    Análise do log: A confiança para frag_alpha foi baixa (0.4). 
    Sugestão: treinar fragmento 'frag_alpha' por 5 épocas. 
    Além disso, talvez criar fragmento 'frag_alpha_especialista_confianca' com base em 'frag_alpha'.
    E também refletir sobre o fragmento 'frag_beta'.
    """
    
    test_text_2 = """
    O modelo base respondeu mal. Tente perguntar ao fragmento 'frag_gamma' com [0.1, 0.9, 0.5].
    Se não funcionar, treinar fragmento 'frag_gamma' por -3 épocas. < Inválido
    """
    
    test_text_3 = "Nada de interessante aqui."

    print("\n--- Teste 1 ---")
    commands_1, metadata_1 = interpreter.interpret_knowledge(test_text_1)
    print(f"Comandos A3L extraídos (Fonte: {metadata_1.get('source')}):")
    for cmd in commands_1:
        print(f"- {cmd}")

    print("\n--- Teste 2 ---")
    commands_2, metadata_2 = interpreter.interpret_knowledge(test_text_2)
    print(f"Comandos A3L extraídos (Fonte: {metadata_2.get('source')}):")
    for cmd in commands_2:
        print(f"- {cmd}")
        
    print("\n--- Teste 3 ---")
    commands_3, metadata_3 = interpreter.interpret_knowledge(test_text_3)
    print(f"Comandos A3L extraídos (Fonte: {metadata_3.get('source')}):")
    for cmd in commands_3:
        print(f"- {cmd}")
        
    print("\n--- Teste 4 (Natural Language) ---")
    test_text_4 = "Analise o desempenho recente e me diga como melhorar a classificação de imagens de gatos."
    commands_4, metadata_4 = interpreter.interpret_knowledge(test_text_4)
    print(f"Comandos A3L extraídos (Fonte: {metadata_4.get('source')}):")
    if commands_4:
         for cmd in commands_4:
             print(f"- {cmd}")
    else:
         print("(Nenhum comando extraído - esperado se Professor não estiver conectado ou não responder com A3L)")

"""
This defines the KnowledgeInterpreterFragment class in its own file. 
It inherits from nn.Module for framework consistency but its core logic uses regex 
to find and convert text patterns into A3L command strings via the 
`interpret_knowledge` method. Includes basic examples and logging.
""" 