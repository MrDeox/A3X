import logging
import json
import os
import re
from typing import Dict, Any, Optional
from datetime import datetime

from a3x.core.skills import skill
from a3x.core.llm_interface import LLMInterface
from a3x.core.config import PROJECT_ROOT
from a3x.core.agent import _ToolExecutionContext
from a3x.core.skills import SkillContext

logger = logging.getLogger(__name__)

PERSONA_PROMPT = """
CONTEXT: Você é um gerador de personas expert em criar perfis fictícios detalhados e atraentes para um público adulto (+18), focando em nichos específicos (como 'alt girl', 'goth', 'kawaii', etc.). O objetivo é criar uma base sólida para geração de imagens e conteúdo NSFW posteriormente. Evite descrições genéricas.

TAREFA: Gere uma persona fictícia única e atraente com base nos seguintes critérios. Seja criativo e detalhista.

REQUISITOS DE SAÍDA:
- Retorne **SOMENTE** um objeto JSON válido, sem nenhum texto antes ou depois.
- O JSON deve ter as seguintes chaves:
  - "persona_name": (string) Nome único e cativante para a persona (ex: "Seraphina Glitch", "Nyx Ravenwood").
  - "bio": (string) Biografia curta (2-4 frases) descrevendo a personalidade, gostos e talvez um toque de mistério ou desejo. Deve ser envolvente e sugerir o tom NSFW.
  - "tags": (list[string]) Lista de 5-10 tags relevantes para o nicho e apelo visual (ex: ["alt girl", "tattoos", "piercings", "fishnets", "choker", "goth makeup", "dark aesthetic", "provocative"]).
  - "visual_prompt": (string) Prompt detalhado para gerar uma imagem da persona no Stable Diffusion. Deve incluir aparência física (cabelo, olhos, corpo), roupas específicas (ou ausência delas), acessórios, cenário/ambiente, iluminação e o estilo artístico desejado. **Crucial:** Inclua termos que reforcem o NSFW e a qualidade (ex: "photorealistic", "sharp focus", "detailed skin texture", "seductive pose", "lingerie", "NSFW"). Pense em como o SD interpretará isso.
  - "style_reference": (string) Referência de estilo visual ou artístico (ex: "Estilo Greg Rutkowski com iluminação dramática", "Fotografia boudoir dark", "Arte digital estilo Sakimichan").

EXEMPLO (NÃO COPIAR DIRETAMENTE, USE COMO GUIA):
{
  "persona_name": "Lilith Vesper",
  "bio": "Uma sombra elegante que dança na linha tênue entre o sedutor e o sinistro. Lilith adora noites chuvosas, livros antigos e o toque frio do couro. Há um convite silencioso em seu olhar que poucos conseguem resistir.",
  "tags": ["goth", "vampire aesthetic", "pale skin", "dark lipstick", "lace", "velvet", "elegant", "mysterious", "sensual", "NSFW"],
  "visual_prompt": "Fotografia ultra-realista de Lilith Vesper, uma mulher gótica pálida com longos cabelos negros e olhos vermelhos penetrantes. Ela usa um corpete de renda preta e uma saia longa de veludo com uma fenda alta, revelando uma cinta-liga. Pose sedutora em uma biblioteca escura e empoeirada, iluminada por velas. Foco nítido, textura de pele detalhada, atmosfera sombria e sensual, NSFW.",
  "style_reference": "Fotografia dark fantasy com iluminação chiaroscuro"
}

PERSONA A GERAR:
Crie uma nova persona AGORA, seguindo estritamente o formato JSON acima.
"""

PERSONAS_DIR = os.path.join("a3x", "memory", "personas")
os.makedirs(PERSONAS_DIR, exist_ok=True)


def extract_json_from_markdown(text: str) -> Optional[Dict[str, Any]]:
    """Extrai um bloco de código JSON de uma string, mesmo que esteja dentro de ```json ... ```."""
    # Regex to find JSON block potentially wrapped in markdown code fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # If no markdown fence, assume the whole string might be JSON, or find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_str = text[start:end+1]
        else:
            logger.warning("Could not find JSON block in the text.")
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}\nJSON string attempted: {json_str}")
        return None

@skill(
    name="persona_generator",
    description="Generates a detailed buyer persona based on a niche description and target audience.",
    parameters={
        "context": {"type": SkillContext, "description": "Execution context for LLM access."},
        "niche": {"type": str, "description": "Description of the niche market."},
        "target_audience": {"type": Optional[str], "default": None, "description": "Specific description of the target audience within the niche."},
        "num_personas": {"type": Optional[int], "default": 1, "description": "Number of distinct personas to generate (default: 1)."}
    }
)
async def generate_persona(
    context: SkillContext,
    niche: str,
    target_audience: Optional[str] = None,
    num_personas: Optional[int] = 1
) -> Dict[str, Any]:
    """
    Generates a fictional +18 persona profile using an LLM call and saves it to a JSON file.

    Args:
        niche: The specific niche for the persona.
        platform: The target platform.
        number_of_personas: How many personas to generate (currently logic generates 1).
        ctx: The skill execution context, providing access to llm_call and logger.

    Returns:
        A dictionary indicating success or failure, including the path to the saved persona file.
    """
    logger = context.logger
    llm_interface = context.llm_interface
    
    if not llm_interface:
        logger.error("LLMInterface not found in execution context for persona generation.")
        return {"status": "error", "message": "Internal error: LLMInterface missing."}
        
    logger.info(f"Starting persona generation for niche '{niche}' on platform '{target_audience}'...")

    full_response_content = ""
    json_data = None

    try:
        logger.debug("Calling LLM with streaming enabled...")
        async for chunk in llm_interface.call_llm(
            messages=[{"role": "user", "content": PERSONA_PROMPT}],
            stream=True
        ):
             if isinstance(chunk, str):
                 full_response_content += chunk
             else:
                 logger.warning(f"Received non-string chunk from call_llm: {type(chunk)}")

        logger.info("LLM stream finished. Attempting to extract JSON from accumulated text.")
        logger.debug(f"Full response content length: {len(full_response_content)}")
        
        json_data = extract_json_from_markdown(full_response_content)

        if json_data is None:
             logger.error("Failed to extract valid JSON persona data from LLM response.")
             logger.debug(f"Full LLM Response Content:\n{full_response_content}")
             return {"status": "error", "message": "Failed to get valid JSON persona data from LLM."}

        # Validate required keys
        required_keys = ["persona_name", "bio", "tags", "visual_prompt", "style_reference"]
        if not all(key in json_data for key in required_keys):
            logger.error(f"Missing required keys in generated persona JSON. Found: {json_data.keys()}")
            return {"status": "error", "message": f"Missing required keys. Required: {required_keys}"}

        persona_name_slug = re.sub(r'\W+', '-', json_data["persona_name"]).lower()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{persona_name_slug}-{timestamp}.json"
        filepath = os.path.join(PERSONAS_DIR, filename)

        logger.info(f"Saving generated persona to: {filepath}")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Persona '{json_data['persona_name']}' saved successfully.")
            return {"status": "success", "persona_path": filepath, "persona_name": json_data["persona_name"]}
        except IOError as e:
            logger.error(f"Failed to save persona file to {filepath}: {e}")
            return {"status": "error", "message": f"Failed to save persona file: {e}"}

    except Exception as e:
        logger.exception("An unexpected error occurred during persona generation:")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

@skill(
    name="generate_persona_profile",
    description="Gera um perfil de persona detalhado para um público-alvo específico, incluindo necessidades, dores e motivações.",
    parameters={
        "target_audience": (str, ...),
        "product_description": (str, ...)
    }
)
async def generate_persona_profile(
    target_audience: str, 
    product_description: str, 
    ctx: _ToolExecutionContext
) -> str:
    """
    Generates a detailed persona profile using an LLM based on target audience and product.

    Args:
        target_audience: The target audience for the persona.
        product_description: The product description for the persona.
        ctx: The skill execution context, providing access to llm_call.

    Returns:
        A string indicating success or failure.
    """
    logger = ctx.logger
    llm_interface = ctx.llm_interface
    
    if not llm_interface:
        logger.error("LLMInterface not found in execution context for persona profile generation.")
        return json.dumps({"status": "error", "message": "Internal error: LLMInterface missing."})
        
    logger.info("Starting persona profile generation...")

    full_response_content = ""
    prompt_content = f"Generate a detailed persona profile for a {target_audience} audience, considering their needs, pains, and motivations. The product description is: {product_description}. Output ONLY the JSON profile."
    
    try:
        logger.debug("Calling LLM with streaming enabled...")
        async for chunk in llm_interface.call_llm(
            messages=[{"role": "user", "content": prompt_content}],
            stream=True
        ):
             if isinstance(chunk, str):
                 full_response_content += chunk
             else:
                 logger.warning(f"Received non-string chunk from call_llm: {type(chunk)}")

        logger.info("LLM stream finished. Attempting to extract JSON from accumulated text.")
        logger.debug(f"Full response content length: {len(full_response_content)}")
        
        json_data = extract_json_from_markdown(full_response_content)

        if json_data is None:
             logger.error("Failed to extract valid JSON persona profile data from LLM response.")
             logger.debug(f"Full LLM Response Content:\n{full_response_content}")
             return json.dumps({"status": "error", "message": "Failed to get valid JSON persona profile data from LLM."})

        logger.info(f"Persona profile generated successfully for target: {target_audience}")
        profile_name = json_data.get("persona_name", "Unnamed Profile")
        return json.dumps({"status": "success", "profile_name": profile_name, "profile_data": json_data})

    except Exception as e:
        logger.exception("An unexpected error occurred during persona profile generation:")
        return json.dumps({"status": "error", "message": f"An unexpected error occurred: {e}"})

"""
# Example usage (for testing purposes, typically called by the agent)
async def main_test():
    # Mock context for testing
    class MockContext:
        async def llm_call(self, prompt):
            # Simulate LLM response with potential Markdown block
            print("--- MOCK LLM CALL --- Prompt received: --- ")
            # print(prompt) # Can be very long
            print("-----------------------")
            return "```json\n{\n  \"persona_name\": \"Nyx Umbra\",\n  \"bio\": \"Enigma envolta em sombras e cetim, Nyx atrai com um olhar que promete prazeres proibidos. Ela coleciona segredos sussurrados e o calor de corpos entrelaçados na escuridão.\",\n  \"tags\": [\"gothic\", \"succubus aesthetic\", \"latex\", \"horns\", \"demon girl\", \"dark fantasy\", \"alluring\", \"teasing\", \"NSFW\"],\n  \"visual_prompt\": \"Fotografia hiper-realista de Nyx Umbra, uma súcubo gótica com pele pálida, longos cabelos negros e pequenos chifres curvados. Veste um body de látex preto decotado e meias arrastão. Olhar convidativo e lascivo para a câmera, levemente inclinada sobre um altar de pedra em ruínas sob a luz da lua. Foco perfeito, detalhes intrincados na textura do látex e da pele, iluminação ambiente suave, clima erótico explícito, NSFW.\",\n  \"style_reference\": \"Estilo de arte de Brom misturado com fotografia erótica dark.\"\n}\n```"

    mock_ctx = MockContext()
    result = await generate_persona(mock_ctx)
    print("--- RESULT ---")
    print(result)
    print("--------------")

if __name__ == "__main__":
    import asyncio
    # Configure logging for standalone test run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    # asyncio.run(main_test()) # Uncomment to run standalone test
    print("Persona Generator Skill Module Loaded. Use agent CLI or import to run.")

""" 