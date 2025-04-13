import logging
from typing import List, Dict, Any, Optional
from a3x.core.llm_interface import LLMInterface
from a3x.core.learning_logs import log_heuristic_with_traceability
from pathlib import Path

logger = logging.getLogger(__name__)

def detect_missing_skills(heuristics: List[Dict[str, Any]]) -> List[str]:
    """
    Analisa heurísticas para identificar skills ausentes ou gaps recorrentes.
    """
    missing_skills = []
    for h in heuristics:
        if h.get("type") in ("missing_skill_attempt", "parsing_fallback"):
            skill = h.get("skill_name") or h.get("action_inferred")
            if skill and skill not in missing_skills:
                missing_skills.append(skill)
    return missing_skills

async def propose_and_generate_skill(
    skill_name: str, 
    llm_interface: LLMInterface,
    context: Optional[Dict[str, Any]] = None, 
) -> str:
    """
    Usa o LLM para propor e gerar o código inicial de uma nova skill.
    """
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "Você é um gerador de skills Python para o agente A³X. "
                "Dado o nome de uma skill e contexto, gere um arquivo Python funcional "
                "com a assinatura padrão de uma skill A³X (@skill decorator, function signature), incluindo docstring e parâmetros esperados. "
                "Responda SOMENTE com o código Python bruto. Não inclua explicações, markdown (```python ...```), ou qualquer outro texto."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Nome da skill: {skill_name}\n"
                f"Contexto (e.g., how it was used or needed): {context}\n"
                f"Gere o código Python completo para esta skill."
            ),
        },
    ]
    code = ""
    try:
        logger.info(f"[SkillAutogen] Solicitando geração de skill '{skill_name}' ao LLM...")
        async for chunk in llm_interface.call_llm(
            messages=prompt_messages, 
            stream=False
        ):
            code += chunk
        
        if not (code.strip().startswith("import") or code.strip().startswith("def") or code.strip().startswith("@skill")):
             logger.warning(f"[SkillAutogen] LLM response for '{skill_name}' might not be Python code: {code[:100]}...")
             return ""
        else:
             code_lines = code.splitlines()
             if code_lines and code_lines[0].strip().startswith("```"):
                  code_lines.pop(0)
             if code_lines and code_lines[-1].strip() == "```":
                  code_lines.pop(-1)
             code = "\n".join(code_lines).strip()

        if not code:
            logger.warning(f"[SkillAutogen] LLM returned empty code for skill '{skill_name}'.")
            return ""
            
        return code
    except Exception as e:
        logger.error(f"[SkillAutogen] Erro ao gerar skill '{skill_name}': {e}", exc_info=True)
        return ""

def save_skill_file(skill_name: str, code: str, skills_dir: Path = Path("a3x/skills/auto_generated")) -> Path:
    """
    Salva o código da skill em um arquivo Python no diretório de skills auto-geradas.
    """
    skills_dir.mkdir(parents=True, exist_ok=True)
    safe_skill_name = "".join(c if c.isalnum() or c == '_' else '' for c in skill_name).lower()
    if not safe_skill_name:
        safe_skill_name = "unnamed_skill"
    file_path = skills_dir / f"{safe_skill_name}.py"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    logger.info(f"[SkillAutogen] Skill '{skill_name}' salva em {file_path}")
    return file_path

async def autogen_skills_from_heuristics(
    heuristics: List[Dict[str, Any]], 
    llm_interface: LLMInterface
) -> List[str]:
    """
    Pipeline completo: detecta gaps, gera código e salva novas skills.
    """
    generated = []
    missing_skills = detect_missing_skills(heuristics)
    for skill in missing_skills:
        code = await propose_and_generate_skill(
            skill,
            llm_interface=llm_interface,
            context={"heuristics": heuristics}
        )
        if code:
            save_skill_file(skill, code)
            generated.append(skill)
            # Log heurística de autogeração
            try:
                plan_id = f"plan-skill-autogen-{skill}"
                execution_id = f"exec-skill-autogen-{skill}"
                heuristic_data = {
                    "type": "skill_autogen",
                    "skill_name": skill,
                    "code_snippet": code[:200],
                }
                log_heuristic_with_traceability(heuristic_data, plan_id, execution_id, validation_status="pending_manual")
            except Exception as log_err:
                logger.warning(f"[SkillAutogen] Falha ao registrar heurística de autogeração para '{skill}': {log_err}")
    return generated