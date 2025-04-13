import logging
from typing import List, Dict, Any, Optional
from a3x.core.llm_interface import call_llm
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

async def propose_and_generate_skill(skill_name: str, context: Optional[Dict[str, Any]] = None, llm_url: Optional[str] = None) -> str:
    """
    Usa o LLM para propor e gerar o código inicial de uma nova skill.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "Você é um gerador de skills Python para o agente A³X. "
                "Dado o nome de uma skill e contexto, gere um arquivo Python funcional "
                "com a assinatura padrão de uma skill A³X, incluindo docstring e parâmetros esperados. "
                "Não inclua explicações, apenas o código Python puro."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Nome da skill: {skill_name}\n"
                f"Contexto: {context}\n"
            ),
        },
    ]
    code = ""
    try:
        logger.info(f"[SkillAutogen] Solicitando geração de skill '{skill_name}' ao LLM...")
        async for chunk in call_llm(prompt, llm_url=llm_url, stream=False):
            code += chunk
        # Extrai apenas o código Python (remove markdown se houver)
        if "```python" in code:
            code = code.split("```python")[-1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[-1].split("```")[0].strip()
        return code
    except Exception as e:
        logger.error(f"[SkillAutogen] Erro ao gerar skill '{skill_name}': {e}")
        return ""

def save_skill_file(skill_name: str, code: str, skills_dir: Path = Path("a3x/skills/auto_generated")) -> Path:
    """
    Salva o código da skill em um arquivo Python no diretório de skills auto-geradas.
    """
    skills_dir.mkdir(parents=True, exist_ok=True)
    file_path = skills_dir / f"{skill_name}.py"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    logger.info(f"[SkillAutogen] Skill '{skill_name}' salva em {file_path}")
    return file_path

async def autogen_skills_from_heuristics(heuristics: List[Dict[str, Any]], llm_url: Optional[str] = None) -> List[str]:
    """
    Pipeline completo: detecta gaps, gera código e salva novas skills.
    """
    generated = []
    missing_skills = detect_missing_skills(heuristics)
    for skill in missing_skills:
        code = await propose_and_generate_skill(skill, context={"heuristics": heuristics}, llm_url=llm_url)
        if code:
            save_skill_file(skill, code)
            generated.append(skill)
            # Log heurística de autogeração
            try:
                plan_id = f"plan-skill-autogen-{skill}"
                execution_id = f"exec-skill-autogen-{skill}"
                heuristic = {
                    "type": "skill_autogen",
                    "skill_name": skill,
                    "code_snippet": code[:200],
                }
                log_heuristic_with_traceability(heuristic, plan_id, execution_id, validation_status="pending_manual")
            except Exception as log_err:
                logger.warning(f"[SkillAutogen] Falha ao registrar heurística de autogeração: {log_err}")
    return generated