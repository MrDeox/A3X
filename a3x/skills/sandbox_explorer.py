import logging
from typing import Dict, Optional, Any

from a3x.core.skills import skill
from a3x.core.context import SharedTaskContext, Context
from a3x.core.context_accessor import ContextAccessor
from a3x.skills.execute_code import execute_code

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize a ContextAccessor instance for use in skills
_context_accessor = ContextAccessor()

class SandboxExplorerSkill:
    """Skill para o Modo Sandbox Autônomo (Modo Artista), que permite ao A³X explorar soluções criativas
de forma independente, gerando e testando código em um ambiente seguro."""

    def __init__(self):
        """Inicializa o SandboxExplorerSkill."""
        logger.info("SandboxExplorerSkill inicializado.")

    @skill(
        name="explore_sandbox",
        description="Gera e testa código ou hipóteses de forma autônoma em um ambiente sandbox seguro.",
        parameters={
            "objective": {"type": Optional[str], "default": None, "description": "Objetivo geral para orientar a exploração (opcional)."},
            "max_attempts": {"type": int, "default": 3, "description": "Número máximo de tentativas de geração e teste (padrão: 3)."},
            "shared_task_context": {"type": "Optional[a3x.core.context.SharedTaskContext]", "description": "O contexto compartilhado para acessar dados relacionados à tarefa.", "optional": True}
        }
    )
    async def explore_sandbox(
        self,
        context: Context,
        objective: Optional[str] = None,
        max_attempts: int = 3,
        shared_task_context: Optional[SharedTaskContext] = None
    ) -> Dict[str, Any]:
        """
        Gera e testa código ou hipóteses de forma autônoma em um ambiente sandbox seguro.
        Registra os resultados no SharedTaskContext para revisão e aprendizado.

        Args:
            context: O contexto de execução fornecido pelo agente.
            objective: Objetivo geral para orientar a exploração (opcional).
            max_attempts: Número máximo de tentativas de geração e teste.
            shared_task_context: O contexto compartilhado para a tarefa atual.

        Returns:
            Dict[str, Any]: Dicionário padronizado com os resultados da exploração.
        """
        log_prefix = "[explore_sandbox]"
        logger.info(f"{log_prefix} Iniciando exploração autônoma no modo sandbox.")
        
        # Atualizar o context accessor se shared_task_context for fornecido
        if shared_task_context:
            _context_accessor.set_context(shared_task_context)
            logger.info(f"{log_prefix} ContextAccessor atualizado com task ID: {shared_task_context._task_id}")
        
        # Determinar o objetivo da exploração
        if objective:
            exploration_objective = objective
        else:
            exploration_objective = _context_accessor.get_task_objective() or "Explorar soluções criativas para problemas gerais."
        logger.info(f"{log_prefix} Objetivo da exploração: {exploration_objective}")
        
        # Limitar o número de tentativas
        if max_attempts < 1:
            max_attempts = 1
        elif max_attempts > 5:
            max_attempts = 5
            logger.warning(f"{log_prefix} max_attempts ajustado para o limite máximo de 5.")
        
        results = []
        successful_experiments = 0
        
        for attempt in range(max_attempts):
            logger.info(f"{log_prefix} Tentativa {attempt + 1} de {max_attempts}")
            
            # Gerar código ou hipótese (implementação inicial simples)
            generated_code = self._generate_code(exploration_objective, attempt)
            logger.debug(f"{log_prefix} Código gerado: {generated_code[:100]}...")
            
            # Executar o código no sandbox seguro
            exec_result = execute_code(
                context=context,
                code=generated_code,
                language="python",
                timeout=30,
                shared_task_context=shared_task_context
            )
            
            # Analisar o resultado da execução
            result_status = exec_result.get("status", "error")
            result_data = exec_result.get("data", {})
            stdout = result_data.get("stdout", "")
            stderr = result_data.get("stderr", "")
            returncode = result_data.get("returncode", -1)
            message = result_data.get("message", "Resultado desconhecido.")
            
            # Determinar se o experimento foi bem-sucedido (simples: status success e returncode 0)
            is_success = result_status == "success" and returncode == 0
            if is_success:
                successful_experiments += 1
            
            # Registrar o resultado no formato padronizado
            experiment_result = {
                "attempt": attempt + 1,
                "generated_code": generated_code,
                "status": result_status,
                "success": is_success,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": returncode,
                "message": message
            }
            results.append(experiment_result)
            logger.info(f"{log_prefix} Resultado da tentativa {attempt + 1}: {result_status} (Sucesso: {is_success})")
            
            # Registrar no SharedTaskContext se disponível
            if shared_task_context:
                context_key = f"sandbox_experiment_{attempt + 1}"
                tags = ["sandbox_experiment", "sandbox_success" if is_success else "sandbox_failure"]
                shared_task_context.set(
                    key=context_key,
                    value=experiment_result,
                    source="sandbox_explorer",
                    tags=tags,
                    metadata={"objective": exploration_objective, "attempt_count": max_attempts}
                )
                logger.debug(f"{log_prefix} Resultado da tentativa {attempt + 1} registrado no contexto com chave '{context_key}'.")
            
            # Parar se tivermos um sucesso (ou ajustar lógica conforme necessário)
            if is_success and successful_experiments >= 1:
                logger.info(f"{log_prefix} Experimento bem-sucedido encontrado na tentativa {attempt + 1}. Encerrando exploração.")
                break
        
        # Resumo dos resultados
        summary_message = f"Exploração concluída: {successful_experiments} experimento(s) bem-sucedido(s) em {len(results)} tentativa(s)."
        logger.info(f"{log_prefix} {summary_message}")
        
        # Retornar resultado padronizado
        return {
            "status": "success" if successful_experiments > 0 else "partial_success",
            "action": "sandbox_exploration_completed",
            "data": {
                "objective": exploration_objective,
                "total_attempts": len(results),
                "successful_experiments": successful_experiments,
                "results": results,
                "message": summary_message
            }
        }

    def _generate_code(self, objective: str, attempt: int) -> str:
        """
        Gera código Python com base no objetivo da exploração e no número da tentativa.
        Implementação inicial simples: retorna um código de exemplo.
        Futuramente, pode ser expandido para usar heurísticas ou modelos de geração.

        Args:
            objective: O objetivo da exploração.
            attempt: O número da tentativa atual (0-based).

        Returns:
            str: Código Python gerado para teste.
        """
        # Implementação inicial: código simples baseado no objetivo e tentativa
        # Futuramente, isso pode ser substituído por lógica mais avançada
        code_templates = [
            f"print('Teste de exploração {attempt + 1}: {objective[:50]}...')\n# Código gerado para teste simples.\nresult = 42\nprint('Resultado: ', result)",
            f"print('Teste de exploração {attempt + 1}: Tentativa de cálculo.')\n# Tentativa de resolver um problema matemático simples.\nx = 10\ny = 20\nresult = x + y\nprint('Resultado da soma: ', result)",
            f"print('Teste de exploração {attempt + 1}: Teste de lógica.')\n# Teste de lógica condicional.\nvalue = 15\nif value > 10:\n    print('Valor maior que 10.')\nelse:\n    print('Valor menor ou igual a 10.')"
        ]
        
        # Escolher um template baseado na tentativa
        template_index = attempt % len(code_templates)
        return code_templates[template_index]

# Instanciar a classe para que os métodos sejam registrados pelo decorador
sandbox_explorer = SandboxExplorerSkill() 