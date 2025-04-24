# a3x/core/execution/fragment_executor.py
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from a3x.core.context import SharedTaskContext, FragmentContext
from a3x.core.tool_registry import ToolRegistry
from a3x.core.llm_interface import LLMInterface

class FragmentExecutor:
    """
    Responsável pelo ciclo principal de execução de sub-tarefas (ReAct loop) de um Fragment,
    delegando interações com LLM, execução de ferramentas e otimização.
    """
    def __init__(self,
                 fragment_context: FragmentContext,
                 logger: logging.Logger,
                 llm_interface: Optional[LLMInterface] = None,
                 tool_registry: Optional[ToolRegistry] = None,
                 shared_task_context: Optional[SharedTaskContext] = None):
        self.ctx = fragment_context
        self.logger = logger
        self.llm_interface = llm_interface or getattr(fragment_context, 'llm_interface', None)
        self.tool_registry = tool_registry or getattr(fragment_context, 'tool_registry', None)
        self.shared_task_context = shared_task_context or getattr(fragment_context, 'shared_task_context', None)

    async def execute_task(self,
                          objective: str,
                          tools: Optional[List[Callable[[str, Dict[str, Any]], Awaitable[str]]]] = None,
                          context: Optional[Dict] = None,
                          max_iterations: int = 5) -> Any:
        """
        Executa uma tarefa usando o ciclo principal (ReAct loop) com o objetivo fornecido.
        """
        context = context or {}
        if tools is None:
            # Busca ferramentas baseadas nas skills do fragmento
            tools = []
            skills = getattr(self.ctx, 'current_skills', [])
            for skill in skills:
                try:
                    tool = self.tool_registry.get_tool(skill)
                    tools.append(tool)
                except Exception:
                    self.logger.warning(f"[FragmentExecutor] Tool for skill '{skill}' not found.")
        if not tools:
            return f"No tools available to execute objective: {objective}"
        return await self._default_execute(objective, tools, context)

    async def _default_execute(self,
                              objective: str,
                              tools: List[Callable[[str, Dict[str, Any]], Awaitable[str]]],
                              context: Dict) -> str:
        """
        Execução simples sem otimização: usa a primeira ferramenta disponível.
        """
        if tools:
            tool = tools[0]
            input_data = {"objective": objective, "context": context}
            if hasattr(self.ctx, 'shared_task_context') and self.ctx.shared_task_context:
                input_data["shared_task_context"] = self.ctx.shared_task_context
            return await tool(objective, input_data)
        return f"No tools available to execute objective: {objective}"

    async def _run_sub_task(self, sub_task: str, max_iterations: int = 5) -> Dict:
        """
        Executa o loop interno ReAct para o sub-task, planejando e agindo com LLM e ferramentas.
        """
        self.logger.info(f"[FragmentExecutor] Starting internal ReAct loop for sub-task: {sub_task}")
        history = []
        for iteration in range(max_iterations):
            self.logger.info(f"[FragmentExecutor] Iteration {iteration + 1}/{max_iterations} for sub-task: {sub_task}")
            # Build prompt for LLM with allowed skills
            allowed_tools = []
            skills = getattr(self.ctx, 'current_skills', [])
            for skill in skills:
                if skill in self.tool_registry._tools:
                    allowed_tools.append(self.tool_registry.get_tool(skill))
            messages = await self._build_worker_messages(sub_task, history, allowed_tools)
            # Get response from LLM
            try:
                response = await self.llm_interface.get_response(messages)
                self.logger.info(f"[FragmentExecutor] LLM response received for sub-task: {sub_task}")
                thought = response.get('thought', '')
                action = response.get('action', {})
                action_name = action.get('name', 'none')
                action_params = action.get('parameters', {})
                history.append({'role': 'assistant', 'content': response})
                if action_name == 'none':
                    self.logger.info(f"[FragmentExecutor] No action chosen for sub-task: {sub_task}. Concluding based on thought.")
                    return {'success': True, 'result': thought, 'sub_task': sub_task}
                # Execute the chosen action
                self.logger.info(f"[FragmentExecutor] Executing action {action_name} for sub_task: {sub_task}")
                try:
                    tool_func = self.tool_registry.get_tool(action_name)
                    observation = await tool_func(action_name, action_params)
                    self.logger.info(f"[FragmentExecutor] Action {action_name} executed with observation: {observation}")
                    history.append({'role': 'observation', 'content': observation})
                    if 'success' in observation and observation['success']:
                        self.logger.info(f"[FragmentExecutor] Sub-task {sub_task} completed successfully via action {action_name}.")
                        return {'success': True, 'result': observation, 'sub_task': sub_task}
                except Exception as e:
                    error_msg = f"Error executing action {action_name}: {str(e)}"
                    self.logger.error(error_msg)
                    history.append({'role': 'observation', 'content': error_msg})
            except Exception as e:
                error_msg = f"Error getting LLM response: {str(e)}"
                self.logger.error(error_msg)
                history.append({'role': 'error', 'content': error_msg})
                return {'success': False, 'error': error_msg, 'sub_task': sub_task}
        self.logger.warning(f"[FragmentExecutor] Max iterations reached for sub-task: {sub_task}. Returning last state.")
        return {'success': False, 'error': 'Max iterations reached', 'sub_task': sub_task, 'history': history}

    async def _build_worker_messages(self, objective: str, history: List[Dict], allowed_tools: List[Callable]) -> List[Dict]:
        """
        Constrói as mensagens para o LLM a partir do objetivo, histórico e ferramentas permitidas.
        """
        messages = [
            {"role": "system", "content": f"Objective: {objective}"},
            {"role": "system", "content": f"Allowed tools: {[t.__name__ for t in allowed_tools]}"}
        ]
        messages.extend(history)
        return messages

    async def execute_autonomous_loop(self, objective: str, max_cycles: int = 10) -> Dict:
        """
        Executa o loop autônomo do fragmento (pensar, agir, observar) até critério de parada.
        """
        self.logger.info(f"[FragmentExecutor] Iniciando loop autônomo para: {objective}")
        cycle_count = 0
        history = []
        result = {"success": False, "result": None, "objective": objective, "learnings": [], "cycles": 0}
        while cycle_count < max_cycles:
            cycle_count += 1
            self.logger.info(f"[FragmentExecutor] Cycle {cycle_count}/{max_cycles} for objective: {objective}")
            sub_task_result = await self._run_sub_task(objective)
            history.append(sub_task_result)
            if sub_task_result.get("success", False):
                self.logger.info(f"[FragmentExecutor] Objective {objective} achieved successfully in cycle {cycle_count}.")
                result["success"] = True
                result["result"] = sub_task_result.get("result")
                break
            else:
                error_msg = sub_task_result.get("error", "Unknown error")
                self.logger.warning(f"[FragmentExecutor] Cycle {cycle_count} failed with error: {error_msg}")
                # Fallbacks e modos experimentais poderiam ser implementados aqui
        result["cycles"] = cycle_count
        if not result["success"]:
            result["error"] = "Max cycles reached without achieving objective"
            self.logger.warning(f"[FragmentExecutor] Max cycles reached for objective: {objective}")
        self.logger.info(f"[FragmentExecutor] Autonomous loop completed for {objective} with success: {result['success']}")
        return result

    async def execute_task(self,
                          objective: str,
                          tools: Optional[List[Callable[[str, Dict[str, Any]], Awaitable[str]]]] = None,
                          context: Optional[Dict] = None,
                          max_iterations: int = 5) -> Any:
        """
        Executa uma tarefa usando o ciclo principal (ReAct loop) com o objetivo fornecido.
        """
        # Implementação simplificada do ciclo principal de execução (ReAct loop)
        context = context or {}
        tools = tools or []
        self.logger.info(f"[FragmentExecutor] Iniciando execução do objetivo: {objective}")
        result = None
        for iteration in range(max_iterations):
            self.logger.debug(f"[FragmentExecutor] Iteração {iteration+1} de {max_iterations}")
            # 1. Planejar (interação com LLM para decidir próxima ação)
            plan = await self._plan(objective, context)
            self.logger.debug(f"[FragmentExecutor] Plano retornado: {plan}")
            # 2. Agir (executar ferramenta/skill)
            if plan.get('tool'):
                tool_name = plan['tool']
                tool_args = plan.get('args', {})
                tool = self.tool_registry.get_tool(tool_name) if self.tool_registry else None
                if tool:
                    tool_result = await tool(objective, tool_args)
                    context['last_tool_result'] = tool_result
                else:
                    self.logger.warning(f"[FragmentExecutor] Ferramenta '{tool_name}' não encontrada.")
                    context['last_tool_result'] = f"Tool '{tool_name}' not found."
            else:
                self.logger.info(f"[FragmentExecutor] Nenhuma ferramenta selecionada, encerrando loop.")
                break
            # 3. Observar (atualizar contexto)
            result = context.get('last_tool_result')
            # Critério de parada simplificado (pode ser expandido)
            if plan.get('done', False):
                break
        self.logger.info(f"[FragmentExecutor] Execução concluída. Resultado: {result}")
        return result

    async def _plan(self, objective: str, context: Dict) -> Dict:
        """
        Simula uma chamada ao LLM para decidir o próximo passo (pensar/plan).
        """
        # Aqui seria feita a chamada ao LLM para decidir a próxima ação.
        # Para simplificação, retornamos um plano fictício.
        # Em produção, usar self.llm_interface.generate_plan(...) ou equivalente.
        return {'tool': None, 'args': {}, 'done': True}  # Placeholder

    async def execute_autonomous_loop(self,
                                      objective: str,
                                      max_cycles: int = 10) -> Any:
        """
        Executa o loop autônomo do fragmento (pensar, agir, observar) até critério de parada.
        """
        self.logger.info(f"[FragmentExecutor] Iniciando loop autônomo para: {objective}")
        context = {}
        for cycle in range(max_cycles):
            plan = await self._plan(objective, context)
            if plan.get('tool'):
                tool = self.tool_registry.get_tool(plan['tool']) if self.tool_registry else None
                if tool:
                    tool_result = await tool(objective, plan.get('args', {}))
                    context['last_tool_result'] = tool_result
                else:
                    context['last_tool_result'] = f"Tool '{plan['tool']}' not found."
            if plan.get('done', False):
                break
        return context.get('last_tool_result')
