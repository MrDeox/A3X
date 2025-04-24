import logging
from typing import Dict, Any, List
from a3x.fragments.base import BaseFragment, FragmentContext
from a3x.fragments.registry import fragment

logger = logging.getLogger(__name__)

@fragment(
    name="professor_llm",
    description="Fragmento responsável por interagir com o LLM para análise e geração de conhecimento",
    category="learning",
    skills=["llm_interaction", "knowledge_generation", "analysis"],
    managed_skills=["llm_interaction"]
)
class ProfessorLLMFragment(BaseFragment):
    """Fragmento que interage com o LLM para análise e geração de conhecimento."""
    
    async def execute(self, ctx: FragmentContext) -> Dict[str, Any]:
        """Executa a interação com o LLM."""
        try:
            # Verifica componentes necessários
            if not ctx.llm_interface:
                raise ValueError("LLM interface não disponível")
                
            # Obtém contexto atual
            context = await self._get_current_context(ctx)
            
            # Gera análise
            analysis = await self._generate_analysis(ctx, context)
            
            # Gera recomendações
            recommendations = await self._generate_recommendations(ctx, analysis)
            
            # Registra resultados
            await self._record_results(ctx, analysis, recommendations)
            
            return {
                "status": "success",
                "analysis": analysis,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.exception("Erro durante execução do ProfessorLLMFragment:")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _get_current_context(self, ctx: FragmentContext) -> Dict[str, Any]:
        """Obtém o contexto atual do sistema."""
        try:
            # Obtém estado dos fragments
            fragments_state = await self._get_fragments_state(ctx)
            
            # Obtém histórico recente
            recent_history = await ctx.memory_manager.get_recent_episodes(limit=50)
            
            # Obtém conhecimento relevante
            relevant_knowledge = await self._get_relevant_knowledge(ctx)
            
            return {
                "fragments_state": fragments_state,
                "recent_history": recent_history,
                "relevant_knowledge": relevant_knowledge
            }
            
        except Exception as e:
            logger.exception("Erro ao obter contexto atual:")
            return {
                "fragments_state": {},
                "recent_history": [],
                "relevant_knowledge": []
            }

    async def _get_fragments_state(self, ctx: FragmentContext) -> Dict[str, Any]:
        """Obtém estado atual dos fragments."""
        state = {}
        
        for fragment in ctx.fragment_registry.get_fragments():
            # Obtém métricas do fragment
            metrics = await self._get_fragment_metrics(ctx, fragment)
            
            # Obtém histórico recente
            history = await ctx.memory_manager.get_recent_episodes(
                filter_metadata={"fragment": fragment.name},
                limit=50
            )
            
            # Calcula estatísticas
            stats = self._calculate_fragment_stats(history)
            
            state[fragment.name] = {
                "metrics": metrics,
                "stats": stats,
                "is_active": fragment.is_active
            }
            
        return state

    async def _get_fragment_metrics(self, ctx: FragmentContext, fragment: Any) -> Dict[str, Any]:
        """Obtém métricas de um fragment específico."""
        try:
            # Obtém histórico de execução
            history = await ctx.memory_manager.get_recent_episodes(
                filter_metadata={"fragment": fragment.name},
                limit=100
            )
            
            if not history:
                return {
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0,
                    "error_rate": 0.0
                }
                
            # Calcula métricas
            success_count = sum(1 for h in history if h.get("status") == "success")
            error_count = sum(1 for h in history if h.get("status") == "error")
            total_count = len(history)
            
            execution_times = [h.get("execution_time", 0.0) for h in history]
            avg_execution_time = sum(execution_times) / len(execution_times)
            
            return {
                "success_rate": success_count / total_count,
                "avg_execution_time": avg_execution_time,
                "error_rate": error_count / total_count
            }
            
        except Exception as e:
            logger.exception(f"Erro ao obter métricas do fragment {fragment.name}:")
            return {
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "error_rate": 1.0
            }

    def _calculate_fragment_stats(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula estatísticas do histórico de um fragment."""
        if not history:
            return {
                "total_executions": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_input_size": 0,
                "avg_output_size": 0
            }
            
        # Conta execuções
        total_executions = len(history)
        success_count = sum(1 for h in history if h.get("status") == "success")
        error_count = sum(1 for h in history if h.get("status") == "error")
        
        # Calcula tamanhos médios
        input_sizes = [len(str(h.get("input", ""))) for h in history]
        output_sizes = [len(str(h.get("output", ""))) for h in history]
        
        return {
            "total_executions": total_executions,
            "success_count": success_count,
            "error_count": error_count,
            "avg_input_size": sum(input_sizes) / len(input_sizes),
            "avg_output_size": sum(output_sizes) / len(output_sizes)
        }

    async def _get_relevant_knowledge(self, ctx: FragmentContext) -> List[Dict[str, Any]]:
        """Obtém conhecimento relevante do sistema."""
        try:
            # Obtém conhecimento semântico
            semantic_knowledge = await ctx.memory_manager.search_semantic_memory(
                query="sistema evolução conhecimento",
                limit=20
            )
            
            # Obtém episódios relevantes
            relevant_episodes = await ctx.memory_manager.get_recent_episodes(
                filter_metadata={"type": "knowledge"},
                limit=30
            )
            
            return semantic_knowledge + relevant_episodes
            
        except Exception as e:
            logger.exception("Erro ao obter conhecimento relevante:")
            return []

    async def _generate_analysis(self, ctx: FragmentContext, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gera análise do sistema usando o LLM."""
        try:
            # Prepara prompt para análise
            prompt = self._prepare_analysis_prompt(context)
            
            # Obtém resposta do LLM
            response = await ctx.llm_interface.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Processa resposta
            analysis = self._process_llm_response(response)
            
            return analysis
            
        except Exception as e:
            logger.exception("Erro durante geração de análise:")
            return {
                "status": "error",
                "message": str(e)
            }

    def _prepare_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Prepara prompt para análise do sistema."""
        fragments_state = context.get("fragments_state", {})
        recent_history = context.get("recent_history", [])
        relevant_knowledge = context.get("relevant_knowledge", [])
        
        prompt = "Analise o estado atual do sistema A³X:\n\n"
        
        # Adiciona estado dos fragments
        prompt += "Estado dos Fragments:\n"
        for fragment_name, state in fragments_state.items():
            metrics = state.get("metrics", {})
            stats = state.get("stats", {})
            prompt += f"- {fragment_name}:\n"
            prompt += f"  * Taxa de sucesso: {metrics.get('success_rate', 0.0):.2f}\n"
            prompt += f"  * Tempo médio de execução: {metrics.get('avg_execution_time', 0.0):.2f}s\n"
            prompt += f"  * Taxa de erro: {metrics.get('error_rate', 0.0):.2f}\n"
            prompt += f"  * Total de execuções: {stats.get('total_executions', 0)}\n"
            
        # Adiciona histórico recente
        prompt += "\nHistórico Recente:\n"
        for event in recent_history[:5]:  # Limita a 5 eventos
            prompt += f"- {event.get('type', 'unknown')}: {event.get('description', '')}\n"
            
        # Adiciona conhecimento relevante
        prompt += "\nConhecimento Relevante:\n"
        for knowledge in relevant_knowledge[:5]:  # Limita a 5 itens
            prompt += f"- {knowledge.get('content', '')}\n"
            
        prompt += "\nBaseado nestas informações, forneça uma análise detalhada do estado do sistema, identificando pontos fortes, fracos e oportunidades de melhoria."
        
        return prompt

    def _process_llm_response(self, response: str) -> Dict[str, Any]:
        """Processa resposta do LLM."""
        try:
            # Aqui você implementaria a lógica para processar a resposta do LLM
            # e extrair informações estruturadas
            return {
                "status": "success",
                "analysis": response,
                "key_points": self._extract_key_points(response),
                "sentiment": self._analyze_sentiment(response)
            }
            
        except Exception as e:
            logger.exception("Erro ao processar resposta do LLM:")
            return {
                "status": "error",
                "message": str(e)
            }

    def _extract_key_points(self, response: str) -> List[str]:
        """Extrai pontos-chave da resposta do LLM."""
        # Implementação simplificada
        lines = response.split('\n')
        key_points = []
        
        for line in lines:
            if line.strip().startswith('-') or line.strip().startswith('*'):
                key_points.append(line.strip())
                
        return key_points

    def _analyze_sentiment(self, response: str) -> str:
        """Analisa o sentimento geral da resposta."""
        # Implementação simplificada
        positive_words = ["bom", "ótimo", "excelente", "positivo", "forte"]
        negative_words = ["ruim", "fraco", "problema", "negativo", "preocupante"]
        
        positive_count = sum(1 for word in positive_words if word in response.lower())
        negative_count = sum(1 for word in negative_words if word in response.lower())
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    async def _generate_recommendations(self, ctx: FragmentContext, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera recomendações baseadas na análise."""
        try:
            # Prepara prompt para recomendações
            prompt = self._prepare_recommendations_prompt(analysis)
            
            # Obtém resposta do LLM
            response = await ctx.llm_interface.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.7
            )
            
            # Processa recomendações
            recommendations = self._process_recommendations(response)
            
            return recommendations
            
        except Exception as e:
            logger.exception("Erro durante geração de recomendações:")
            return []

    def _prepare_recommendations_prompt(self, analysis: Dict[str, Any]) -> str:
        """Prepara prompt para geração de recomendações."""
        prompt = "Baseado na seguinte análise do sistema A³X, forneça recomendações específicas para melhorias:\n\n"
        
        prompt += f"Análise:\n{analysis.get('analysis', '')}\n\n"
        prompt += f"Pontos-chave:\n"
        for point in analysis.get('key_points', []):
            prompt += f"- {point}\n"
            
        prompt += f"\nSentimento geral: {analysis.get('sentiment', 'neutral')}\n\n"
        prompt += "Forneça recomendações específicas e acionáveis para melhorar o sistema, priorizando as mais importantes."
        
        return prompt

    def _process_recommendations(self, response: str) -> List[Dict[str, Any]]:
        """Processa recomendações da resposta do LLM."""
        try:
            recommendations = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    # Extrai prioridade se presente
                    priority = "medium"
                    if "alta prioridade" in line.lower():
                        priority = "high"
                    elif "baixa prioridade" in line.lower():
                        priority = "low"
                        
                    recommendations.append({
                        "content": line.lstrip('-* ').strip(),
                        "priority": priority
                    })
                    
            return recommendations
            
        except Exception as e:
            logger.exception("Erro ao processar recomendações:")
            return []

    async def _record_results(self, ctx: FragmentContext, analysis: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> None:
        """Registra resultados da análise e recomendações."""
        try:
            # Registra análise
            await ctx.memory_manager.record_episodic_event(
                context="llm_analysis",
                action="system_analysis",
                outcome=analysis,
                metadata={
                    "type": "analysis",
                    "timestamp": "now",
                    "source": "professor_llm"
                }
            )
            
            # Registra recomendações
            for rec in recommendations:
                await ctx.memory_manager.record_episodic_event(
                    context="llm_recommendation",
                    action="system_improvement",
                    outcome=rec,
                    metadata={
                        "type": "recommendation",
                        "priority": rec.get("priority", "medium"),
                        "timestamp": "now",
                        "source": "professor_llm"
                    }
                )
                
        except Exception as e:
            logger.exception("Erro ao registrar resultados:") 