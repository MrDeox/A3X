import asyncio
import logging
import json # Import json for potential parsing if needed later

from a3x.core.context import SharedTaskContext
from a3x.core.tool_registry import ToolRegistry
from a3x.fragments.basic_fragments import PlannerFragment, FinalAnswerProvider
from a3x.fragments.base import FragmentDef
from a3x.core.llm_interface import LLMInterface
from a3x.core.server_manager import ServerManager # Import ServerManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_sandbox_test():
    """Run a test of Sandbox Mode with multiple Fragments interacting, managing the LLM server."""
    logger.info("Starting Sandbox Mode test for A³X system with multiple Fragments.")
    
    server_manager = ServerManager()
    llm_server_name = "llama" # Assuming 'llama' is the configured name for the LLM server

    # Use ServerManager to manage the LLM server lifecycle
    try:
        async with server_manager.managed_server(llm_server_name):
            logger.info(f"{llm_server_name.upper()} server started via ServerManager.")
            
            # Initialize shared components *after* server is confirmed running
            project_goal = "Achieve autonomous complex problem-solving and self-improvement"
            task_context = SharedTaskContext(task_id="sandbox_guided_test_001", initial_objective=project_goal)
            tool_registry = ToolRegistry()
            llm_interface = LLMInterface() # LLMInterface finds the URL automatically
            
            # Define and instantiate multiple Fragments
            planner_def = FragmentDef(
                name="Planner",
                fragment_class=PlannerFragment,
                description="Plans tasks.",
                skills=["read_file"]
            )
            planner_fragment = PlannerFragment(tool_registry)

            answer_provider_def = FragmentDef(
                name="AnswerProvider",
                fragment_class=FinalAnswerProvider,
                description="Provides final answers.",
                skills=[]
            )
            answer_provider_fragment = FinalAnswerProvider(tool_registry)

            fragments = [planner_fragment, answer_provider_fragment]
            
            # Run the Fragments concurrently in Sandbox Mode
            logger.info(f"Running {len(fragments)} Fragments concurrently in Sandbox Mode.")
            tasks = []
            for fragment in fragments:
                tasks.append(
                    asyncio.create_task(
                        fragment.run_sandbox_mode(
                            shared_task_context=task_context,
                            tool_registry=tool_registry,
                            llm_interface=llm_interface,
                            max_interactions=3
                        ),
                        name=f"sandbox_{fragment.get_name()}"
                    )
                )

            # Wait for all fragments to complete their sandbox sessions
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log the overall results and dialogue
            logger.info("Sandbox Mode interactions completed for all Fragments.")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                     logger.error(f"Fragment {fragments[i].get_name()} failed: {result}")
                else:
                     logger.info(f"Fragment {fragments[i].get_name()} finished with summary: {result.get('summary', 'No summary')}")

            logger.info("--- Full Sandbox Dialogue History --- ")
            full_dialogue = await task_context.get_sandbox_messages()
            for sender, message_content in full_dialogue:
                dialogue = message_content.get('dialogue', '(no dialogue)')
                logger.info(f"[{sender}]: {dialogue}")
            logger.info("--- End of Dialogue History --- ")

            return results
            
    except RuntimeError as e:
         logger.error(f"Failed to start managed server '{llm_server_name}': {e}")
         return None # Indicate failure due to server start issue
    except Exception as e:
         logger.exception(f"An unexpected error occurred during the sandbox test:")
         return None
    # ServerManager context manager ensures server shutdown here

if __name__ == "__main__":
    results = asyncio.run(run_sandbox_test())
    if results:
        logger.info("Sandbox Mode multi-fragment test completed successfully.")
    else:
         logger.error("Sandbox Mode multi-fragment test failed (likely server issue).") 