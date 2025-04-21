import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Any

# A³X Core Components
from a3x.core.tool_registry import ToolRegistry
from a3x.core.context import SharedTaskContext, FragmentContext
from a3x.fragments.base import FragmentDef
from a3x.fragments.registry import FragmentRegistry # Might be needed later

# Fragments
from a3x.fragments.structure_auto_refactor import StructureAutoRefactorFragment
from a3x.fragments.mutator import MutatorFragment
from a3x.fragments.anomaly_detector import AnomalyDetectorFragment

# Skills/Tools (or mocks if needed, though registry handles them)
# from a3x.skills.file_manager import FileManagerSkill 
# ... other skills ...

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmergenceRunner")

async def main():
    logger.info("--- Starting Emergence Test Runner ---")

    # 1. Initialize Shared Components
    message_queue = asyncio.Queue()
    tool_registry = ToolRegistry() # Shared registry for all fragments
    # TODO: Register actual skills or mocks here if not using fixtures
    # Example: tool_registry.register_tool(...)
    
    shared_context = SharedTaskContext(
        task_id="emergence_test_001",
        initial_objective="Improve the structure and resilience of the A³X system itself."
    )
    shared_context.message_queue = message_queue # Attach queue for message passing

    logger.info(f"Initialized shared context for Task ID: {shared_context.task_id}")
    logger.info(f"Initial Objective: {shared_context.initial_objective}")

    # 2. Instantiate Fragments
    fragments = []
    fragment_defs = [
        FragmentDef(name="StructureAutoRefactor", fragment_class=StructureAutoRefactorFragment, description="Handles code structure changes", category="Refactoring", skills=["generate_module_from_directive", "write_file"]), # Add relevant skills
        FragmentDef(name="Mutator", fragment_class=MutatorFragment, description="Mutates code based on failures", category="Correction", skills=["read_file", "modify_code", "write_file"]), # Add relevant skills
        FragmentDef(name="AnomalyDetector", fragment_class=AnomalyDetectorFragment, description="Detects system anomalies", category="Monitoring", skills=[]), # No active skills needed
        # Add other fragments like Planner, Evaluator later
    ]

    for frag_def in fragment_defs:
        try:
            # Each fragment gets the shared tool registry
            fragment_instance = frag_def.fragment_class(fragment_def=frag_def, tool_registry=tool_registry)
            # Provide context access (important!)
            fragment_instance.set_context(shared_context) 
            fragments.append(fragment_instance)
            logger.info(f"Instantiated Fragment: {frag_def.name}")
        except Exception as e:
            logger.error(f"Failed to instantiate fragment {frag_def.name}: {e}", exc_info=True)
            return # Stop if core fragments fail

    # 3. Inject Initial Objective / Seed Message
    initial_message = {
        "type": "system_broadcast",
        "sender": "EmergenceRunner",
        "content": {"message": "System startup. Initial objective active.", "objective": shared_context.initial_objective}
    }
    await message_queue.put(initial_message)
    logger.info("Injected initial system message and objective into the queue.")

    # --- Implement Step 2: Fragment Execution Loops --- #
    tasks = []
    for fragment in fragments:
        # Create a dedicated context for each fragment instance if needed,
        # but here they primarily interact via shared_context and queue
        fragment_context = FragmentContext(shared_context, tool_registry, fragment.get_name())
        task = asyncio.create_task(
            run_fragment_loop(fragment, message_queue, fragment_context),
            name=f"{fragment.get_name()}_loop"
        )
        tasks.append(task)
        logger.info(f"Created and started task for Fragment: {fragment.get_name()}")

    # --- TODO: Implement Step 3 (Feedback/Reward Mechanism) --- #
    logger.info("Feedback/Reward mechanism needs implementation.")

    # Keep runner alive by waiting for tasks (or a timeout/condition)
    if tasks:
        # Wait for all fragment tasks to complete (they won't in this forever-loop setup unless cancelled)
        # Or run for a fixed duration for testing
        try:
            logger.info(f"Running {len(tasks)} fragment loops for up to 60 seconds...")
            # Let tasks run for a specific duration for this test
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=60.0) 
            logger.info("Runner finished normally (all tasks completed - unexpected for loops).")
        except asyncio.TimeoutError:
            logger.info("Runner timeout reached (60s). Stopping fragment tasks.")
        except asyncio.CancelledError:
            logger.info("Runner main gather task cancelled.")
        except Exception as e:
            logger.error(f"Error during asyncio.gather: {e}", exc_info=True)
        finally:
            # Cleanly cancel tasks on exit/timeout/error
            logger.info("Attempting to cancel running fragment tasks...")
            cancelled_count = 0
            for task in tasks:
                if not task.done():
                    task.cancel()
                    cancelled_count += 1
                    # logger.debug(f"Cancelled task: {task.get_name()}")
            if cancelled_count > 0:
                logger.info(f"Cancelled {cancelled_count} tasks.")
            # Wait for cancellation to propagate
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("All fragment tasks stopped.")
    else:
        logger.warning("No fragment tasks were created.")

    logger.info("--- Emergence Test Runner Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Emergence runner interrupted by user.") 