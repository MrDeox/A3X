# Test script for SandboxExplorer skill

import logging
import asyncio
import sys  # Import sys for explicit handler output
from a3x.core.context import SharedTaskContext, Context
from a3x.skills.sandbox_explorer import SandboxExplorerSkill

# Configure logging more explicitly
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Set root level to DEBUG to capture everything

# Remove existing handlers if any (to avoid duplicates)
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add a handler to output to console (stderr)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)  # Set console handler level to INFO
root_logger.addHandler(console_handler)

# Also configure the specific logger for this script
logger = logging.getLogger(__name__) # Logger for this test script
logger.setLevel(logging.DEBUG) # Allow this script's DEBUG messages

async def test_sandbox_explorer():
    logger.info("--- Starting test for SandboxExplorer skill ---")
    
    # Initialize context and skill
    logger.debug("Initializing SharedTaskContext...")
    shared_context = SharedTaskContext(task_id="test_sandbox", initial_objective="Test autonomous sandbox exploration")
    logger.debug("Initializing Context...")
    context = Context()  # Placeholder for actual context if needed
    logger.debug("Initializing SandboxExplorerSkill...")
    explorer = SandboxExplorerSkill()
    
    # Set some initial data in SharedTaskContext
    logger.debug("Setting initial test data in SharedTaskContext...")
    shared_context.set(key="test_data", value={"initial": "data"}, source="test_script", tags=["test"])
    logger.info("SharedTaskContext initialized with test data.")
    
    # Run the explore_sandbox skill
    logger.info("Invoking explorer.explore_sandbox...")
    result = await explorer.explore_sandbox(
        context=context,
        objective="Teste de exploração autônoma no sandbox",
        max_attempts=3,
        shared_task_context=shared_context
    )
    
    logger.info(f"--- Test completed. Result: {result} ---")
    
    # Check SharedTaskContext for results
    logger.info("Retrieving all entries from SharedTaskContext...")
    context_data = shared_context.get_all_entries()
    logger.info(f"SharedTaskContext data after test: {context_data}")
    
    return result

if __name__ == "__main__":
    logger.info("Running test script...")
    asyncio.run(test_sandbox_explorer())
    logger.info("Test script finished.") 