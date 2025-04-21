import asyncio
import logging
from a3x.core.context import SharedTaskContext
# Assuming the original dispatcher is accessible, perhaps needs refactoring into a core module?
# For now, let's copy its definition here or import if possible.
# Let's assume we can import it from the test runner for now.
from a3x.tests.emergent_test_runner import message_dispatcher # Might need adjustment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting isolated dispatcher test...")
    shared_context = SharedTaskContext(task_id="dispatcher_test_01")
    
    # The original dispatcher expects (queue, fragment_context, shared_context)
    # We'll pass shared_context for both context arguments as decided before.
    logger.info("Starting the message dispatcher task...")
    dispatcher_task = asyncio.create_task(
        message_dispatcher(shared_context.internal_chat_queue, shared_context, shared_context),
        name="IsolatedDispatcher"
    )

    # Give the dispatcher a moment to start
    await asyncio.sleep(0.1)

    # Inject a test message
    test_message = {"type": "test_message", "sender": "IsolatedTest", "content": "hello dispatcher"}
    logger.info(f"Putting test message onto queue: {test_message}")
    await shared_context.internal_chat_queue.put(test_message)

    # Wait to see if the message is processed
    await asyncio.sleep(1)

    # Inject the stop message
    stop_message = {"type": "STOP_DISPATCHER"}
    logger.info(f"Putting stop message onto queue: {stop_message}")
    await shared_context.internal_chat_queue.put(stop_message)

    # Wait for the dispatcher to finish
    logger.info("Waiting for dispatcher task to complete...")
    try:
        await asyncio.wait_for(dispatcher_task, timeout=5.0)
        logger.info("Dispatcher task completed normally.")
    except asyncio.TimeoutError:
        logger.error("Dispatcher task timed out.")
        dispatcher_task.cancel()
        await asyncio.gather(dispatcher_task, return_exceptions=True)
    except Exception as e:
        logger.error(f"Error waiting for dispatcher task: {e}", exc_info=True)

    # Check logs manually for output like:
    # "Got message from queue: test_message"
    # "Unknown message type received: test_message" (expected for this test)
    # "STOP_DISPATCHER received. Exiting loop."

    logger.info("Isolated dispatcher test finished.")

if __name__ == "__main__":
    # We might need to handle imports related to handle_realtime_chat if the dispatcher calls it directly
    # For this isolated test, we might need to mock or prevent that call if it causes issues.
    # Let's run it first and see.
    try:
        asyncio.run(main())
    except Exception as e:
         logger.error(f"Test script failed: {e}", exc_info=True) 