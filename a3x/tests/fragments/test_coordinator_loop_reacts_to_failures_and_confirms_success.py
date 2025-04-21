import pytest
import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

from a3x.fragments.coordinator_fragment import CoordinatorFragment, CoordinatorFragmentDef
from a3x.core.context import FragmentContext

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock ContextStore (same as before)
class MockContextStore:
    def __init__(self):
        self._store = {}

    async def get(self, key: str):
        logger.debug(f"MockContextStore.get('{key}') -> {self._store.get(key)}")
        return self._store.get(key)

    async def set(self, key: str, value: Any):
        logger.debug(f"MockContextStore.set('{key}', {value})")
        self._store[key] = value

@pytest.fixture
def mock_context():
    context = MagicMock(spec=FragmentContext)
    context.store = MockContextStore()
    context.logger = logger
    context.task_id = "test_task_loop_123"
    return context

@pytest.fixture
async def coordinator_fragment_with_loop(mock_context):
    """Provides CoordinatorFragment, starts its loop, and yields for testing."""
    fragment_def = CoordinatorFragmentDef
    fragment = CoordinatorFragment(fragment_def)
    fragment.post_chat_message = AsyncMock()

    # Set context which automatically starts the loop via fragment.start()
    fragment.set_context(mock_context)

    yield fragment # Provide the fragment to the test

    # Cleanup: Stop the loop after the test
    fragment.stop()
    # Give the loop a moment to finish processing and exit cleanly
    if fragment._monitoring_task:
        try:
            await asyncio.wait_for(fragment._monitoring_task, timeout=1.5)
        except asyncio.TimeoutError:
            logger.warning("Coordinator loop task did not finish cleanly after stop signal.")
        except asyncio.CancelledError:
            pass # Expected if cancelled directly

@pytest.mark.asyncio
async def test_coordinator_loop_reacts_to_failures_and_confirms_success(coordinator_fragment_with_loop, mock_context):
    """Tests the loop processes failures, sends directives, and logs confirmations."""
    fragment = coordinator_fragment_with_loop
    subtask_id = "subtask_xyz"
    target_fragment_name = "FailingFragment"

    # --- Send Failure 1 -> Expect 'reassess' --- 
    failure_msg_1 = {
        "type": "status",
        "sender": target_fragment_name,
        "content": {"status": "failed", "subtask_id": subtask_id, "details": "Failure 1"}
    }
    await fragment.handle_realtime_chat(failure_msg_1, mock_context) # Put on queue
    await asyncio.sleep(0.01) # Allow loop to process

    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 1
    expected_directive_1 = {
        "type": "directive", "action": "reassess", "target": target_fragment_name,
        "reason": f"1 failure(s) recorded for subtask {subtask_id}", "subtask_id": subtask_id
    }
    fragment.post_chat_message.assert_called_once_with(
        context=mock_context, message_type="directive",
        content=expected_directive_1, target_fragment=target_fragment_name
    )
    fragment.post_chat_message.reset_mock()

    # --- Send Reassess Success Confirmation -> Expect Log --- 
    confirm_msg_1 = {
        "type": "status",
        "sender": target_fragment_name,
        "content": {"status": "reassess_success", "subtask_id": subtask_id}
    }
    # We expect the coordinator to log this, but not necessarily change state or send new directives
    # Temporarily capture logs to verify (can be brittle)
    log_capture = []
    original_info = fragment._logger.info
    def mock_info(msg, *args, **kwargs):
        log_capture.append(msg % args)
        original_info(msg, *args, **kwargs)
    fragment._logger.info = mock_info

    await fragment.handle_realtime_chat(confirm_msg_1, mock_context)
    await asyncio.sleep(0.01) # Allow loop to process

    # Restore logger
    fragment._logger.info = original_info
    # Check if the confirmation was logged
    assert any(f"Received confirmation 'reassess_success' from '{target_fragment_name}' for subtask '{subtask_id}'" in log for log in log_capture)
    # Ensure no new directive was sent
    fragment.post_chat_message.assert_not_called()
    # Failure count should remain 1
    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 1

    # --- Send Failure 2 -> Expect 'retry' --- 
    failure_msg_2 = {
        "type": "status",
        "sender": target_fragment_name,
        "content": {"status": "failed", "subtask_id": subtask_id, "details": "Failure 2"}
    }
    await fragment.handle_realtime_chat(failure_msg_2, mock_context)
    await asyncio.sleep(0.01)

    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 2
    expected_directive_2 = {
        "type": "directive", "action": "retry", "target": target_fragment_name,
        "reason": f"2 failure(s) recorded for subtask {subtask_id}", "subtask_id": subtask_id
    }
    fragment.post_chat_message.assert_called_once_with(
        context=mock_context, message_type="directive",
        content=expected_directive_2, target_fragment=target_fragment_name
    )
    fragment.post_chat_message.reset_mock()

    # --- Send Retry Success Confirmation -> Expect Log --- 
    confirm_msg_2 = {
        "type": "status",
        "sender": target_fragment_name,
        "content": {"status": "retry_success", "subtask_id": subtask_id}
    }
    log_capture = []
    original_info = fragment._logger.info
    fragment._logger.info = mock_info # Reuse mock_info function

    await fragment.handle_realtime_chat(confirm_msg_2, mock_context)
    await asyncio.sleep(0.01)

    fragment._logger.info = original_info
    assert any(f"Received confirmation 'retry_success' from '{target_fragment_name}' for subtask '{subtask_id}'" in log for log in log_capture)
    fragment.post_chat_message.assert_not_called()
    # Failure count should remain 2 (unless reset logic is uncommented in fragment)
    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 2

    # --- Send Failure 3 -> Expect 'abort' --- 
    failure_msg_3 = {
        "type": "status",
        "sender": target_fragment_name,
        "content": {"status": "failed", "subtask_id": subtask_id, "details": "Failure 3"}
    }
    await fragment.handle_realtime_chat(failure_msg_3, mock_context)
    await asyncio.sleep(0.01)

    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 3
    expected_directive_3 = {
        "type": "directive", "action": "abort", "target": target_fragment_name,
        "reason": f"3 failure(s) recorded for subtask {subtask_id}", "subtask_id": subtask_id
    }
    fragment.post_chat_message.assert_called_once_with(
        context=mock_context, message_type="directive",
        content=expected_directive_3, target_fragment=target_fragment_name
    )
    fragment.post_chat_message.reset_mock()

    # --- Send Abort Acknowledged Confirmation -> Expect Log --- 
    confirm_msg_3 = {
        "type": "status",
        "sender": target_fragment_name,
        "content": {"status": "abort_acknowledged", "subtask_id": subtask_id}
    }
    log_capture = []
    original_info = fragment._logger.info
    fragment._logger.info = mock_info

    await fragment.handle_realtime_chat(confirm_msg_3, mock_context)
    await asyncio.sleep(0.01)

    fragment._logger.info = original_info
    assert any(f"Received confirmation 'abort_acknowledged' from '{target_fragment_name}' for subtask '{subtask_id}'" in log for log in log_capture)
    fragment.post_chat_message.assert_not_called()
    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 3 