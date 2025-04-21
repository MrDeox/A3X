import pytest
import asyncio
import logging
from typing import Any # Import Any for MockContextStore typing
from unittest.mock import AsyncMock, MagicMock, call

# Assume a3x root is in PYTHONPATH or adjust as needed
from a3x.fragments.coordinator_fragment import CoordinatorFragment, CoordinatorFragmentDef
from a3x.core.context import FragmentContext

# Configure logging for the test
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock ContextStore
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
    """Provides a mocked FragmentContext with a MockContextStore."""
    context = MagicMock(spec=FragmentContext)
    context.store = MockContextStore()
    context.logger = logger # Use the test logger
    context.task_id = "test_task_123"
    # Mock other necessary attributes/methods if CoordinatorFragment uses them
    return context

@pytest.fixture
def coordinator_fragment(mock_context):
    """Provides an instance of CoordinatorFragment with mocked post_chat_message."""
    # Use the example FragmentDef provided in the fragment file
    fragment_def = CoordinatorFragmentDef
    fragment = CoordinatorFragment(fragment_def)
    fragment.post_chat_message = AsyncMock()
    # Set the context for the fragment instance
    fragment.set_context(mock_context)
    return fragment

@pytest.mark.asyncio
async def test_directive_escalation_logic(coordinator_fragment, mock_context):
    """Tests that the coordinator sends reassess, retry, then abort directives."""
    subtask_id = "subtask_abc"
    failing_sender = "StructureAutoRefactor" # Example sender

    # --- Failure 1: Expect 'reassess' --- 
    failure_msg_1 = {
        "type": "status",
        "sender": failing_sender,
        "content": {
            "status": "failed",
            "subtask_id": subtask_id,
            "responsible_fragment": failing_sender, # Explicitly set
            "details": "First failure details..."
        }
    }
    await coordinator_fragment.handle_realtime_chat(failure_msg_1, mock_context)

    # Check ContextStore update
    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 1

    # Check directive sent
    expected_directive_1 = {
        "type": "directive",
        "action": "reassess",
        "target": failing_sender,
        "reason": f"1 failure(s) recorded for subtask {subtask_id}",
        "subtask_id": subtask_id
    }
    coordinator_fragment.post_chat_message.assert_called_once_with(
        context=mock_context,
        message_type="directive",
        content=expected_directive_1
    )

    # Reset mock for next call
    coordinator_fragment.post_chat_message.reset_mock()

    # --- Failure 2: Expect 'retry' --- 
    failure_msg_2 = {
        "type": "status",
        "sender": failing_sender,
        "content": {
            "status": "failed",
            "subtask_id": subtask_id,
            # responsible_fragment omitted, should default to sender
            "details": "Second failure details..."
        }
    }
    await coordinator_fragment.handle_realtime_chat(failure_msg_2, mock_context)

    # Check ContextStore update
    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 2

    # Check directive sent
    expected_directive_2 = {
        "type": "directive",
        "action": "retry",
        "target": failing_sender, # Should default to sender
        "reason": f"2 failure(s) recorded for subtask {subtask_id}",
        "subtask_id": subtask_id
    }
    coordinator_fragment.post_chat_message.assert_called_once_with(
        context=mock_context,
        message_type="directive",
        content=expected_directive_2
    )

    # Reset mock for next call
    coordinator_fragment.post_chat_message.reset_mock()

    # --- Failure 3: Expect 'abort' --- 
    failure_msg_3 = {
        "type": "status",
        "sender": failing_sender,
        "content": {
            "status": "failed",
            "subtask_id": subtask_id,
            "responsible_fragment": failing_sender,
            "details": "Third failure details..."
        }
    }
    await coordinator_fragment.handle_realtime_chat(failure_msg_3, mock_context)

    # Check ContextStore update
    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 3

    # Check directive sent
    expected_directive_3 = {
        "type": "directive",
        "action": "abort",
        "target": failing_sender,
        "reason": f"3 failure(s) recorded for subtask {subtask_id}",
        "subtask_id": subtask_id
    }
    coordinator_fragment.post_chat_message.assert_called_once_with(
        context=mock_context,
        message_type="directive",
        content=expected_directive_3
    )

    # Reset mock for next call
    coordinator_fragment.post_chat_message.reset_mock()

    # --- Failure 4: Expect 'abort' again --- 
    # (Ensure it doesn't reset or change after 3)
    failure_msg_4 = {
        "type": "status",
        "sender": failing_sender,
        "content": {
            "status": "failed",
            "subtask_id": subtask_id,
            "responsible_fragment": failing_sender,
            "details": "Fourth failure details..."
        }
    }
    await coordinator_fragment.handle_realtime_chat(failure_msg_4, mock_context)

    # Check ContextStore update
    assert await mock_context.store.get(f"task_failures:{subtask_id}") == 4

    # Check directive sent
    expected_directive_4 = {
        "type": "directive",
        "action": "abort",
        "target": failing_sender,
        "reason": f"4 failure(s) recorded for subtask {subtask_id}",
        "subtask_id": subtask_id
    }
    coordinator_fragment.post_chat_message.assert_called_once_with(
        context=mock_context,
        message_type="directive",
        content=expected_directive_4
    )

 