import pytest
import logging
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, List, Any

# --- A³X Core Imports ---
try:
    from a3x.fragments.anomaly_detector import AnomalyDetectorFragment
    from a3x.fragments.base import FragmentDef
    # Import find_message implicitly from conftest.py
    from a3x.tests.emergence.conftest import find_message 
except ImportError as e:
    pytest.skip(f"Skipping AnomalyDetector tests due to import errors: {e}", allow_module_level=True)

logger = logging.getLogger(__name__)

# --- Constants for Tests ---
REPEATED_FAILURE_THRESHOLD = 3 # Assuming this is the threshold
EXCESSIVE_ATTEMPTS_THRESHOLD = 10 # Assuming this is the threshold
TARGET_FILE_PATH = "a/b/c.py"
SENDER_NAME = "TestFragment"

# --- Fixture for the Fragment Under Test --- #

@pytest.fixture
def anomaly_detector_fragment(
    mock_fragment_context: MagicMock, # From conftest.py
    tool_registry: MagicMock          # From conftest.py
) -> AnomalyDetectorFragment:
    """Provides an instance of AnomalyDetectorFragment with mocked context and patched post_chat_message."""
    # AnomalyDetector doesn't require specific skills, it just listens
    metadata = {"name": "AnomalyDetector", "description": "Detects anomalies", "category": "System",
                "skills": []} 
    frag_def = FragmentDef(name=metadata["name"], description=metadata["description"],
                           category=metadata["category"], skills=metadata["skills"],
                           fragment_class=AnomalyDetectorFragment)

    fragment = AnomalyDetectorFragment(fragment_def=frag_def, tool_registry=tool_registry)
    fragment._logger.setLevel(logging.DEBUG)
    fragment.REPEATED_FAILURE_THRESHOLD = REPEATED_FAILURE_THRESHOLD # Set for testing
    fragment.EXCESSIVE_ATTEMPTS_THRESHOLD = EXCESSIVE_ATTEMPTS_THRESHOLD # Set for testing
    
    # --- Patch post_chat_message to bypass base class check --- #
    # Store the original mock for assertion later if needed
    original_add_chat_mock = mock_fragment_context.shared_task_context.add_chat_message
    
    # Create a new mock to replace the method on the instance
    mock_post_chat = AsyncMock(name="patched_post_chat_message")

    # Define a side effect to capture calls and potentially call the original mock if needed
    async def post_chat_side_effect(context, message_type, content, target_fragment=None):
        # You could add logic here if needed, e.g., call the original mock:
        # original_add_chat_mock(fragment_name=fragment.get_name(), message_type=message_type, message_content=content)
        # For now, the AsyncMock itself records the call, which is enough for assert_called_once
        pass

    mock_post_chat.side_effect = post_chat_side_effect
    fragment.post_chat_message = mock_post_chat # Replace the instance method
    # ------------------------------------------------------------
    
    return fragment

# --- Helper Function to Create Messages ---

def create_message(msg_type: str, status: str, sender: str, target: str = TARGET_FILE_PATH, **kwargs) -> Dict[str, Any]:
    """Helper to create consistent test messages."""
    content = {"status": status, **kwargs}
    # Add target/path based on type expectation
    if msg_type == "refactor_result":
        content["target"] = target
    elif msg_type == "mutation_attempt":
        content["target_file"] = target
        
    return {
        "type": msg_type,
        "sender": sender,
        "content": content
    }

# --- Test Cases --- #

@pytest.mark.asyncio
async def test_repeated_failure_detection(
    anomaly_detector_fragment: AnomalyDetectorFragment,
    mock_fragment_context: MagicMock
):
    """Test anomaly detection for repeated failures on the same file."""
    # Send two error messages - should not trigger anomaly yet
    msg1 = create_message("refactor_result", "error", SENDER_NAME, TARGET_FILE_PATH, details="Fail 1")
    msg2 = create_message("mutation_attempt", "error", SENDER_NAME, TARGET_FILE_PATH, details="Fail 2")
    
    await anomaly_detector_fragment.handle_realtime_chat(msg1, mock_fragment_context)
    # Assert on the patched method
    anomaly_detector_fragment.post_chat_message.assert_not_called()
    
    await anomaly_detector_fragment.handle_realtime_chat(msg2, mock_fragment_context)
    # Assert on the patched method
    anomaly_detector_fragment.post_chat_message.assert_not_called()
    
    # Send the third error message - should trigger anomaly
    msg3 = create_message("refactor_result", "error", SENDER_NAME, TARGET_FILE_PATH, details="Fail 3")
    await anomaly_detector_fragment.handle_realtime_chat(msg3, mock_fragment_context)
    
    # Assert that the patched post_chat_message was called exactly once
    anomaly_detector_fragment.post_chat_message.assert_called_once()
    
    # --- Check arguments passed to the patched method --- 
    call_args, call_kwargs = anomaly_detector_fragment.post_chat_message.call_args
    
    assert call_kwargs['context'] == mock_fragment_context # Check context was passed
    assert call_kwargs['message_type'] == "anomaly"
    content = call_kwargs['content'] # The content dict passed to post_chat_message
    assert content["type"] == "anomaly" # Check inner content structure
    assert content["issue"] == "repeated_failure"
    assert content["file_path"] == TARGET_FILE_PATH
    assert content["suspect_fragment"] == SENDER_NAME
    assert isinstance(content["details"], str)
    assert TARGET_FILE_PATH in content["details"]
    assert str(REPEATED_FAILURE_THRESHOLD) in content["details"]
    assert isinstance(content["extra_context"], dict)
    # --------------------------------------------------------

@pytest.mark.asyncio
async def test_excessive_attempts_detection(
    anomaly_detector_fragment: AnomalyDetectorFragment,
    mock_fragment_context: MagicMock
):
    """Test anomaly detection for excessive attempts from the same sender."""
    # Send messages up to the threshold - should not trigger anomaly yet
    for i in range(EXCESSIVE_ATTEMPTS_THRESHOLD - 1):
        msg_type = "refactor_result" if i % 2 == 0 else "mutation_attempt"
        status = "success" if i % 3 == 0 else "error" # Mix success and error
        msg = create_message(msg_type, status, SENDER_NAME, f"file_{i}.py")
        await anomaly_detector_fragment.handle_realtime_chat(msg, mock_fragment_context)
        anomaly_detector_fragment.post_chat_message.assert_not_called()
    
    # Send the message that hits the threshold
    last_msg_type = "refactor_result" if (EXCESSIVE_ATTEMPTS_THRESHOLD - 1) % 2 == 0 else "mutation_attempt"
    last_status = "success" if (EXCESSIVE_ATTEMPTS_THRESHOLD - 1) % 3 == 0 else "error"
    threshold_msg = create_message(last_msg_type, last_status, SENDER_NAME, f"file_{EXCESSIVE_ATTEMPTS_THRESHOLD-1}.py")
    await anomaly_detector_fragment.handle_realtime_chat(threshold_msg, mock_fragment_context)

    # Assert that the patched post_chat_message was called exactly once
    anomaly_detector_fragment.post_chat_message.assert_called_once()

    # Check the arguments passed to the patched method
    call_args, call_kwargs = anomaly_detector_fragment.post_chat_message.call_args
    assert call_kwargs['context'] == mock_fragment_context
    assert call_kwargs['message_type'] == "anomaly"
    content = call_kwargs['content']
    assert content["type"] == "anomaly"
    assert content["issue"] == "excessive_attempts"
    assert content["file_path"] is None
    assert content["suspect_fragment"] == SENDER_NAME
    assert isinstance(content["details"], str)
    assert SENDER_NAME in content["details"]
    assert str(EXCESSIVE_ATTEMPTS_THRESHOLD) in content["details"]
    assert content["extra_context"]["current_attempt_count"] == EXCESSIVE_ATTEMPTS_THRESHOLD

@pytest.mark.asyncio
async def test_ignores_irrelevant_messages(
    anomaly_detector_fragment: AnomalyDetectorFragment,
    mock_fragment_context: MagicMock
):
    """Test that the fragment ignores messages it doesn't track."""
    irrelevant_messages = [
        {"type": "directive", "sender": "Planner", "content": {"action": "do_something"}},
        {"type": "user_feedback", "sender": "User", "content": {"rating": 5}},
        {"type": "system_log", "sender": "System", "content": {"message": "Booting up"}},
        create_message("refactor_result", "error", "OtherSender", "other_file.py"),
        create_message("mutation_attempt", "success", "AnotherSender", "another_file.py"),
    ]
    
    for msg in irrelevant_messages:
        try:
            await anomaly_detector_fragment.handle_realtime_chat(msg, mock_fragment_context)
        except Exception as e:
            pytest.fail(f"Handling irrelevant message {msg['type']} raised an exception: {e}")
            
    # Assert that the patched post_chat_message was not called
    anomaly_detector_fragment.post_chat_message.assert_not_called()
    # Also check the underlying mock's captured list just in case
    # assert len(mock_fragment_context.shared_task_context._captured_messages) == 0

@pytest.mark.asyncio
async def test_success_resets_failure_counter(
    anomaly_detector_fragment: AnomalyDetectorFragment,
    mock_fragment_context: MagicMock
):
    """Test that a success message resets the failure counter for a file."""
    # Send two error messages (threshold is 3)
    msg1 = create_message("refactor_result", "error", SENDER_NAME, TARGET_FILE_PATH, details="Fail 1")
    msg2 = create_message("mutation_attempt", "error", SENDER_NAME, TARGET_FILE_PATH, details="Fail 2")
    
    await anomaly_detector_fragment.handle_realtime_chat(msg1, mock_fragment_context)
    await anomaly_detector_fragment.handle_realtime_chat(msg2, mock_fragment_context)
    # Assert on the patched method
    anomaly_detector_fragment.post_chat_message.assert_not_called()

    # Send a success message for the same file
    msg_success = create_message("refactor_result", "success", SENDER_NAME, TARGET_FILE_PATH, details="Success!")
    await anomaly_detector_fragment.handle_realtime_chat(msg_success, mock_fragment_context)
    # Assert on the patched method
    anomaly_detector_fragment.post_chat_message.assert_not_called()

    # Send another error message for the same file - should NOT trigger anomaly
    msg3_after_success = create_message("mutation_attempt", "error", SENDER_NAME, TARGET_FILE_PATH, details="Fail 3")
    await anomaly_detector_fragment.handle_realtime_chat(msg3_after_success, mock_fragment_context)

    # Assert that no anomaly message was posted because the counter was reset
    # Assert on the patched method
    anomaly_detector_fragment.post_chat_message.assert_not_called()
    # assert len(mock_fragment_context.shared_task_context._captured_messages) == 0

    # Just to be sure, send two more errors, which *should* trigger it now
    msg4 = create_message("refactor_result", "error", SENDER_NAME, TARGET_FILE_PATH, details="Fail 4")
    msg5 = create_message("mutation_attempt", "error", SENDER_NAME, TARGET_FILE_PATH, details="Fail 5")
    await anomaly_detector_fragment.handle_realtime_chat(msg4, mock_fragment_context)
    # Assert on the patched method
    anomaly_detector_fragment.post_chat_message.assert_not_called()
    await anomaly_detector_fragment.handle_realtime_chat(msg5, mock_fragment_context)

    # Now the patched method should be called
    anomaly_detector_fragment.post_chat_message.assert_called_once()

    # Check arguments passed to the patched method
    call_args, call_kwargs = anomaly_detector_fragment.post_chat_message.call_args
    assert call_kwargs['context'] == mock_fragment_context
    assert call_kwargs['message_type'] == "anomaly"
    content = call_kwargs['content']
    assert content["type"] == "anomaly"
    assert content["issue"] == "repeated_failure"
    assert content["file_path"] == TARGET_FILE_PATH
    assert content["suspect_fragment"] == SENDER_NAME
    assert isinstance(content["details"], str)
    assert TARGET_FILE_PATH in content["details"]
    assert str(REPEATED_FAILURE_THRESHOLD) in content["details"]
    assert isinstance(content["extra_context"], dict)