import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile
from pathlib import Path

import sys
import types

import a3x.core.orchestrator as orchestrator_module
from a3x.core.orchestrator import TaskOrchestrator
from a3x.core.memory.memory_manager import MemoryManager
from a3x.core.tool_registry import ToolRegistry
from a3x.fragments.registry import FragmentRegistry, fragment
from a3x.fragments.base import BaseFragment

# Mocks simples para fragmentos meta
from a3x.fragments.base import BaseFragment, FragmentDef

class MockSelfEvolverFragment(BaseFragment):
    async def get_purpose(self, context=None):
        return "Mock purpose"
    async def execute(self, **kwargs):
        print("[DEBUG] MockSelfEvolverFragment.execute called (REAL METHOD)")
        return {"status": "success", "message": "Mock Self Evolver Executed"}

class MockMetaReflectorFragment(BaseFragment):
    async def get_purpose(self, context=None):
        return "Mock purpose"
    async def execute(self, **kwargs):
        print("[DEBUG] MockMetaReflectorFragment.execute called (REAL METHOD)")
        return {"status": "success", "message": "Mock Meta Reflector Executed"}

@pytest.fixture
def integration_test_env(monkeypatch):
    # Patch autodiscovery on the CLASS before any instance is created
    monkeypatch.setattr(
        "a3x.fragments.registry.FragmentRegistry.discover_and_register_fragments",
        lambda self, *args, **kwargs: None
    )
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    db_path = temp_path / "test_memory.db"
    index_path = temp_path / "test_index"
    test_config = {
        "SEMANTIC_INDEX_PATH": str(index_path),
        "DATABASE_PATH": str(db_path),
        "EPISODIC_MEMORY_LIMIT": 10,
    }
    tool_registry = ToolRegistry()
    fragment_registry = FragmentRegistry()
    class DummyLLMInterface:
        pass
    class DummyExceptionPolicy:
        def handle(self, exc, context=None):
            pass
    llm_interface = DummyLLMInterface()
    memory_manager = MemoryManager(test_config)
    exception_policy = DummyExceptionPolicy()
    workspace_root = str(temp_path)
    logger = MagicMock()
    yield {
        "tool_registry": tool_registry,
        "fragment_registry": fragment_registry,
        "llm_interface": llm_interface,
        "memory_manager": memory_manager,
        "exception_policy": exception_policy,
        "workspace_root": workspace_root,
        "logger": logger,
        "temp_dir": temp_dir,
    }
    temp_dir.cleanup()

class DummyLLMInterface:
    pass

class DummyExceptionPolicy:
    def handle(self, exc, context=None):
        pass

class DummySharedTaskContext:
    def __init__(self, task_id="test_task_id"):
        self.task_id = task_id
    def get_data(self, key, default=None):
        return default

@pytest.mark.asyncio
async def test_task_completion_triggers_self_evolver_fragment(monkeypatch, integration_test_env):
    tool_registry = integration_test_env["tool_registry"]
    fragment_registry = integration_test_env["fragment_registry"]
    llm_interface = integration_test_env["llm_interface"]
    memory_manager = integration_test_env["memory_manager"]
    exception_policy = integration_test_env["exception_policy"]
    workspace_root = integration_test_env["workspace_root"]
    logger = integration_test_env["logger"]

    # Registro manual do fragmento mock
    mock_se_def = FragmentDef(
        name="SelfEvolverFragment",
        fragment_class=MockSelfEvolverFragment,
        description="Mock para SelfEvolver",
        category="Evolution",
        skills=["mock_skill_for_test"],
        managed_skills=["mock_managed_skill"],
        prompt_template="mock template for test",
        capabilities=[]
    )
    fragment_registry._fragment_defs["SelfEvolverFragment"] = mock_se_def
    fragment_registry._fragment_classes["SelfEvolverFragment"] = MockSelfEvolverFragment

    orchestrator = TaskOrchestrator(
        fragment_registry=fragment_registry,
        tool_registry=tool_registry,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        agent_logger=logger,
        workspace_root=Path(workspace_root),
        config={},

    )
    
    class DummySharedTaskContext:
        def __init__(self, task_id="test_task_id"):
            self.task_id = task_id
        def get_data(self, key, default=None):
            return default
    shared_context = DummySharedTaskContext()
    main_history = []
    objective = "Test objective"
    final_status = "success"

    # Mock MemoryManager.learn_from_task para retornar trigger self_evolution
    async def fake_learn_from_task(learning_data):
        return {"trigger": "self_evolution", "context": {"foo": "bar"}}
    monkeypatch.setattr(memory_manager, "learn_from_task", fake_learn_from_task)

    # Mock execute do MockSelfEvolverFragment
    print("[DEBUG] Entering patch for MockSelfEvolverFragment.execute")
    with patch.object(MockSelfEvolverFragment, "execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = {"status": "success"}
        await orchestrator._invoke_learning_cycle(
            objective=objective,
            main_history=main_history,
            final_status=final_status,
            shared_context=shared_context,
        )
        # Asserts
        assert mock_execute.call_count == 1
        mock_execute.assert_awaited_with(foo="bar")

@pytest.mark.asyncio
async def test_task_completion_triggers_meta_reflector_fragment(monkeypatch, integration_test_env):
    tool_registry = integration_test_env["tool_registry"]
    fragment_registry = integration_test_env["fragment_registry"]
    llm_interface = integration_test_env["llm_interface"]
    memory_manager = integration_test_env["memory_manager"]
    exception_policy = integration_test_env["exception_policy"]
    workspace_root = integration_test_env["workspace_root"]
    logger = integration_test_env["logger"]

    # Registro manual do fragmento mock
    mock_mr_def = FragmentDef(
        name="MetaReflectorFragment",
        fragment_class=MockMetaReflectorFragment,
        description="Mock para MetaReflector",
        category="Reflection",
        skills=["mock_skill_for_test"],
        managed_skills=["mock_managed_skill"],
        prompt_template="mock template for test",
        capabilities=[]
    )
    fragment_registry._fragment_defs["MetaReflectorFragment"] = mock_mr_def
    fragment_registry._fragment_classes["MetaReflectorFragment"] = MockMetaReflectorFragment

    orchestrator = TaskOrchestrator(
        fragment_registry=fragment_registry,
        tool_registry=tool_registry,
        llm_interface=llm_interface,
        memory_manager=memory_manager,
        agent_logger=logger,
        workspace_root=Path(workspace_root),
        config={},

    )
    class DummySharedTaskContext:
        def __init__(self, task_id="test_task_id"):
            self.task_id = task_id
        def get_data(self, key, default=None):
            return default
    shared_context = DummySharedTaskContext()
    main_history = []
    objective = "Test objective"
    final_status = "failure"

    # Mock MemoryManager.learn_from_task para retornar trigger reflection_on_failure
    async def fake_learn_from_task(learning_data):
        return {"trigger": "reflection_on_failure", "context": {"fail_reason": "unit test"}}
    monkeypatch.setattr(memory_manager, "learn_from_task", fake_learn_from_task)

    # Mock execute do MockMetaReflectorFragment
    print("[DEBUG] Entering patch for MockMetaReflectorFragment.execute")
    with patch.object(MockMetaReflectorFragment, "execute", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = {"status": "success"}
        await orchestrator._invoke_learning_cycle(
            objective=objective,
            main_history=main_history,
            final_status=final_status,
            shared_context=shared_context,
        )
        # Asserts
        assert mock_execute.call_count == 1
        mock_execute.assert_awaited_with(fail_reason="unit test")
