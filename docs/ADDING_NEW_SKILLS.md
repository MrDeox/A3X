# How to Add a New Skill to A³X

This guide outlines the standard procedure for adding new capabilities (skills) to the A³X agent, based on the current architecture (`a3x.core.tools` decorator).

## 1. File Location and Naming

- Create a new Python file (e.g., `my_new_skill.py`) within an appropriate subdirectory under `a3x/skills/`.
  - Example: For a skill interacting with a calendar API, place it in `a3x/skills/communication/calendar_skill.py`.
  - Example: For a skill analyzing financial data, use `a3x/skills/analysis/finance_analyzer.py`.
- Ensure the subdirectory has an `__init__.py` file (it can be empty or used for specific sub-package setup).

## 2. Skill Implementation Structure

- **Standalone Function**: Implement the skill as a standalone `async def` function (preferred for I/O operations) or a regular `def` function. **Avoid wrapping the skill in a class** unless absolutely necessary, as the current execution logic primarily supports standalone functions found via the decorator.
- **Import Decorator**: Import the `@skill` decorator from `a3x.core.tools`.

```python
# Example: a3x/skills/example/my_new_skill.py
import logging
from pathlib import Path
from typing import Any, Dict

# Import the core skill decorator
from a3x.core.tools import skill

# Get a logger for the module
logger = logging.getLogger(__name__)

@skill(
    # ... decorator arguments ...
)
async def my_new_skill_function(ctx: Any, required_param: str, optional_param: int = 10) -> Dict[str, Any]:
    # ... skill logic ...
```

## 3. The `@skill` Decorator

Decorate your skill function with `@skill`, providing the following arguments:

- `name` (str): A unique, descriptive name for the skill. This is used by the LLM to identify and invoke the skill. Use snake_case (e.g., `read_calendar_events`).
- `description` (str): A clear, concise explanation of what the skill does, its purpose, and when it should be used. This is crucial for the LLM's decision-making.
- `parameters` (Dict[str, tuple]): A dictionary defining the skill's input parameters.
  - **Keys**: Parameter names (strings) matching the function signature arguments.
  - **Values**: A tuple `(type, default_value)`.
    - `type`: The expected Python type (e.g., `str`, `int`, `bool`, `Path`, `Dict`, `List`). Import necessary types (`pathlib`, `typing`).
    - `default_value`:
      - Use `...` (Ellipsis) for **required** parameters.
      - Provide a literal default value (e.g., `10`, `None`, `False`, `"default_string"`) for **optional** parameters.

```python
@skill(
    name="example_skill_with_params",
    description="An example skill demonstrating parameter definition.",
    parameters={
        "target_file": (Path, ...),         # Required Path parameter
        "search_query": (str, ...),         # Required string parameter
        "max_results": (int, 10),          # Optional integer, defaults to 10
        "case_sensitive": (bool, False),   # Optional bool, defaults to False
        "user_settings": (Dict, None),     # Optional dict, defaults to None
    }
)
async def example_skill_with_params(
    ctx: Any,
    target_file: Path,
    search_query: str,
    max_results: int = 10,
    case_sensitive: bool = False,
    user_settings: Dict = None
) -> Dict[str, Any]:
    # ... implementation ...
```

## 4. Function Signature

- The function signature **must match** the parameters defined in the `@skill` decorator (excluding `ctx` and potentially `agent_memory`).
- **`ctx` Argument (Optional but Recommended)**:
  - You can include `ctx: Any` as the first argument.
  - The context object provides `ctx.log` (a logger instance specific to the agent run). Use this for logging within your skill.
  - Other attributes like `ctx.llm_call` might be available when run by the full agent but are **not guaranteed** in all execution contexts (like `--run-skill`). Use `hasattr(ctx, 'attribute_name')` to check if needed.
- **`agent_memory` Argument (Optional)**:
  - If your skill needs access to the agent's short-term memory, include `agent_memory: Dict[str, Any]` in the signature. The `tool_executor` will automatically pass the memory dictionary if this argument is present.
- **Parameter Type Hinting**: Use standard Python type hints (`typing` module) for clarity and potential static analysis.

## 5. Return Value

- Your skill function **MUST** return a dictionary.
- **Standard Format**: It's highly recommended to return a dictionary following the structure used by `create_skill_response` (in `a3x/core/skills_utils.py`), although you don't *have* to import and use that function directly.
  - **Success**:
    ```python
    return {
        "status": "success",
        "action": "action_performed", # A code indicating what happened
        "data": {"result_key": "result_value", ...}, # The actual results
        "message": "User-friendly success message."
    }
    ```
  - **Error**:
    ```python
    return {
        "status": "error",
        "action": "error_type_code", # A code indicating the error
        "data": {}, # Optional: Include relevant data even on error
        "message": "User-friendly error message.",
        "error_details": "Optional technical details about the error."
    }
    ```
- The `tool_executor` expects this dictionary format. Returning other types (like simple strings or `None`) may cause errors in the agent loop.

## 6. Registration (Automatic Discovery)

- **Sub-package `__init__.py`**: Ensure the `__init__.py` file in the skill's parent directory (e.g., `a3x/skills/communication/__init__.py`) imports your skill's module. This triggers the execution of your skill file and runs the `@skill` decorator.

  ```python
  # Example: a3x/skills/communication/__init__.py

  # Import modules to ensure their @skill decorators run
  from . import calendar_skill
  from . import email_skill
  from . import my_new_skill # <--- Add this line

  import logging
  logger = logging.getLogger(__name__)
  logger.debug("Communication skills package initialized.")
  ```

- **Main `skills/__init__.py`**: The top-level `a3x/skills/__init__.py` uses `pkgutil.walk_packages` to automatically find and import all sub-packages (like `communication`), which in turn import the individual skill modules. You generally don't need to modify the top-level `__init__.py`.

## 7. Logging and Error Handling

- Use `ctx.log` or `logging.getLogger(__name__)` for logging within your skill.
- Implement robust `try...except` blocks to catch potential errors (API issues, file not found, invalid input, etc.).
- Return informative error dictionaries (see Step 5) when exceptions occur.

## Example Checklist

1.  [ ] Create `a3x/skills/category/my_skill.py`.
2.  [ ] Implement `async def my_skill(...)` or `def my_skill(...)`.
3.  [ ] Import `from a3x.core.tools import skill`.
4.  [ ] Add `@skill(...)` decorator with `name`, `description`, `parameters`.
5.  [ ] Ensure function signature matches decorator `parameters` (+ optional `ctx`, `agent_memory`).
6.  [ ] Implement skill logic using `ctx.log` for logging.
7.  [ ] Ensure the function returns a dictionary (`{"status": ...}`).
8.  [ ] Add `from . import my_skill` to `a3x/skills/category/__init__.py`.
9.  [ ] Test using `python -m a3x.cli.interface --run-skill your_skill_name --skill-args '{"param": "value"}'`. 