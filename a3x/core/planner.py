import logging
import json
import asyncio  # Import asyncio
import re  # Add re import, needed for json_find_gpt
from typing import List, Optional

# Assuming llm_interface and prompt_builder are accessible -> # Package imports
# from .llm_interface import call_llm
# from .prompt_builder import build_planning_prompt
from a3x.core.llm_interface import call_llm
from a3x.core.prompt_builder import build_planning_prompt

# Assuming config is available for the test block
try:
    # from .config import LLAMA_SERVER_URL
    from a3x.core.config import LLAMA_SERVER_URL
except ImportError:
    LLAMA_SERVER_URL = None  # Define as None if config can't be imported

# <<< ADD PLANNING_SCHEMA Definition >>>
PLANNING_SCHEMA = {
    "type": "array",
    "items": {
        "type": "string",
        "description": "A single, clear, concise step in the plan.",
    },
    "description": "A list of strings representing the sequential steps of the plan.",
}

# Corrected PLANNER_SYSTEM_PROMPT declaration (standard triple quotes)
PLANNER_SYSTEM_PROMPT = """
You are an expert planning assistant AI. Your goal is to break down a complex user objective into a sequence of actionable steps that can be executed by another AI agent possessing a specific set of tools.

**Input:**
1.  **User Objective:** The high-level goal the user wants to achieve.
2.  **Available Tools:** A list of tools the execution agent can use, with their descriptions.

**Output:**
You must output a JSON list of strings, where each string is a clear and concise step towards achieving the objective. The steps should be logical, sequential, and leverage the available tools where appropriate. The final step should almost always be using the 'final_answer' tool unless the objective implies otherwise.

**Example:**

User Objective: "Research the current price of Bitcoin and save it to a file named btc_price.txt"
Available Tools:
- search_web(query: str) -> str: Searches the web for the query and returns results.
- write_file(filename: str, content: str) -> str: Writes the content to the specified file.
- final_answer(answer: str) -> str: Provides the final answer to the user.

You may also propose hypothetical tools that are not in the list above if you believe they would help achieve the objective. Clearly specify the tool name and its intended function. The system will treat such attempts as opportunities to learn and expand its capabilities.

Output:
```json
[
  "Use the search_web tool to find the current price of Bitcoin.",
  "Use the write_file tool to save the found price into 'btc_price.txt'.",
  "Use the final_answer tool to confirm the price has been saved."
]
```

**Constraints:**
- Output ONLY the JSON list of steps. Do not include any introductory text, explanations, or markdown formatting around the JSON.
- Ensure the output is a valid JSON list of strings.
- Each step should be a self-contained instruction.
- Consider the flow of information between steps (e.g., the output of a search step might be needed for a write step).
- If the objective is simple enough to be achieved in one step, create a plan with a single step (often just using 'final_answer').
- If the objective seems impossible or unclear with the given tools, output an empty JSON list `[]`.
"""


async def generate_plan(
    objective: str,
    tool_descriptions: str,
    agent_logger: logging.Logger,
    llm_url: Optional[str] = None,  # Allow passing LLM URL if needed
    heuristics_context: Optional[str] = None # <<< ADDED: Novo parâmetro para heurísticas
) -> Optional[List[str]]:
    """
    Generates a plan (list of steps) to achieve the objective using the LLM.

    Args:
        objective: The user's high-level objective.
        tool_descriptions: A string describing the available tools.
        agent_logger: The logger instance for logging messages.
        llm_url: Optional URL for the LLM service.
        heuristics_context: Optional string containing relevant learned heuristics.

    Returns:
        A list of strings representing the plan steps, or None if planning fails.
    """
    agent_logger.info(
        f"[Planner] Generating plan for objective: '{objective[:100]}...'"
    )

    # 1. Build the planning prompt
    planning_prompt_messages = build_planning_prompt(
        objective=objective,
        tool_descriptions=tool_descriptions,
        planner_system_prompt=PLANNER_SYSTEM_PROMPT,
        heuristics_context=heuristics_context # Passar o novo contexto
    )

    # 2. Call the LLM
    try:
        # <<< REVERTED: Use async for even for non-streaming, as call_llm always yields >>>
        llm_response_raw = ""  # Initialize raw response
        async for chunk in call_llm(
            planning_prompt_messages, llm_url=llm_url, stream=False
        ):
            llm_response_raw += (
                chunk  # Accumulate (should be one chunk for stream=False)
            )

        # Basic validation (optional, can be done during parsing)
        # if not isinstance(llm_response_raw, str):
        #     agent_logger.error(f"[Planner] LLM call yielded unexpected type: {type(llm_response_raw)}")
        #     return None
        if not llm_response_raw:
            agent_logger.warning(
                "[Planner] LLM call (non-streaming) yielded an empty response string."
            )
            # Allow processing empty strings

        agent_logger.debug(f"[Planner] Raw LLM response:\n{llm_response_raw}")
    except Exception as e:
        agent_logger.exception(f"[Planner] Error calling LLM for planning: {e}")
        return None

    # 3. Parse the LLM response (expecting JSON list)
    plan_str_for_logging = "[Extraction Failed]"
    try:
        json_match = json_find_gpt(llm_response_raw)
        if not json_match:
            agent_logger.error(
                f"[Planner] No JSON block found in LLM response: {llm_response_raw[:300]}..."
            )
            try:
                # Fallback: Check if the entire string might be JSON
                plan = json.loads(llm_response_raw)
                plan_str_for_logging = (
                    llm_response_raw  # Log the whole thing if it parsed
                )
            except json.JSONDecodeError:
                agent_logger.error(
                    "[Planner] Fallback failed: Could not decode entire response as JSON either."
                )
                return None  # Give up if no JSON found and direct parse fails
        else:
            plan_str = json_match.group(1)
            plan_str_for_logging = plan_str  # Log the extracted part
            if plan_str is None:
                agent_logger.error(
                    "[Planner] JSON block found by regex, but the capturing group was empty. Cannot parse."
                )
                return None
            plan = json.loads(plan_str)

        # <<< MODIFIED: Split type checking for more specific error logging >>>
        if not isinstance(plan, list):
            agent_logger.error(
                f"[Planner] LLM response is not a list: Parsed: {plan_str_for_logging}"
            )
            return None

        if not all(isinstance(step, str) for step in plan):
            agent_logger.error(
                f"[Planner] Plan list contains non-string elements: Parsed: {plan_str_for_logging}"
            )
            return None

        # Original combined check (now split above):
        # if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
        #     if not plan:
        #         agent_logger.warning(
        #             "[Planner] LLM returned an empty plan. Objective might be unachievable or trivial."
        #         )
        #         return []
        #     agent_logger.info(
        #         f"[Planner] Plan generated successfully with {len(plan)} steps."
        #     )
        #     return plan
        # else:
        #     agent_logger.error(
        #         f"[Planner] Parsed JSON is not a list of strings: {plan}"
        #     )
        #     return None

        # <<< If checks passed, the plan is valid >>>
        if not plan:  # Handle empty list case
            agent_logger.warning(
                "[Planner] LLM returned an empty plan list []. Objective might be unachievable or trivial."
            )
            return []  # Return empty list, not None

        agent_logger.info(
            f"[Planner] Plan generated successfully with {len(plan)} steps."
        )
        return plan

    except json.JSONDecodeError as json_err:
        # This catches errors from json.loads(llm_response_raw) OR json.loads(plan_str)
        agent_logger.error(
            f"[Planner] Failed to decode JSON from LLM response: {json_err}. Response/Extracted: '{plan_str_for_logging}'"
        )
        return None
    except Exception as e:
        agent_logger.exception(
            f"[Planner] Unexpected error parsing planning response: {e}"
        )
        return None


def json_find_gpt(input_str: str):
    """
    Finds the first json object demarcated by ```json ... ```
    Helper based on AutoGPT's parsing. Also handles if the whole string is JSON.
    """
    # Try finding ```json ``` block
    im_json = re.search(
        r"```(?:json)?\s*\n(.*?)\n```", input_str, re.DOTALL | re.IGNORECASE
    )

    if im_json:
        return im_json
    else:
        # Fallback: Check if the entire string is valid JSON
        try:
            json.loads(input_str)

            # If parsing succeeds, wrap it to mimic the regex group structure
            class MockMatch:
                _content = input_str

                def group(self, num):
                    if num == 1:
                        return self._content
                    return None

            return MockMatch()
        except json.JSONDecodeError:
            return None


# Example usage needs async context
async def main_test():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    test_logger = logging.getLogger("test_planner")
    test_objective = "Get the capital of France and write it to 'capital.txt'"
    test_tools = """
- get_capital(country: str) -> str
- write_file(filename: str, content: str) -> str
- final_answer(answer: str) -> str
"""

    if not LLAMA_SERVER_URL:
        print(
            "Error: LLAMA_SERVER_URL not configured (likely missing core/config.py or env var). Cannot run test."
        )
        return

    print(f"Testing planner with objective: '{test_objective}'")
    plan = await generate_plan(
        test_objective, test_tools, test_logger, llm_url=LLAMA_SERVER_URL
    )
    print("--- Generated Plan ---")
    if plan is not None:
        print(json.dumps(plan, indent=2))
    else:
        print("Failed to generate plan.")
    print("----------------------")


if __name__ == "__main__":
    try:
        # Check if running in an already running event loop
        loop = asyncio.get_running_loop()
        loop.create_task(main_test())
        # If in a running loop, let it manage execution.
        # This is common in environments like Jupyter.
        print("Test scheduled in existing event loop.")
    except RuntimeError:
        # No running event loop, run main_test directly
        try:
            asyncio.run(main_test())
        except ImportError:
            print(
                "Could not import config. Ensure core/config.py exists or environment variables are set."
            )
        except Exception as e:
            print(f"An error occurred during async test execution: {e}")
            import traceback

            traceback.print_exc()
