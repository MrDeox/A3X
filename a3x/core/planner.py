import logging
import json
import re  # Add re import, needed for json_find_gpt
from typing import List, Optional

# Assuming llm_interface and prompt_builder are accessible -> # Package imports
# from .llm_interface import call_llm
# from .prompt_builder import build_planning_prompt
from a3x.core.llm_interface import LLMInterface
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
    llm_interface: LLMInterface,
    heuristics_context: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Generates a plan (list of steps) to achieve the objective using the LLM.

    Args:
        objective: The user's high-level objective.
        tool_descriptions: A string describing the available tools.
        agent_logger: The logger instance for logging messages.
        llm_interface: The LLMInterface instance to use for the API call.
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
        heuristics_context=heuristics_context,
    )

    # 2. Call the LLM
    try:
        llm_response_raw = ""
        async for chunk in llm_interface.call_llm(
            messages=planning_prompt_messages, stream=False
        ):
            llm_response_raw += chunk

        if not llm_response_raw:
            agent_logger.warning(
                "[Planner] LLM call (non-streaming) yielded an empty response string."
            )

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
                plan = json.loads(llm_response_raw)
                plan_str_for_logging = llm_response_raw
            except json.JSONDecodeError:
                agent_logger.error(
                    "[Planner] Fallback failed: Could not decode entire response as JSON either."
                )
                return None
        else:
            plan_str = json_match
            plan_str_for_logging = plan_str
            if plan_str is None:
                agent_logger.error(
                    "[Planner] JSON block found by regex, but the capturing group was empty. Cannot parse."
                )
                return None
            plan = json.loads(plan_str)

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

        if not plan:
            agent_logger.warning(
                "[Planner] LLM returned an empty plan list []. Objective might be unachievable or trivial."
            )
            return []

        agent_logger.info(
            f"[Planner] Plan generated successfully with {len(plan)} steps."
        )
        return plan

    except json.JSONDecodeError as json_err:
        agent_logger.error(
            f"[Planner] Failed to decode JSON response: {plan_str_for_logging} | Error: {json_err}"
        )
        return None
    except Exception as parse_err:
        agent_logger.exception(f"[Planner] Error parsing plan: {parse_err}")
        return None


def json_find_gpt(input_str: str) -> Optional[str]:
    """
    Finds the first ```json ... ``` block or raw JSON object/list in a string.
    Handles optional language specifier (e.g., ```json). Non-greedy match.
    Raw JSON detection is basic and assumes the string starts/ends with {} or [].
    """
    # Pattern 1: Code block ```json ... ``` or ``` ... ``` (non-greedy)
    # Group 1 captures the content inside the block.
    # Pattern 2: Raw JSON object {.*} or list [.*] (non-greedy)
    # Group 2 captures the raw object or list.
    # We prioritize the code block match.
    pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```|(\{.*?\}|\[.*?\])", re.DOTALL)
    match = pattern.search(input_str)

    if match:
        # If Group 1 matched (code block), return its content.
        if match.group(1) is not None:
            return match.group(1)
        # If Group 2 matched (raw JSON), return its content.
        elif match.group(2) is not None:
            # Basic check: Does the *whole* match look like the start/end of the input?
            # This avoids matching JSON snippets embedded in other text.
            matched_raw = match.group(2)
            if input_str.strip() == matched_raw:
                return matched_raw
            # else: Fall through, likely an embedded snippet not intended

    # Fallback/Alternative: Check if the entire stripped string is a JSON object/list
    # This covers cases where the regex might fail for complex raw JSON but the
    # overall structure is simple.
    stripped_input = input_str.strip()
    if stripped_input.startswith("{") and stripped_input.endswith("}"):
        return stripped_input
    if stripped_input.startswith("[") and stripped_input.endswith("]"):
        # Ensure this wasn't already caught and returned by the regex match for raw JSON
        # to avoid returning twice or overriding a potentially better regex match.
        # However, if regex didn't match, this is a valid fallback.
        if not match or match.group(2) is None:
            return stripped_input

    return None  # No match found by regex or basic checks
