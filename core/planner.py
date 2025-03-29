# core/planner.py
import json
import logging
from typing import List, Dict, Any, Optional

# Import the central LLM call function and constants
# We might need to refactor call_llm later to accept a custom schema,
# but for now, generate_plan will use requests directly for simplicity.
from .llm_interface import LLAMA_DEFAULT_HEADERS, LLM_TIMEOUT # Reuse existing constants if needed
import requests # Needed for exceptions

# Logger for this module
logger = logging.getLogger(__name__)

# Define the expected JSON schema for the planning response
PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "A single, actionable step in the plan."
            },
            "description": "A list of sequential steps to achieve the objective."
        }
    },
    "required": ["plan"]
}

def build_planning_prompt(objective: str, tool_descriptions: str, agent_logger: logging.Logger) -> List[Dict[str, Any]]:
    """Builds the prompt messages for the planning phase LLM call."""
    # Use an f-string for cleaner formatting
    system_message = f"""You are an expert planner AI. Your goal is to break down a complex objective into a sequence of actionable steps using a predefined set of tools.

Available Tools:
{tool_descriptions}

Objective: {objective}

Based on the objective and the available tools, create a step-by-step plan. Each step should be a clear, concise instruction that directly uses one of the available tools or provides a final answer.
Return the plan ONLY as a JSON object conforming to the following schema:
{json.dumps(PLANNING_SCHEMA, indent=2)}

Example:
Objective: Read the file 'config.txt' and then list the contents of the 'data/' directory.
Output:
{{
  "plan": [
    "Use the 'read_file' tool to read the contents of 'config.txt'.",
    "Use the 'list_files' tool to list the contents of the 'data/' directory."
  ]
}}
"""
    # User message can be simpler, the system prompt carries the main instruction
    user_message = f"Generate a plan for the objective: {objective}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    agent_logger.debug(f"[Planner] Planning prompt built.")
    # agent_logger.debug(f"[Planner] Planning prompt messages: {messages}") # Avoid logging large prompts unless needed
    return messages

def generate_plan(objective: str, tool_descriptions: str, agent_logger: logging.Logger, llm_url: str) -> Optional[List[str]]:
    """Generates a plan by calling the LLM with a planning prompt."""
    agent_logger.info("[Planner] Generating plan...")
    prompt_messages = build_planning_prompt(objective, tool_descriptions, agent_logger)

    # Construct payload for the planning LLM call, forcing JSON output with PLANNING_SCHEMA
    payload = {
        "messages": prompt_messages,
        "temperature": 0.2, # Lower temperature for more deterministic planning
        "max_tokens": 1000, # Adjust if plans might be very long
        "stream": False,
        "response_format": {
            "type": "json_object",
            "schema": PLANNING_SCHEMA
        }
    }

    agent_logger.info(f"[Planner] Sending planning request to: {llm_url}")
    # agent_logger.debug(f"[Planner] Planning payload: {json.dumps(payload, indent=2)}") # Avoid logging large payloads

    try:
        # Using requests directly for now to enforce specific schema easily.
        response = requests.post(llm_url, headers=LLAMA_DEFAULT_HEADERS, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        agent_logger.debug(f"[Planner] Raw planning response received.") # Avoid logging full response data by default
        # agent_logger.debug(f"[Planner] Raw planning response received: {response_data}")

        if 'choices' in response_data and response_data['choices']:
            # Check if message content exists
            message = response_data['choices'][0].get('message', {})
            message_content = message.get('content', '').strip()

            if message_content:
                try:
                    # Parse the JSON content
                    parsed_json = json.loads(message_content)
                    plan = parsed_json.get('plan')

                    # Validate the plan structure
                    if isinstance(plan, list) and all(isinstance(step, str) for step in plan):
                        agent_logger.info(f"[Planner] Plan generated successfully with {len(plan)} steps.")
                        return plan
                    else:
                        # Log specific validation error
                        if not isinstance(plan, list):
                            error_detail = f"'plan' key is not a list (Type: {type(plan)})."
                        else:
                            error_detail = "'plan' list contains non-string items."
                        agent_logger.error(f"[Planner ERROR] LLM response JSON is valid, but structure is incorrect: {error_detail} Content: {message_content[:500]}...")
                        return None # Indicate failure

                except json.JSONDecodeError as e:
                    agent_logger.error(f"[Planner ERROR] Failed to parse LLM response content as JSON: {e}. Content: {message_content[:500]}...")
                    return None # Indicate failure
            else:
                agent_logger.error(f"[Planner ERROR] LLM planning response OK, but 'content' is empty or missing. Message object: {message}")
                return None # Indicate failure
        else:
            # Log if 'choices' or message structure is unexpected
            agent_logger.error(f"[Planner ERROR] LLM planning response OK, but unexpected format (missing 'choices' or 'message'). Response: {response_data}")
            return None # Indicate failure

    # Handle specific exceptions
    except requests.exceptions.Timeout as e:
        agent_logger.error(f"[Planner ERROR] Request timed out contacting LLM for planning at {llm_url}: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        # Log HTTP errors specifically, including response if possible
        err_response_text = e.response.text[:500] + '...' if e.response else 'N/A'
        agent_logger.error(f"[Planner ERROR] HTTP error during LLM planning call to {llm_url}: {e}. Response: {err_response_text}")
        return None
    except requests.exceptions.RequestException as e:
        # Catch other connection/request errors
        agent_logger.error(f"[Planner ERROR] Failed to connect/communicate with LLM for planning at {llm_url}: {e}")
        return None
    except json.JSONDecodeError as e:
         # If response.json() itself fails (e.g., server returns non-JSON error page)
         agent_logger.error(f"[Planner ERROR] Failed to decode LLM server's main planning response as JSON: {e}. Response text: {response.text[:500]}...")
         return None
    except Exception as e:
        # Catch any other unexpected errors
        agent_logger.exception("[Planner ERROR] Unexpected error during plan generation:") # Use exception to include traceback
        return None 