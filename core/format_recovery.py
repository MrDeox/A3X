import re
import json

def parse_react_output(text: str) -> dict:
    try:
        thought = re.search(r"Thought:\s*(.+)", text).group(1)
        action = re.search(r"Action:\s*(.+)", text).group(1)
        action_input_raw = re.search(r"Action Input:\s*(\{.+\})", text, re.DOTALL).group(1)
        action_input = json.loads(action_input_raw.replace("'", '"'))
        return {"thought": thought, "action": action, "action_input": action_input}
    except Exception as e:
        return {"error": f"Failed to parse ReAct output: {e}"}
