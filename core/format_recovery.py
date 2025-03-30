import re
import json

def parse_react_output(text: str) -> dict:
    try:
        thought = re.search(r"Thought:\s*(.+)", text).group(1).strip()
        action_match = re.search(r"Action:\s*(\w+)", text)
        action = action_match.group(1).strip() if action_match else None
        
        action_input_raw_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
        action_input = {}
        if action_input_raw_match:
            action_input_raw = action_input_raw_match.group(1)
            cleaned_json = action_input_raw.replace("'", '"').strip()
            try:
                action_input = json.loads(cleaned_json)
            except json.JSONDecodeError as json_e:
                return {"error": f"Failed to decode Action Input JSON: {json_e}. Raw: {cleaned_json}"}

        if not action:
             return {"error": "Could not find valid Action name."}
             
        return {"thought": thought, "action": action, "action_input": action_input}
    except AttributeError as ae:
        return {"error": f"Could not find required ReAct pattern element: {ae}"}
    except Exception as e:
        return {"error": f"Failed to parse ReAct output: {e}"}
