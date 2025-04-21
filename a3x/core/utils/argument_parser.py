# a3x/core/utils/argument_parser.py
# LLM desativado. Esta utilidade não deve mais acessar modelos para inferir argumentos. Toda cognição é feita pelo A³Net.

import logging
import json
import re
import inspect # Needed to inspect skill signatures
from typing import Dict, Any, Optional

# Assuming these imports might be necessary based on potential implementation
# You might need to adjust these based on your actual context structure
try:
    from a3x.core.tool_registry import ToolRegistry
except ImportError:
    ToolRegistry = None # Placeholder if not found
try:
    from a3x.core.llm_interface import LLMInterface
except ImportError:
    LLMInterface = None # Placeholder if not found
try:
    # Use a general context type if specific ones cause issues
    from a3x.core.context import Context 
except ImportError:
    Context = None # Placeholder if not found

class ArgumentParser:
    """
    Utility class to parse arguments for skills/tools from natural language text.
    """

    @staticmethod
    async def _get_tool_signature_str(tool_name: str, tool_registry: ToolRegistry, logger: logging.Logger) -> Optional[str]:
        """Helper to get a string representation of the tool's signature from its schema."""
        if not tool_registry:
            logger.warning(f"Tool registry not available to get signature for {tool_name}")
            return None
        try:
            # Get the pre-generated schema from the registry
            tool_info = tool_registry.get_tool_info(tool_name)
            if not tool_info or 'schema' not in tool_info:
                logger.warning(f"Schema not found for tool '{tool_name}' in registry.")
                return None

            schema = tool_info['schema']
            if not schema or 'parameters' not in schema or 'properties' not in schema['parameters']: 
                logger.warning(f"Schema for tool '{tool_name}' is missing parameter properties.")
                return None # Or return "()" if appropriate?

            properties = schema['parameters'].get('properties', {})
            required_params = schema['parameters'].get('required', [])
            
            param_details = []
            for name, prop_info in properties.items():
                # Skip internally managed context params if they somehow leak into schema
                # (Though generate_schema should handle this)
                if name == 'ctx': 
                    continue 
                    
                param_type = prop_info.get('type', 'any')
                description = prop_info.get('description', 'No description')
                # Determine if required - check schema's required list
                is_required = name in required_params
                # Default value might be in schema or could be inferred (complex)
                # For simplicity, just indicate if required for now.
                # default_info = f" (default: {prop_info['default']})" if 'default' in prop_info else ""
                required_info = " (required)" if is_required else " (optional)"
                
                param_details.append(f"- {name} ({param_type}){required_info}: {description}")
                
            if not param_details:
                return "(No user-configurable parameters)"
            else:
                return "\n".join(param_details)

        except Exception as e:
            logger.error(f"Error getting signature/schema info for tool '{tool_name}': {e}", exc_info=True) # Log traceback
            return None

    @staticmethod
    async def parse_arguments_from_natural_language(
        natural_language_text: str,
        tool_name: str,
        tool_registry: Optional[ToolRegistry],
        logger: logging.Logger,
        context: Optional[Context] = None, # Pass context for potential LLM use
    ) -> Dict[str, Any]:
        """
        Parses arguments for a given tool from a natural language description.

        Args:
            natural_language_text: The text describing the arguments (e.g., the part
                                   of the plan step after the skill name).
            tool_name: The name of the tool/skill for which to parse arguments.
            tool_registry: The ToolRegistry instance to get tool schema/metadata.
            logger: Logger instance.
            context: Optional broader context, potentially containing LLMInterface.

        Returns:
            A dictionary containing the parsed arguments. Returns an empty dict
            if parsing fails or is not implemented.
        """
        logger.debug(f"Attempting to parse args for tool '{tool_name}' from text: '{natural_language_text}'")
        parsed_args: Dict[str, Any] = {}

        # --- Primary Parsing Logic: LLM Call ---
        # This section should be uncommented and implemented when ready
        
        llm_available = context and hasattr(context, 'llm_interface') and context.llm_interface and LLMInterface is not None
        registry_available = tool_registry and ToolRegistry is not None

        if llm_available and registry_available:
            try:
                signature_str = await ArgumentParser._get_tool_signature_str(tool_name, tool_registry, logger)
                
                if signature_str:
                    # Updated prompt using the detailed parameter info from schema
                    prompt = f"""Analyze the following natural language instruction intended for the tool '{tool_name}':
Instruction: "{natural_language_text.strip()}"

The tool '{tool_name}' expects arguments defined by the following schema:
Schema:
{signature_str}

Your task is to extract the values for these arguments *solely* from the provided Instruction text.
Output *only* a single valid JSON object where:
- Keys are the exact parameter names from the Schema.
- Values are the corresponding arguments extracted from the Instruction.
- Include only the arguments explicitly mentioned or clearly implied in the Instruction.
- If an argument is mentioned but its value cannot be determined, you *may* omit it or represent it reasonably (e.g., boolean true/false).
- Adhere strictly to the parameter names and expected types suggested by the schema (string, boolean, integer, etc.) when determining the value.
- If no relevant arguments are found in the Instruction, return an empty JSON object {{}}.

JSON Output:"""
                    
                    # messages = [{"role": "user", "content": prompt}]
                    # response_str = ""
                    # # Assuming call_llm handles non-streaming correctly if stream=False
                    # async for chunk in context.llm_interface.call_llm(messages=messages, stream=False, temperature=0.0): # Low temp for structured output
                    #      response_str += chunk
                    # 
                    # logger.debug(f"ArgumentParser LLM raw response for {tool_name}: {{response_str}}") # Add debug log
                    # 
                    # # Robust JSON extraction: Find first valid JSON object
                    # json_match = re.search(r'{{\s*.*?\s*}}', response_str, re.DOTALL | re.IGNORECASE)
                    # 
                    # # REMOVED: markdown cleaning regex - replaced by search
                    # # response_str = re.sub(r\"```(?:json)?\\s*(.*?)\\s*```\", r\"\\1\", response_str.strip(), flags=re.DOTALL)
                    # 
                    # # Basic validation: check if it looks like JSON
                    # # if response_str.startswith(\"{\") and response_str.endswith(\"}\"): # Check if JSON was found
                    # if json_match:
                    #     json_str = json_match.group(0)
                    #     try:
                    #         parsed_args = json.loads(json_str)
                    #         logger.info(f"LLM extracted args for {tool_name}: {{parsed_args}}")
                    #     except json.JSONDecodeError as json_err:
                    #          logger.error(f"LLM response for argument extraction looked like JSON but failed to parse: {{json_err}}. JSON string: {{json_str}}")
                    #          #parsed_args remains empty
                    # else:
                    #     logger.warning(f"Could not find valid JSON object {{{{...}}}} in LLM response for argument extraction: {{response_str}}")
                    #     #parsed_args remains empty
                    logger.warning(f"LLM-based argument parsing is deprecated. Relying on fallbacks for tool '{tool_name}'.")
                    parsed_args = {} # Ensure it's empty if LLM is disabled

                else:
                    logger.warning(f"Could not get signature for tool '{tool_name}'. Cannot use LLM for parsing.")
            except Exception as e:
                logger.error(f"LLM-based argument parsing failed for {tool_name}: {e}", exc_info=True)
        else:
            if not llm_available: logger.warning("LLMInterface not available in context for argument parsing.")
            if not registry_available: logger.warning("ToolRegistry not available for argument parsing.")

        # --- End LLM Parsing Section ---


        # --- Fallback/Specific Regex Parsing ---
        # Only run fallbacks if LLM parsing failed or was skipped
        if not parsed_args: 
            logger.debug(f"LLM parsing skipped or failed for '{tool_name}'. Attempting fallbacks.")
            if tool_name == 'propose_skill_from_gap':
                logger.debug("Attempting regex fallback for propose_skill_from_gap")
                # Refined Regex: Capture name, then the rest as description
                prop_match = re.search(
                    r"(?:named|the skill|skill)\s+'([^']+)'\s*(.*)", 
                    natural_language_text, 
                    re.IGNORECASE
                )
                if prop_match:
                    skill_name_suggestion = prop_match.group(1)
                    skill_description = prop_match.group(2).strip()
                    if skill_description and re.search(r"\\w", skill_description): 
                        parsed_args = {
                            "skill_name_suggestion": skill_name_suggestion,
                            "skill_description": skill_description
                        }
                        logger.info(f"Regex fallback extracted args for {tool_name}: {parsed_args}")
                    else:
                        logger.warning(f"Regex fallback for {tool_name} extracted name '{skill_name_suggestion}' but description was empty/invalid: '{skill_description}'")
                else:
                    logger.warning(f"Regex fallback failed to match expected pattern for {tool_name} in: '{natural_language_text}'")
            
            elif tool_name == 'reload_generated_skills':
                 logger.debug(f"No arguments needed for {tool_name}, using empty dict.")
                 parsed_args = {}
            
            elif tool_name == 'get_public_ip': # Example for the generated skill
                 # This skill might have args defined in its generated code.
                 # The LLM parser *should* handle this if enabled.
                 # If only using regex, we have no way to know the args here.
                 logger.warning(f"No specific regex fallback for {tool_name}. Relying on LLM parser (if enabled) or returning empty args.")
                 parsed_args = {}
            
            elif tool_name == 'final_answer':
                 logger.debug(f"Attempting regex fallback for final_answer")
                 # Simple fallback: assume the whole text is the answer message
                 if natural_language_text and natural_language_text.strip():
                     parsed_args = {"answer": natural_language_text.strip()}
                     logger.info(f"Regex fallback extracted args for {tool_name}: {parsed_args}")
                 else:
                     logger.warning(f"Could not extract message for {tool_name} using regex fallback.")


        # --- Final Check ---
        if not parsed_args:
             logger.warning(f"Argument parsing did not yield results for tool '{tool_name}' using available methods. Returning empty args.")
        
        return parsed_args 