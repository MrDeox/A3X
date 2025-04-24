import logging
from typing import Dict, Any, Optional, List

# Assuming FragmentContext is importable from a core module
# from a3x.core.context import FragmentContext
# For now, using Any as a placeholder if the exact import path is unknown
from typing import Any as FragmentContext

# Import the fragment that performs the mutation
from a3x.fragments.generative_mutator_fragment import GenerativeMutatorFragment

logger = logging.getLogger(__name__)

async def mutar_fragmento(ctx: FragmentContext, fragment_name: str, num_variations: Optional[int] = None, strategy: Optional[str] = None) -> Dict[str, Any]:
    """
    Skill to mutate a given fragment using the GenerativeMutatorFragment.

    This skill acts as the callable action for the A3L directive:
    'mutar fragmento <fragment_name> [gerando <num_variations>] [usando <strategy>]'

    Args:
        ctx: The execution context (FragmentContext or similar).
        fragment_name: The name of the fragment to mutate (captured from A3L).
        num_variations: (Optional) Number of variations to generate.
        strategy: (Optional) Mutation strategy to use.

    Returns:
        A dictionary containing the status, IDs of the generated mutations,
        and potentially suggested A3L commands for evaluation.
        Example:
        {
            "status": "success",
            "mutation_ids": ["X_mut_1", "X_mut_2"],
            "suggested_a3l_commands": [
                "avaliar fragmento 'X_mut_1'",
                "avaliar fragmento 'X_mut_2'"
            ]
        }
        or
        {"status": "error", "message": "Error message..."}
    """
    logger.info(f"Executing mutar_fragmento skill for fragment: {fragment_name}, variations: {num_variations}, strategy: {strategy}")

    try:
        # 1. Instantiate the GenerativeMutatorFragment
        # We need to pass the context (ctx) to the fragment constructor if it requires it.
        # Assuming the fragment constructor takes the context.
        mutator = GenerativeMutatorFragment(context=ctx) # Pass context if needed

        # 2. Prepare parameters for the fragment's execute method
        # The GenerativeMutatorFragment's execute method might expect a dictionary
        # or specific arguments. Let's assume it takes the fragment name and optional params.
        # We need to align this with the actual signature of GenerativeMutatorFragment.execute
        params = {
            "fragment_to_mutate": fragment_name,
            "num_variations": num_variations,
            "strategy": strategy
        }
        # Filter out None values if the fragment's execute method doesn't handle them
        execute_params = {k: v for k, v in params.items() if v is not None}


        # 3. Call execute() on the fragment instance
        # The execute method might be async or sync. Assuming async based on context.
        result = await mutator.execute(**execute_params) # Adjust based on actual execute signature

        # 4. Format and return the result
        # The fragment should ideally return a dictionary compatible with our desired output.
        # If not, we might need to adapt the result here.
        if isinstance(result, dict) and "mutation_ids" in result:
             logger.info(f"Mutation successful for {fragment_name}. Mutations: {result.get('mutation_ids')}")
             # Ensure status is included
             if 'status' not in result:
                 result['status'] = 'success'
             return result
        else:
             # Handle unexpected result format from the fragment
             logger.warning(f"GenerativeMutatorFragment returned an unexpected result format: {result}")
             # Return a success status but indicate potential issue or wrap the result
             return {"status": "success", "message": "Mutation process initiated, but result format unclear.", "raw_result": result}


    except Exception as e:
        logger.exception(f"Error during mutar_fragmento skill execution for '{fragment_name}':")
        return {"status": "error", "message": f"Failed to mutate fragment '{fragment_name}': {e}"}

# Example of how this skill might be registered or used (conceptual)
# In a skill registration mechanism:
# register_skill("mutar_fragmento", mutar_fragmento) 