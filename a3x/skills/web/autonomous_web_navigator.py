"""Skill to autonomously navigate the web to achieve objectives by iteratively perceiving, planning, and acting using a cognitive loop."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import asyncio
import json
import sys
from collections import namedtuple

# Core A3X imports
from a3x.core.tools import skill
from a3x.core.config import project_root # Import project_root for path resolution
from a3x.skills.perception.describe_image_blip import describe_image_blip
from a3x.skills.perception.ocr_extract import extract_text_boxes_from_image

# Playwright import
try:
    from playwright.async_api import async_playwright, Error
    _playwright_imported_successfully = True
except ImportError:
    _playwright_imported_successfully = False
    async_playwright = None
    Error = Exception

logger = logging.getLogger(__name__)

# --- Cognitive Planning Prompt ---

PLANNING_PROMPT_COGNITIVE = """
**Objective:** {objective}

**Current Perception:**
*   Visual Description: "{visual_description}"
*   OCR Text Boxes (if any):
{ocr_text_boxes}

**Action History (Recent Steps):**
{action_history}

**Available Actions:**
- "click": Clicks on an element. Requires CSS `selector`.
- "fill": Fills a form field. Requires CSS `selector` and `value`.
- "scroll": Scrolls the page up or down. Requires `value` ("up" or "down").
- "pressEnter": Presses the Enter key on an element. Requires CSS `selector`.
- "finish": Ends the navigation task (use if objective is complete, impossible, or blocked).

**Your Task:** Analyze the perception and history, reason about the next step, and decide on the single best action to take towards the objective.

**Output Format (JSON Object ONLY):**
{{
  "reasoning": {{
    "percebo": "(Briefly describe the current state based on perception and history, acknowledging previous errors)",
    "considerando": "(Explain your thought process for choosing the next action, referencing the objective, perception, and potential error recovery)",
    "decido": "(State the chosen action clearly)",
    "porque": "(Justify the decision and the specific parameters chosen, especially the selector)"
  }},
  "action_plan": {{
    "action": "(One of: click, fill, scroll, pressEnter, finish)",
    "selector": "(CSS selector or text=... selector, REQUIRED unless action is finish or scroll)",
    "value": "(Value for fill/scroll, otherwise null or empty string)"
  }}
}}

**IMPORTANT INSTRUCTIONS:**
*   Analyze the `execution_result['status']` of the **last step** in `action_history`. If it was `error`, DO NOT repeat the same action/selector. Analyze the error message and current perception to choose a different action (e.g., scroll, finish, different selector).
*   **VERIFY BEFORE ACTING:** Before deciding on an action, explicitly compare the current perception (especially OCR text boxes) with the objective and the result of the *last executed action*.
*   **SUBMIT AFTER FILL:** If the *last successful action* was `\"fill\"` on an input/textarea element (confirm presence and content with OCR/visuals), your **immediate next action** MUST be to submit that form (usually `\"pressEnter\"` on the same element or `\"click\"` on a 'submit'/'search' button identified in OCR/visuals), UNLESS the perception clearly indicates an error or unexpected state. **DO NOT `fill` the same field again if the OCR shows the correct text is already present.**
*   If the visual description or OCR suggests a CAPTCHA, error page, or blockage, consider the `\"finish\"` action.
*   ⚠️ IMPORTANT: If a field was filled in a previous step (e.g. a search bar), you must now submit it by clicking the search button or pressing Enter. Do **not** use `\"finish\"` until you've completed the action and seen the result.
*   Base selectors ONLY on the CURRENT perception (visual description + OCR). Use `text=...` selectors for text found in OCR.
"""

# --- OLD Prompts Removed ---
# BROWSER_ACTION_PROMPT_NO_HISTORY_NO_OCR = ... (Removed)
# BROWSER_ACTION_PROMPT_NO_HISTORY_WITH_OCR = ... (Removed)
# BROWSER_ACTION_PROMPT_WITH_HISTORY = ... (Removed - Replaced by Cognitive Prompt)


# --- Internal Helper for Playwright Actions ---
async def _execute_playwright_action(page, action_details: Dict[str, Any], logger) -> Dict[str, Any]:
    """Executes a specified Playwright action based on a dictionary."""
    action_type = action_details.get("action")
    selector = action_details.get("selector")
    value = action_details.get("value")

    logger.info(f"Attempting Playwright action: {action_type} on selector '{selector}' with value '{value}'")

    if not action_type:
        return {"status": "error", "message": "Missing action type."}

    # Actions requiring a selector
    if action_type in ["click", "fill", "pressEnter"]:
        if not selector:
            logger.error(f"Selector is required for action type '{action_type}'.")
            return {"status": "error", "message": f"Missing selector for action '{action_type}'."}
        try:
            element = page.locator(selector).first
            await element.wait_for(state="visible", timeout=5000)

            if action_type == "click":
                await element.click(timeout=5000)
                logger.info(f"Successfully clicked element: {selector}")
                return {"status": "success", "message": f"Clicked '{selector}'"}
            elif action_type == "fill":
                if value is None:
                    return {"status": "error", "message": "Value is required for fill action."}
                await element.fill(value, timeout=5000)
                logger.info(f"Successfully filled element '{selector}' with value.")
                return {"status": "success", "message": f"Filled '{selector}'"}
            elif action_type == "pressEnter":
                 await page.wait_for_timeout(300) # Small delay before pressing Enter
                 await element.press("Enter", timeout=5000)
                 logger.info(f"Successfully pressed Enter on element: {selector}")
                 return {"status": "success", "message": f"Pressed Enter on '{selector}'"}

        except Error as e:
            logger.warning(f"Playwright error during action '{action_type}' on '{selector}': {e}")
            return {"status": "error", "message": f"Playwright error: {e}"}
        except Exception as e:
            logger.warning(f"Failed to execute action '{action_type}' on '{selector}': {e}")
            return {"status": "error", "message": f"Failed action '{action_type}': {e}"}

    # Scroll action (no selector needed)
    elif action_type == "scroll":
        scroll_amount = 800 # Default scroll down amount
        if value == "up":
            scroll_amount = -800
        elif value == "down":
            scroll_amount = 800
        elif value: # Allow specific pixel amounts? Maybe later.
             logger.warning(f"Unsupported scroll value '{value}'. Defaulting to down.")

        try:
            await page.mouse.wheel(0, scroll_amount)
            logger.info(f"Successfully scrolled {value or 'down'}.")
            return {"status": "success", "message": f"Scrolled {value or 'down'}"}
        except Exception as e:
            logger.warning(f"Failed to execute scroll action: {e}")
            return {"status": "error", "message": f"Failed scroll action: {e}"}

    # Finish action
    elif action_type == "finish":
        logger.info("'finish' action planned. Stopping interaction.")
        # This action type doesn't execute anything in Playwright, handled in main loop
        return {"status": "success", "message": "Finish action indicated."}

    else:
        return {"status": "error", "message": f"Unsupported action type: {action_type}"}


# --- Search Heuristic Helper ---
def extract_search_query(objective: str) -> Optional[str]:
    """Simple heuristic to extract query from objectives like 'search for X' or 'buscar por Y'."""
    objective_lower = objective.lower()
    # Added Portuguese prefix
    prefixes = ["search for ", "find ", "look up ", "buscar por "]
    for prefix in prefixes:
        if objective_lower.startswith(prefix):
            query = objective[len(prefix):].strip()
            if (query.startswith('"') and query.endswith('"')) or \
               (query.startswith('\'') and query.endswith('\'')):
                query = query[1:-1]
            return query if query else None
    return None

SEARCH_INPUT_SELECTORS = [
    "textarea[name='q']", # Google Search
    "input[name='q']",    # Google Search (alternative?)
    "input[type='search']",
    "input[type='text'][name*='search']",
    "input[type='text'][id*='search']",
    "input[type='text'][aria-label*='search']",
    "input[aria-label*='Search query']" # DuckDuckGo
]

async def try_heuristic_search_fill(page, objective: str, logger) -> Optional[Dict[str, Any]]:
    """Tries to fill a search bar based on the objective using common selectors."""
    search_query = extract_search_query(objective)
    if not search_query:
        return None # Not a recognized search objective

    logger.info(f"Heuristic: Detected search objective. Query: '{search_query}'")
    for selector in SEARCH_INPUT_SELECTORS:
        logger.debug(f"Heuristic: Trying selector '{selector}'")
        fill_action = {
            "action": "fill",
            "selector": selector,
            "value": search_query,
            "reason": "Heuristic attempt to fill search bar based on objective." # Add reason
        }
        result = await _execute_playwright_action(page, fill_action, logger)
        if result["status"] == "success":
            logger.info(f"Heuristic: Successfully filled '{selector}' with query.")
            # Return the successful action details for history
            return fill_action
        else:
            logger.debug(f"Heuristic: Selector '{selector}' failed: {result.get('message')}")

    logger.warning("Heuristic: Could not find or fill any common search input selectors.")
    return None # Indicate heuristic failed


@skill(
    name="autonomous_web_navigator",
    description="Autonomously navigates web pages using a cognitive loop (Perceive -> Plan -> Act -> Reflect) to achieve a user-defined objective.",
    parameters={
        "url": (str,),
        "objective": (str,),
        "max_steps": (int, 10)
    }
)
async def autonomous_web_navigator(ctx: Any, url: str, objective: str, max_steps: int = 10) -> Dict[str, Any]:
    """Main function for the autonomous web navigation skill with cognitive loop."""
    if not _playwright_imported_successfully:
        return {"status": "error", "message": "Playwright library not found. Please install it: pip install playwright && playwright install chromium"}

    if not ctx or not hasattr(ctx, 'llm_call'):
         return {"status": "error", "message": "LLM call context (ctx.llm_call) is not available."}

    action_history = [] # Stores detailed logs for each step
    current_status = "running"
    final_screenshot_path = None

    # --- Setup Screenshot Directory ---
    screenshots_dir = Path(project_root) / "memory" / "screenshots" / "autonav" # Subfolder for this skill
    try:
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured screenshot directory exists: {screenshots_dir}")
    except OSError as e:
        logger.error(f"Failed to create screenshot directory {screenshots_dir}: {e}")
        return {"status": "error", "message": f"Could not create screenshot directory: {e}"}

    def get_screenshot_path(step: int) -> Path:
        filename = f"step_{step}.png"
        return screenshots_dir / filename

    try:
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                logger.info(f"Browser launched. Navigating to initial URL: {url}")
                await page.goto(url, wait_until="load", timeout=30000)
                logger.info(f"Initial navigation to {url} successful.")
            except Error as e:
                logger.error(f"Playwright failed to launch/navigate: {e}")
                return {"status": "error", "message": f"Playwright initialization/navigation failed: {e}"}
            except Exception as e:
                 logger.error(f"Unexpected error during setup: {e}")
                 return {"status": "error", "message": f"Unexpected error during setup: {e}"}

            # --- Main Cognitive Loop ---
            for step in range(max_steps + 1):
                logger.info(f"--- Starting Cognitive Step {step}/{max_steps} ---")
                step_log: Dict[str, Any] = {"step": step} # Initialize log for this step

                # 1. PERCEIVE
                # ==================
                perception_result = {}
                visual_description = "Perception failed." # Default
                ocr_text_boxes = []
                try:
                    screenshot_path = get_screenshot_path(step)
                    await page.screenshot(path=str(screenshot_path), full_page=True)
                    logger.info(f"Step {step}: Screenshot saved to {screenshot_path}")
                    final_screenshot_path = str(screenshot_path)
                    perception_result["screenshot_path"] = str(screenshot_path)

                    # Call perception skills (BLIP/OCR) - non-critical
                    try:
                        # Use run_skill if available, otherwise direct call (assuming direct for now)
                        blip_result = await describe_image_blip(ctx=None, image_path=str(screenshot_path))
                        visual_description = blip_result.get('data', {}).get('description', 'Visual description failed.')
                        if blip_result.get('status') != 'success': logger.warning(f"Step {step}: BLIP non-success: {blip_result}")
                        else: logger.info(f"Step {step}: Visual Description: {visual_description[:100]}...")
                    except Exception as blip_err: logger.warning(f"Step {step}: BLIP exception: {blip_err}", exc_info=True)
                    perception_result["visual_description"] = visual_description

                    try:
                        ocr_result = await extract_text_boxes_from_image(ctx=None, image_path=str(screenshot_path))
                        ocr_text_boxes = ocr_result.get('data', {}).get('text_boxes', [])
                        if ocr_result.get('status') != 'success' or not ocr_text_boxes: logger.warning(f"Step {step}: OCR non-success or empty: {ocr_result}")
                        else: logger.info(f"Step {step}: OCR extracted {len(ocr_text_boxes)} boxes.")
                    except Exception as ocr_err: logger.warning(f"Step {step}: OCR exception: {ocr_err}", exc_info=True)
                    perception_result["ocr_text_boxes"] = ocr_text_boxes

                    step_log["perception_result"] = perception_result # Log perception

                except Exception as perc_err:
                    logger.error(f"Step {step}: Critical error during perception phase: {perc_err}", exc_info=True)
                    current_status = "error"
                    step_log["perception_result"] = perception_result # Log partial perception
                    step_log["error"] = f"Perception failed: {perc_err}"
                    action_history.append(step_log)
                    break # Stop loop

                # Handle Heuristic Fill at Step 0
                if step == 0:
                    heuristic_action = await try_heuristic_search_fill(page, objective, logger)
                    if heuristic_action:
                         # Log heuristic success and skip planning/execution for this step
                         step_log["planned_reasoning"] = {"reasoning": {"decido": "Apply heuristic fill"}, "action_plan": heuristic_action}
                         step_log["action_to_execute"] = heuristic_action # Action was already executed by helper
                         step_log["execution_result"] = {"status": "success", "message": "Heuristic fill successful."}
                         step_log["reflection"] = None # No reflection on heuristic
                         action_history.append(step_log)
                         logger.info(f"Step {step}: Heuristic fill applied. Proceeding.")
                         continue # Move to next step

                # 2. PLAN (Cognitive Reasoning)
                # ============================
                llm_reasoning_json: Optional[Dict] = None
                planned_action: Optional[Dict] = None
                llm_error: Optional[str] = None
                try:
                    # Prepare context for the LLM prompt
                    # Limit history to avoid overly long prompts (e.g., last 3 steps)
                    recent_history = action_history[-3:]
                    history_str = json.dumps(recent_history, indent=2, default=lambda o: "<non-serializable>") if recent_history else "[]"
                    ocr_str = json.dumps(ocr_text_boxes[:50], indent=2) if ocr_text_boxes else "[]" # Limit OCR data too

                    # Format the cognitive prompt
                    prompt = PLANNING_PROMPT_COGNITIVE.format(
                        objective=objective,
                        visual_description=visual_description,
                        action_history=history_str,
                        ocr_text_boxes=ocr_str
                    )
                    logger.debug(f"Step {step}: Sending cognitive prompt to LLM...")

                    # Call LLM
                    llm_response_str = ""
                    async for chunk in ctx.llm_call(prompt):
                        llm_response_str += chunk

                    # Parse LLM Response (JSON expected)
                    try:
                        # Basic JSON block extraction
                        json_start = llm_response_str.find('{')
                        json_end = llm_response_str.rfind('}') + 1
                        if json_start != -1 and json_end != -1:
                            json_str = llm_response_str[json_start:json_end]
                            llm_reasoning_json = json.loads(json_str)
                            # Extract the action plan part
                            planned_action = llm_reasoning_json.get("action_plan")
                            if not planned_action or "action" not in planned_action:
                                raise ValueError("LLM JSON missing 'action_plan' or 'action'.")
                            logger.info(f"Step {step}: Parsed LLM Plan: {planned_action}")
                            # Log the reasoning part
                            logger.debug(f"Step {step}: LLM Reasoning: {llm_reasoning_json.get('reasoning')}")
                        else:
                            raise json.JSONDecodeError("No JSON object found in LLM response", llm_response_str, 0)
                    except (json.JSONDecodeError, ValueError) as json_err:
                        logger.error(f"Step {step}: Failed to decode/validate JSON from LLM: {json_err}. Response: {llm_response_str}")
                        llm_error = f"LLM response error: {json_err}"
                        current_status = "error"

                except Exception as llm_call_err:
                    logger.error(f"Step {step}: Error during LLM call: {llm_call_err}")
                    llm_error = f"LLM call failed: {llm_call_err}"
                    current_status = "error"

                # Log planning results (even if failed)
                step_log["planned_reasoning"] = llm_reasoning_json # Store full reasoning JSON
                step_log["llm_error"] = llm_error

                if current_status == "error" and llm_error:
                    action_history.append(step_log) # Log failed plan
                    break # Stop loop

                if not planned_action: # Should not happen if error handling above is correct
                     logger.error(f"Step {step}: No valid action planned after LLM call. Stopping.")
                     current_status = "error"
                     step_log["error"] = "No valid action planned by LLM."
                     action_history.append(step_log)
                     break

                # 3. ADAPT (Auto-Evaluation based on History)
                # ===========================================
                action_to_execute = planned_action # Default to LLM plan
                consecutive_failures = 0
                if len(action_history) >= 2:
                    last_log = action_history[-1]
                    second_last_log = action_history[-2]
                    # Check if the last two steps had execution errors
                    if last_log.get("execution_result", {}).get("status") == "error" and \
                       second_last_log.get("execution_result", {}).get("status") == "error":
                        # Check if the *intended* actions and selectors were identical
                        last_planned = last_log.get("planned_reasoning", {}).get("action_plan", {})
                        second_last_planned = second_last_log.get("planned_reasoning", {}).get("action_plan", {})
                        if last_planned and second_last_planned and \
                           last_planned.get("action") == second_last_planned.get("action") and \
                           last_planned.get("selector") == second_last_planned.get("selector"):
                            # Check if the CURRENTLY planned action is the SAME
                            if planned_action.get("action") == last_planned.get("action") and \
                               planned_action.get("selector") == last_planned.get("selector"):
                                logger.warning(f"Step {step}: Detected 2 consecutive identical failures for action '{planned_action.get('action')}' on selector '{planned_action.get('selector')}'. Forcing 'finish'.")
                                action_to_execute = {"action": "finish", "selector": None, "value": None} # Override action
                                step_log["adaptation_applied"] = f"Forced 'finish' due to 2 consecutive identical failures."

                step_log["action_to_execute"] = action_to_execute # Log the action we will actually try

                # Check for finish action (either planned or forced)
                if action_to_execute.get("action") == "finish":
                    logger.info(f"Step {step}: 'finish' action determined. Stopping.")
                    current_status = "success" # Assume 'finish' means success unless forced by error
                    step_log["execution_result"] = {"status": current_status, "message": "Finish action determined."}
                    # No reflection needed for 'finish' usually
                    step_log["reflection"] = None
                    action_history.append(step_log)
                    break # Exit loop

                # Check max steps *before* final execution
                if step >= max_steps:
                     logger.warning(f"Step {step}: Maximum number of steps ({max_steps}) reached before final execution.")
                     current_status = "max_steps_reached"
                     # Log the final perception/plan attempt, but no execution/reflection
                     step_log["execution_result"] = None
                     step_log["reflection"] = None
                     action_history.append(step_log)
                     break # Stop loop

                # 4. ACT (Execute Action)
                # =======================
                execution_result = await _execute_playwright_action(page, action_to_execute, logger)
                step_log["execution_result"] = execution_result
                if execution_result["status"] == "error":
                    logger.warning(f"Step {step}: Action execution failed: {execution_result.get('message')}")
                    # Continue loop, LLM should see error in history next step

                # 5. REFLECT (using skill call)
                # ===========================
                reflection_result = None # Default if skill cannot be called
                if hasattr(ctx, 'run_skill'):
                    try:
                        logger.info(f"Step {step}: Calling reflect_on_execution skill...")
                        # Pass the log of the current step as execution_results
                        reflection_result = await ctx.run_skill(
                            "reflect_on_execution",
                            objective=objective,
                            # plan=???, # Plan not easily available here
                            execution_results=[step_log]
                        )
                        logger.info(f"Step {step}: Reflection skill completed.")
                        logger.debug(f"Step {step}: Reflection Result: {reflection_result}")
                        step_log["reflection"] = reflection_result # Store the actual reflection
                    except Exception as reflect_err:
                        logger.error(f"Step {step}: Error calling reflection skill: {reflect_err}", exc_info=True)
                        step_log["reflection"] = {"status": "error", "message": f"Reflection skill call failed: {reflect_err}"}
                        reflection_result = step_log["reflection"] # Ensure reflection_result has the error
                else:
                    logger.warning(f"Step {step}: ctx.run_skill not available. Skipping reflection step.")
                    step_log["reflection"] = None # Explicitly set to None

                # 6. LEARN (using skill call)
                # =========================
                if hasattr(ctx, 'run_skill') and isinstance(reflection_result, dict) and reflection_result.get('status') != 'error':
                    try:
                        logger.info(f"Step {step}: Calling learn_from_reflection_logs skill...")
                        # learn_from_reflection_logs might not need specific args from this step
                        await ctx.run_skill("learn_from_reflection_logs")
                        logger.info(f"Step {step}: Learning skill call completed.")
                    except Exception as learn_err:
                        logger.error(f"Step {step}: Error calling learning skill: {learn_err}", exc_info=True)
                        # Log learning error, maybe add to step_log if needed
                elif not hasattr(ctx, 'run_skill'):
                     logger.warning(f"Step {step}: ctx.run_skill not available. Skipping learning step.")
                else:
                    logger.warning(f"Step {step}: Skipping learning because reflection result was invalid or indicated an error: {reflection_result}")

                # Add completed step log to history
                action_history.append(step_log)

                # Optional delay
                await asyncio.sleep(1)

            # --- Loop End ---
            await browser.close()
            logger.info("Browser closed.")

    except Exception as e:
        logger.exception(f"An unexpected error occurred during autonomous navigation: {e}")
        current_status = "error"
        if not action_history or action_history[-1].get("error") is None:
             action_history.append({"step": len(action_history), "error": f"Unhandled exception: {e}"})

    # --- Prepare Final Result ---
    final_result = {
        "status": current_status if current_status != "running" else ("max_steps_reached" if step >= max_steps else "error"),
        "objective": objective,
        "start_url": url,
        "final_screenshot": final_screenshot_path,
        "history": action_history, # Include the detailed cognitive history
    }

    logger.info(f"Autonomous navigation finished with status: {final_result['status']}")
    # TODO: Persist final_result or action_history to a database/log store for long-term learning
    return final_result