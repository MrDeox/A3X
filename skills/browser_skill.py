import logging
import asyncio
from playwright.async_api import async_playwright, Playwright, Browser, Page
from core.tools import skill
from core.skills_utils import create_skill_response

logger = logging.getLogger(__name__)

# --- Global State for Browser Instance ---
# We need to manage a single browser instance across skill calls
# Note: This simple global state might have issues with concurrency later.
# A more robust solution might involve a dedicated browser manager class.
_playwright_instance: Playwright | None = None
_browser_instance: Browser | None = None
_page_instance: Page | None = None

async def _get_page() -> Page:
    """Ensures Playwright is initialized and returns the current page instance."""
    global _playwright_instance, _browser_instance, _page_instance
    if _page_instance:
        return _page_instance

    logger.info("Initializing Playwright and launching browser...")
    try:
        _playwright_instance = await async_playwright().start()
        # Using chromium, could be parameterized later
        _browser_instance = await _playwright_instance.chromium.launch(headless=True) # Run headless for server environments
        _page_instance = await _browser_instance.new_page()
        logger.info("Playwright browser launched successfully.")
        return _page_instance
    except Exception as e:
        logger.error(f"Failed to initialize Playwright or launch browser: {e}", exc_info=True)
        raise RuntimeError(f"Playwright initialization failed: {e}") from e

async def _close_browser_resources():
    """Closes the browser and Playwright resources."""
    global _playwright_instance, _browser_instance, _page_instance
    logger.info("Closing Playwright browser resources...")
    if _page_instance:
        try:
            await _page_instance.close()
        except Exception as e:
            logger.warning(f"Error closing page: {e}")
        _page_instance = None
    if _browser_instance:
        try:
            await _browser_instance.close()
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        _browser_instance = None
    if _playwright_instance:
        try:
            await _playwright_instance.stop()
        except Exception as e:
            logger.warning(f"Error stopping Playwright: {e}")
        _playwright_instance = None
    logger.info("Playwright resources closed.")

# --- Browser Control Skills ---

@skill(
    name="browser_open_url",
    description="Opens a specified URL in the browser.",
    parameters={"url": (str, ...)} # URL is required
)
async def open_url(url: str, agent_history: list | None = None) -> dict:
    """Opens a URL in the Playwright browser instance."""
    logger.info(f"Executing skill: browser_open_url with URL: {url}")
    try:
        page = await _get_page()
        await page.goto(url, wait_until='domcontentloaded') # Wait for DOM to be ready
        logger.info(f"Successfully navigated to URL: {url}")
        # Return snippet of content or just success confirmation?
        # For now, just success. LLM can request content separately.
        return create_skill_response(
            status="success",
            message=f"Successfully opened URL: {url}",
            data={"current_url": page.url}
        )
    except Exception as e:
        logger.error(f"Error opening URL {url}: {e}", exc_info=True)
        return create_skill_response(
            status="error",
            message=f"Failed to open URL {url}.",
            error_details=str(e)
        )

@skill(
    name="browser_get_page_content",
    description="Retrieves the full HTML content of the current page.",
    parameters={}
)
async def get_page_content(agent_history: list | None = None) -> dict:
    """Gets the full HTML content of the current page."""
    logger.info("Executing skill: browser_get_page_content")
    try:
        page = await _get_page()
        content = await page.content()
        logger.info(f"Successfully retrieved page content (length: {len(content)}).")
        # Warning: Content can be very large. Consider truncation or specific element targeting.
        return create_skill_response(
            status="success",
            message="Successfully retrieved page content.",
            data={"html_content": content}
        )
    except Exception as e:
        logger.error(f"Error getting page content: {e}", exc_info=True)
        return create_skill_response(
            status="error",
            message="Failed to retrieve page content.",
            error_details=str(e)
        )

@skill(
    name="browser_click",
    description="Clicks on an element specified by a CSS selector.",
    parameters={"selector": (str, ...)} # Selector is required
)
async def click(selector: str, agent_history: list | None = None) -> dict:
    """Clicks on an element identified by a CSS selector."""
    logger.info(f"Executing skill: browser_click with selector: {selector}")
    try:
        page = await _get_page()
        # Add reasonable timeout
        await page.click(selector, timeout=5000) # 5 second timeout
        logger.info(f"Successfully clicked element with selector: {selector}")
        # Maybe wait for navigation or check URL after click? Depends on LLM plan.
        return create_skill_response(
            status="success",
            message=f"Successfully clicked element: {selector}",
            data={"current_url": page.url}
        )
    except Exception as e:
        logger.error(f"Error clicking selector {selector}: {e}", exc_info=True)
        # Provide more specific error if possible (e.g., timeout, element not found)
        error_message = f"Failed to click selector '{selector}'."
        if "Timeout" in str(e):
            error_message += " Timeout exceeded."
        elif "strict mode violation" in str(e):
             error_message += " Selector likely matched multiple elements or element not visible/enabled."
        else:
             error_message += " Element might not exist or be interactable."

        return create_skill_response(
            status="error",
            message=error_message,
            error_details=str(e)
        )

@skill(
    name="browser_fill_form",
    description="Fills a form field specified by a CSS selector with the given text.",
    parameters={
        "selector": (str, ...), # Selector is required
        "text": (str, ...)      # Text is required
    }
)
async def fill_form(selector: str, text: str, agent_history: list | None = None) -> dict:
    """Fills a form field identified by a CSS selector."""
    logger.info(f"Executing skill: browser_fill_form with selector: {selector}")
    try:
        page = await _get_page()
        # Add reasonable timeout
        await page.fill(selector, text, timeout=5000) # 5 second timeout
        logger.info(f"Successfully filled form field {selector}")
        return create_skill_response(
            status="success",
            message=f"Successfully filled field '{selector}'.",
            data={}
        )
    except Exception as e:
        logger.error(f"Error filling form field {selector}: {e}", exc_info=True)
        # Provide more specific error if possible
        error_message = f"Failed to fill field '{selector}'."
        if "Timeout" in str(e):
            error_message += " Timeout exceeded."
        elif "strict mode violation" in str(e):
             error_message += " Selector likely matched multiple elements or element not visible/enabled."
        else:
             error_message += " Element might not exist or be interactable."
        return create_skill_response(
            status="error",
            message=error_message,
            error_details=str(e)
        )

@skill(
    name="browser_get_text",
    description="Retrieves the text content of an element specified by a CSS selector.",
     parameters={"selector": (str, ...)} # Selector is required
)
async def get_text(selector: str, agent_history: list | None = None) -> dict:
    """Gets the text content of an element identified by a CSS selector."""
    logger.info(f"Executing skill: browser_get_text with selector: {selector}")
    try:
        page = await _get_page()
        # Add reasonable timeout
        text_content = await page.text_content(selector, timeout=5000) # 5 second timeout
        if text_content is None:
             text_content = "" # Return empty string if element has no text or doesn't exist
        logger.info(f"Successfully retrieved text from selector: {selector}")
        return create_skill_response(
            status="success",
            message="Successfully retrieved text content.",
            data={"text_content": text_content}
        )
    except Exception as e:
        logger.error(f"Error getting text from selector {selector}: {e}", exc_info=True)
        error_message = f"Failed to get text from selector '{selector}'."
        if "Timeout" in str(e):
            error_message += " Timeout exceeded."
        elif "strict mode violation" in str(e):
             error_message += " Selector likely matched multiple elements."
        else:
             error_message += " Element might not exist."
        return create_skill_response(
            status="error",
            message=error_message,
            error_details=str(e)
        )

# TODO: Add a skill or mechanism to properly close the browser when the agent session ends.
# A simple approach could be registering an exit handler, but might need integration
# with the agent's lifecycle management.
# For now, resources might leak if the agent process is killed abruptly.

# Example of how to potentially register cleanup (might need adjustment based on main app structure)
# import atexit
# atexit.register(lambda: asyncio.run(_close_browser_resources()))
