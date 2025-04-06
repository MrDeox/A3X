import logging
from typing import Dict, Any, Optional

from a3x.core.tools import skill

# <<< ADDED Imports for BrowserManager >>>
from a3x.core.browser_manager import get_browser_manager, BrowserManagerError

logger = logging.getLogger(__name__)

# <<< REMOVED Global State and _close_existing_browser >>>
# _current_browser: Optional[Browser] = None
# _current_page: Optional[Page] = None
# async def _close_existing_browser(): ...


@skill(
    name="open_url",
    description="Opens the specified URL in a new browser instance (closes any previous one). Stores the page for subsequent actions.",
    parameters={
        "url": (str, ...),  # Ellipsis means the parameter is required
    },
)
async def open_url(url: str) -> Dict[str, Any]:
    """
    Opens the specified URL. Uses the shared BrowserManager.
    """
    logger.info(f"Attempting to open URL: {url}")
    manager = get_browser_manager()

    try:
        page = await manager.open_page(url)  # Handles closing old page/browser logic
        page_title = await page.title()
        success_message = f"Successfully navigated to '{url}'. Title: '{page_title}'. Browser ready for next actions."
        logger.info(success_message)
        return {
            "status": "success",
            "data": {"message": success_message, "title": page_title},
        }

    except BrowserManagerError as e:
        # Errors during browser/page opening are handled by manager
        logger.error(
            f"BrowserManagerError opening URL '{url}': {e}", exc_info=False
        )  # Log concise error
        return {"status": "error", "data": {"message": str(e)}}
    except Exception as e:
        # Catch any other unexpected errors
        error_message = (
            f"Unexpected error opening URL '{url}': {type(e).__name__} - {e}"
        )
        logger.error(error_message, exc_info=True)
        # Attempt cleanup via manager in case of unexpected error
        await manager.close_browser()
        return {"status": "error", "data": {"message": error_message}}


@skill(
    name="click_element",
    description="Clicks an element on the currently open web page specified by a CSS selector.",
    parameters={
        "selector": (str, ...),
    },
)
async def click_element(selector: str) -> Dict[str, Any]:
    """
    Clicks an element using the shared BrowserManager's current page.
    """
    logger.info(f"Attempting to click element with selector: {selector}")
    manager = get_browser_manager()

    try:
        page = await manager.get_current_page()
        if not page:
            return {
                "status": "error",
                "data": {
                    "message": "No page currently open or accessible. Use 'open_url' first."
                },
            }

        # Increased timeout slightly for clicking
        await page.locator(selector).click(timeout=15000)
        success_message = f"Successfully clicked element with selector: {selector}"
        logger.info(success_message)
        return {"status": "success", "data": {"message": success_message}}
    except BrowserManagerError as e:
        logger.error(f"BrowserManagerError checking for page: {e}", exc_info=False)
        return {"status": "error", "data": {"message": str(e)}}
    except Exception as e:
        # Errors from Playwright (e.g., selector not found, timeout)
        error_message = f"Error clicking element '{selector}': {type(e).__name__} - {e}"
        logger.error(
            error_message, exc_info=True
        )  # Log full traceback for playwright errors
        return {"status": "error", "data": {"message": error_message}}


@skill(
    name="fill_form_field",
    description="Fills a form field on the currently open web page, identified by a CSS selector, with the specified value.",
    parameters={
        "selector": (str, ...),
        "value": (str, ...),
    },
)
async def fill_form_field(selector: str, value: str) -> Dict[str, Any]:
    """
    Fills a form field using the shared BrowserManager's current page.
    """
    logger.info(f"Attempting to fill field '{selector}' with value: '{value[:50]}...'")
    manager = get_browser_manager()

    try:
        page = await manager.get_current_page()
        if not page:
            return {
                "status": "error",
                "data": {
                    "message": "No page currently open or accessible. Use 'open_url' first."
                },
            }

        # Use fill for inputs, timeout might be needed
        await page.locator(selector).fill(value, timeout=15000)
        success_message = f"Successfully filled field '{selector}'."
        logger.info(success_message)
        return {"status": "success", "data": {"message": success_message}}
    except BrowserManagerError as e:
        logger.error(f"BrowserManagerError checking for page: {e}", exc_info=False)
        return {"status": "error", "data": {"message": str(e)}}
    except Exception as e:
        error_message = f"Error filling field '{selector}': {type(e).__name__} - {e}"
        logger.error(error_message, exc_info=True)
        return {"status": "error", "data": {"message": error_message}}


@skill(
    name="get_page_content",
    description="Retrieves the HTML content of the currently open page, optionally filtered by a CSS selector.",
    parameters={
        "selector": (str, None), # Type is str, default is None making it optional
    },
)
async def get_page_content(selector: Optional[str] = None) -> Dict[str, Any]:
    """
    Gets HTML content using the shared BrowserManager's current page.
    """
    logger.info(f"Attempting to get page content. Selector: {selector}")
    manager = get_browser_manager()

    try:
        page = await manager.get_current_page()
        if not page:
            return {
                "status": "error",
                "data": {
                    "message": "No page currently open or accessible. Use 'open_url' first."
                },
            }

        html_content = ""
        if selector:
            html_content = await page.locator(selector).first.inner_html(timeout=10000)
            logger.info(
                f"Retrieved inner HTML for selector '{selector}'. Length: {len(html_content)}"
            )
        else:
            html_content = await page.content()
            logger.info(
                f"Retrieved full page HTML content. Length: {len(html_content)}"
            )

        log_content_preview = html_content[:500].replace("\n", " ") + (
            "..." if len(html_content) > 500 else ""
        )
        logger.debug(f"Content Preview: {log_content_preview}")

        return {"status": "success", "data": {"html_content": html_content}}
    except BrowserManagerError as e:
        logger.error(f"BrowserManagerError checking for page: {e}", exc_info=False)
        return {"status": "error", "data": {"message": str(e)}}
    except Exception as e:
        error_message = f"Error getting page content (selector: {selector}): {type(e).__name__} - {e}"
        logger.error(error_message, exc_info=True)
        return {"status": "error", "data": {"message": error_message}}


@skill(
    name="close_browser",
    description="Closes the currently open browser instance managed by the browser skill.",
    parameters={},
)
async def close_browser() -> Dict[str, Any]:
    """
    Closes the browser using the shared BrowserManager.
    """
    logger.info("Attempting to close browser via BrowserManager.")
    manager = get_browser_manager()
    try:
        await manager.close_browser()
        logger.info("BrowserManager close_browser called successfully.")
        return {
            "status": "success",
            "data": {"message": "Browser closed successfully or was not open."},
        }
    except BrowserManagerError as e:
        logger.error(f"BrowserManagerError during close_browser: {e}", exc_info=False)
        # Even if manager reports error, likely best effort was made.
        return {
            "status": "warning",
            "data": {"message": f"Error reported during browser close: {e}"},
        }
    except Exception as e:
        # Catch any other unexpected errors
        error_message = f"Unexpected error closing browser: {type(e).__name__} - {e}"
        logger.error(error_message, exc_info=True)
        return {"status": "error", "data": {"message": error_message}}
