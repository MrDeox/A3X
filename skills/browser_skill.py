import asyncio
from playwright.async_api import async_playwright, Page, Browser
import logging
from typing import Dict, Any, Optional

from core.tools import skill

logger = logging.getLogger(__name__)

# --- State Management (Simple Global Variables - Not Thread Safe!) ---
# Stores the currently active browser and page instance
_current_browser: Optional[Browser] = None
_current_page: Optional[Page] = None
# --------------------------------------------------------------------

async def _close_existing_browser():
    """Internal function to close the browser if it's open."""
    global _current_browser, _current_page
    if _current_browser:
        logger.info("Closing existing browser instance.")
        try:
            await _current_browser.close()
        except Exception as e:
            logger.warning(f"Error closing existing browser: {e}")
        _current_browser = None
        _current_page = None

@skill(
    name="open_url",
    description="Opens the specified URL in a new browser instance (closes any previous one). Stores the page for subsequent actions.",
    parameters={
        "url": (str, ...), # Ellipsis means the parameter is required
    }
)
async def open_url(url: str) -> Dict[str, Any]:
    """
    Opens the specified URL in a new headless browser instance. Closes any previously opened browser by this skill.
    Stores the page reference for use by other browser skills like click_element, fill_form_field, etc.

    Args:
        url (str): The URL to open. Must be a complete URL (e.g., 'https://www.google.com').

    Returns:
        Dict[str, Any]: A dictionary containing status ('success' or 'error') and data (e.g., page title or error message).
    """
    global _current_browser, _current_page
    logger.info(f"Attempting to open URL: {url}")

    # Close any existing browser before opening a new one
    await _close_existing_browser()

    try:
        p = await async_playwright().start()
        # Using Chromium, headless=True by default now in Playwright, but being explicit
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Navigating to URL: {url}")
        await page.goto(url, timeout=60000, wait_until='domcontentloaded') # 60 seconds timeout

        page_title = await page.title()

        # Store browser and page globally
        _current_browser = browser
        _current_page = page

        success_message = f"Successfully navigated to '{url}'. Title: '{page_title}'. Browser ready for next actions."
        logger.info(success_message)
        return {"status": "success", "data": {"message": success_message, "title": page_title}}

    except Exception as e:
        error_message = f"Error opening URL '{url}': {type(e).__name__} - {e}"
        logger.error(error_message, exc_info=True)
        await _close_existing_browser() # Ensure cleanup on error
        return {"status": "error", "data": {"message": error_message}}

@skill(
    name="click_element",
    description="Clicks an element on the currently open web page specified by a CSS selector.",
    parameters={
        "selector": (str, ...),
    }
)
async def click_element(selector: str) -> Dict[str, Any]:
    """
    Clicks an element specified by a CSS selector on the page previously opened by 'open_url'.

    Args:
        selector (str): The CSS selector to identify the element to click.

    Returns:
        Dict[str, Any]: Dictionary with status ('success' or 'error') and a message.
    """
    global _current_page
    logger.info(f"Attempting to click element with selector: {selector}")

    if not _current_page:
        return {"status": "error", "data": {"message": "No page currently open. Use 'open_url' first."}}

    try:
        # Increased timeout slightly for clicking
        await _current_page.locator(selector).click(timeout=15000)
        success_message = f"Successfully clicked element with selector: {selector}"
        logger.info(success_message)
        return {"status": "success", "data": {"message": success_message}}
    except Exception as e:
        error_message = f"Error clicking element '{selector}': {type(e).__name__} - {e}"
        logger.error(error_message)
        # Do not close browser here, let the agent decide
        return {"status": "error", "data": {"message": error_message}}

@skill(
    name="fill_form_field",
    description="Fills a form field on the currently open web page, identified by a CSS selector, with the specified value.",
    parameters={
        "selector": (str, ...),
        "value": (str, ...),
    }
)
async def fill_form_field(selector: str, value: str) -> Dict[str, Any]:
    """
    Fills a form field (e.g., input, textarea) specified by a CSS selector with the given value on the page opened by 'open_url'.

    Args:
        selector (str): The CSS selector for the form field.
        value (str): The text value to fill into the field.

    Returns:
        Dict[str, Any]: Dictionary with status ('success' or 'error') and a message.
    """
    global _current_page
    logger.info(f"Attempting to fill field '{selector}' with value: '{value[:50]}...'") # Log truncated value

    if not _current_page:
        return {"status": "error", "data": {"message": "No page currently open. Use 'open_url' first."}}

    try:
        # Use fill for inputs, timeout might be needed
        await _current_page.locator(selector).fill(value, timeout=15000)
        success_message = f"Successfully filled field '{selector}'."
        logger.info(success_message)
        return {"status": "success", "data": {"message": success_message}}
    except Exception as e:
        error_message = f"Error filling field '{selector}': {type(e).__name__} - {e}"
        logger.error(error_message)
        return {"status": "error", "data": {"message": error_message}}

@skill(
    name="get_page_content",
    description="Retrieves the HTML content of the currently open page, optionally filtered by a CSS selector.",
    parameters={
        "selector": (Optional[str], None), # Optional parameter, defaults to None
    }
)
async def get_page_content(selector: Optional[str] = None) -> Dict[str, Any]:
    """
    Gets the HTML content of the current page opened by 'open_url'.
    If a selector is provided, gets the inner HTML of the element(s) matching the selector.
    Otherwise, returns the full page HTML source.

    Args:
        selector (Optional[str]): CSS selector to get HTML for a specific part of the page. Defaults to None (full page).

    Returns:
        Dict[str, Any]: Dictionary with status ('success' or 'error') and the requested HTML content in data['html_content'] or an error message.
    """
    global _current_page
    logger.info(f"Attempting to get page content. Selector: {selector}")

    if not _current_page:
        return {"status": "error", "data": {"message": "No page currently open. Use 'open_url' first."}}

    try:
        html_content = ""
        if selector:
            # Get inner HTML of the first matching element
            # Consider how to handle multiple matches if needed later
            html_content = await _current_page.locator(selector).first.inner_html(timeout=10000)
            logger.info(f"Retrieved inner HTML for selector '{selector}'. Length: {len(html_content)}")
        else:
            html_content = await _current_page.content()
            logger.info(f"Retrieved full page HTML content. Length: {len(html_content)}")

        # Basic truncation for logging, return full content in response
        log_content_preview = html_content[:500].replace('\n', ' ') + ('...' if len(html_content) > 500 else '')
        logger.debug(f"Content Preview: {log_content_preview}")

        # Return potentially large content. The agent/LLM needs to handle it.
        return {"status": "success", "data": {"html_content": html_content}}
    except Exception as e:
        error_message = f"Error getting page content (selector: {selector}): {type(e).__name__} - {e}"
        logger.error(error_message)
        return {"status": "error", "data": {"message": error_message}}


@skill(
    name="close_browser",
    description="Closes the currently open browser instance managed by the browser skill.",
    parameters={}
)
async def close_browser() -> Dict[str, Any]:
    """
    Closes the browser instance previously opened by 'open_url'.

    Returns:
        Dict[str, Any]: Dictionary indicating success or if no browser was open.
    """
    logger.info("Attempting to close browser.")
    closed = await _close_existing_browser()
    if _current_browser is None: # Check state after _close_existing_browser runs
         logger.info("Browser closed successfully or was already closed.")
         return {"status": "success", "data": {"message": "Browser closed successfully or was not open."}}
    else:
         # This case should ideally not happen if _close_existing_browser works
         logger.warning("Browser state indicates it might not have closed properly.")
         return {"status": "warning", "data": {"message": "Attempted to close browser, but state is inconsistent."}}

