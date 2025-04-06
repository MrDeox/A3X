# core/browser_manager.py
import asyncio
import logging
from playwright.async_api import async_playwright, Browser, Page, Playwright
from typing import Optional

logger = logging.getLogger(__name__)


class BrowserManagerError(Exception):
    """Custom exception for browser manager errors."""

    pass


class BrowserManager:
    """Manages a single headless browser instance using Playwright."""

    _instance: Optional["BrowserManager"] = None
    _playwright_context: Optional[Playwright] = None
    _browser: Optional[Browser] = None
    _page: Optional[Page] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        # Basic Singleton pattern (consider a more robust dependency injection later)
        if cls._instance is None:
            cls._instance = super(BrowserManager, cls).__new__(cls)
            # Initialize attributes that should only be set once
            cls._playwright_context = None
            cls._browser = None
            cls._page = None
            cls._lock = asyncio.Lock()
        return cls._instance

    async def _ensure_playwright_started(self):
        """Starts the Playwright context if not already started."""
        async with self._lock:
            if self._playwright_context is None:
                try:
                    logger.info("Starting Playwright context...")
                    self._playwright_context = await async_playwright().start()
                    logger.info("Playwright context started successfully.")
                except Exception as e:
                    logger.critical(
                        f"Failed to start Playwright context: {e}", exc_info=True
                    )
                    raise BrowserManagerError(
                        f"Failed to initialize Playwright: {e}"
                    ) from e

    async def _ensure_browser_launched(self):
        """Launches the browser if not already launched."""
        await self._ensure_playwright_started()
        if self._browser is None:
            async with self._lock:
                # Double-check inside lock
                if self._browser is None:
                    if self._playwright_context is None:
                        # This should not happen if _ensure_playwright_started worked
                        raise BrowserManagerError(
                            "Playwright context is None, cannot launch browser."
                        )
                    try:
                        logger.info("Launching headless Chromium browser...")
                        self._browser = await self._playwright_context.chromium.launch(
                            headless=True
                        )
                        logger.info("Browser launched successfully.")
                    except Exception as e:
                        logger.error(f"Failed to launch browser: {e}", exc_info=True)
                        await self.close_browser()  # Attempt cleanup
                        raise BrowserManagerError(
                            f"Failed to launch browser: {e}"
                        ) from e

    async def open_page(self, url: str) -> Page:
        """Opens a new page (closing any existing one) and navigates to the URL."""
        async with self._lock:
            await self.close_page()  # Close existing page first
            await self._ensure_browser_launched()

            if self._browser is None:
                raise BrowserManagerError("Browser is not launched, cannot open page.")

            try:
                logger.info(f"Opening new page and navigating to: {url}")
                self._page = await self._browser.new_page()
                await self._page.goto(url, timeout=60000, wait_until="domcontentloaded")
                page_title = await self._page.title()
                logger.info(
                    f"Successfully navigated to '{url}'. Title: '{page_title}'."
                )
                return self._page
            except Exception as e:
                error_message = f"Error opening URL '{url}': {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)
                await self.close_page()  # Close the failed page
                # Optionally close the whole browser on navigation failure?
                # await self.close_browser()
                raise BrowserManagerError(error_message) from e

    async def get_current_page(self) -> Optional[Page]:
        """Returns the currently managed page, if one exists."""
        async with self._lock:
            # Basic check, more robust checks might be needed (e.g., page.is_closed())
            if self._page and not self._page.is_closed():
                return self._page
            elif self._page and self._page.is_closed():
                logger.warning("Attempted to get current page, but it was closed.")
                self._page = None  # Clear the stale reference
                return None
            else:
                return None

    async def close_page(self):
        """Closes the currently managed page, if it exists."""
        async with self._lock:
            if self._page and not self._page.is_closed():
                logger.info("Closing current page.")
                try:
                    await self._page.close()
                except Exception as e:
                    logger.warning(f"Error closing current page: {e}", exc_info=True)
                finally:
                    self._page = None
            elif self._page:
                # Already closed, just clear reference
                self._page = None

    async def close_browser(self):
        """Closes the browser instance and the Playwright context."""
        async with self._lock:
            await self.close_page()  # Ensure page is closed first
            browser_closed = False
            if self._browser:
                logger.info("Closing browser instance...")
                try:
                    await self._browser.close()
                    browser_closed = True
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}", exc_info=True)
                finally:
                    self._browser = None

            playwright_stopped = False
            if self._playwright_context:
                logger.info("Stopping Playwright context...")
                try:
                    await self._playwright_context.stop()
                    playwright_stopped = True
                except Exception as e:
                    logger.warning(
                        f"Error stopping Playwright context: {e}", exc_info=True
                    )
                finally:
                    self._playwright_context = None
                    # Reset singleton instance if fully closed?
                    # BrowserManager._instance = None

            logger.info(
                f"Browser close attempt finished. Browser closed: {browser_closed}, Playwright stopped: {playwright_stopped}"
            )


# --- Singleton Instance ---
# Provide a way to get the single instance
_browser_manager_instance = BrowserManager()


def get_browser_manager() -> BrowserManager:
    """Returns the singleton instance of the BrowserManager."""
    # This ensures the instance is created if accessed for the first time here
    return BrowserManager()


# --- Cleanup Function ---
# Optional: Register a cleanup function to run on application exit
async def cleanup_browser_manager():
    logger.info("Running cleanup for BrowserManager...")
    await get_browser_manager().close_browser()


# Example of registering with atexit (sync version needed or run async cleanup)
# import atexit
# atexit.register(lambda: asyncio.run(cleanup_browser_manager()))
