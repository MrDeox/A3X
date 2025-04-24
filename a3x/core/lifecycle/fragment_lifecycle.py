import asyncio
import logging
from typing import Optional, Callable, Coroutine, Any

class FragmentLifecycleManager:
    """
    Gerencia o ciclo de vida (start/stop) e o controle da task em background de um fragmento.
    """
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._is_running = asyncio.Event()
        self._background_task: Optional[asyncio.Task] = None

    def is_running(self) -> bool:
        return self._is_running.is_set()

    async def start(self, target_coro: Callable[[], Coroutine[Any, Any, None]], name: str = "fragment_execute"):
        if self.is_running():
            self._logger.warning(f"Fragment already started.")
            return
        self._logger.info(f"Starting fragment background task...")
        self._is_running.set()
        self._background_task = asyncio.create_task(self._run(target_coro), name=name)
        self._logger.info(f"Fragment background task started.")

    async def stop(self):
        if not self.is_running():
            self._logger.info(f"Fragment was not running.")
            return
        self._logger.info(f"Stopping fragment background task...")
        self._is_running.clear()
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
                self._logger.info(f"Fragment background task cancelled successfully.")
            except asyncio.CancelledError:
                self._logger.info(f"Fragment background task cancellation confirmed.")
            except Exception as e:
                self._logger.error(f"Error during fragment background task cancellation: {e}", exc_info=True)
        self._background_task = None
        self._logger.info(f"Fragment stopped.")

    async def _run(self, target_coro: Callable[[], Coroutine[Any, Any, None]]):
        try:
            await target_coro()
        except asyncio.CancelledError:
            self._logger.info("Fragment background task received cancellation.")
        except Exception as e:
            self._logger.error(f"Exception in fragment background task: {e}", exc_info=True)
