import asyncio
from a3x.fragments.fragment_context import FragmentContext
from a3x.fragments.shared_task_context import SharedTaskContext
from a3x.utils.logger import logger

class CoordinatorFragment:
    async def set_context(self, context: FragmentContext, shared_context: SharedTaskContext):
        await super().set_context(context, shared_context)
        # Temporarily disable automatic loop start for debugging dispatcher queue issues
        # self.main_loop_task = asyncio.create_task(self._main_loop(), name="CoordinatorLoop")
        logger.warning("CoordinatorFragment main loop DISABLED for debugging.")
        logger.info("CoordinatorFragment context set and main loop potentially started.") 