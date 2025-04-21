from typing import Optional, Any, Dict, List
import logging
from a3x.core.context import SharedTaskContext

logger = logging.getLogger(__name__)

class ContextAccessor:
    """
    An abstraction layer for accessing and updating data in SharedTaskContext.
    This class decouples Fragments from the internal structure of the context,
    providing standardized methods for common operations, aligning with the principles
    of 'Hierarquia Cognitiva em Pirâmide' for structured data flow and 'Fragmentação Cognitiva'
    for minimal context dependency.
    """
    def __init__(self, shared_task_context: Optional[SharedTaskContext] = None):
        self._context = shared_task_context
        logger.info("ContextAccessor initialized.")

    def set_context(self, shared_task_context: SharedTaskContext) -> None:
        """
        Sets or updates the SharedTaskContext instance to be accessed.
        
        Args:
            shared_task_context: The SharedTaskContext instance to use.
        """
        self._context = shared_task_context
        if shared_task_context:
             # Use correct attribute from dataclass
             logger.info(f"ContextAccessor updated with task ID: {shared_task_context.task_id}")
        else:
             logger.info("ContextAccessor context cleared.")

    def get_context(self) -> Optional[SharedTaskContext]:
        """
        Retrieves the current SharedTaskContext instance.
        
        Returns:
            The SharedTaskContext instance if available, otherwise None.
        """
        return self._context

    def get_last_read_file(self) -> Optional[str]:
        """
        Retrieves the path of the last read file from the context.
        
        Returns:
            The path of the last read file if available, otherwise None.
        """
        if not self._context:
            logger.warning("No SharedTaskContext available for get_last_read_file.")
            return None
        value = self._context.get("last_file_read_path")
        if value is not None:
            logger.info(f"Retrieved last read file: {value}")
        return value

    async def set_last_read_file(self, file_path: str) -> None:
        """
        Sets the path of the last read file in the context.
        
        Args:
            file_path: The path of the file that was last read.
        """
        if not self._context:
            logger.warning("No SharedTaskContext available for set_last_read_file.")
            return
        await self._context.update_data("last_file_read_path", file_path)
        logger.info(f"Set last read file: {file_path}")

    def get_task_objective(self) -> Optional[str]:
        """
        Retrieves the current task objective from the context.
        
        Returns:
            The task objective if available, otherwise None.
        """
        if not self._context:
            logger.warning("No SharedTaskContext available for get_task_objective.")
            return None
        return self._context.initial_objective

    async def set_task_result(self, key: str, result: Any, tags: Optional[List[str]] = None, source: Optional[str] = None) -> None:
        """
        Sets a result or data point in the context associated with the task.
        NOTE: tags and source are ignored by the current SharedTaskContext.update_data method.

        Args:
            key: The key to store the result under.
            result: The result or data to store.
            tags: Optional list of tags to categorize the data (ignored).
            source: Optional source identifier for the data (ignored).
        """
        if not self._context:
            logger.warning("No SharedTaskContext available for set_task_result.")
            return
        await self._context.update_data(key, result)
        logger.info(f"Set task result for key '{key}' (source/tags ignored)")

    async def get_task_data(self, key: str) -> Optional[Any]:
        """
        Retrieves data from the context by key. This is now async.
        
        Args:
            key: The key to look up in the context.
        
        Returns:
            The data associated with the key if found, otherwise None.
        """
        if not self._context:
            logger.warning("No SharedTaskContext available for get_task_data.")
            return None
        value = await self._context.get_data(key)
        if value is not None:
            logger.info(f"Retrieved task data for key '{key}'")
        return value

    def get_data_by_tag(self, tag: str) -> Dict[str, Any]:
        """
        Retrieves all data entries from the context that have a specific tag.
        
        Args:
            tag: The tag to search for.
        
        Returns:
            A dictionary of key-value pairs that match the tag.
        """
        if not self._context:
            logger.warning("No SharedTaskContext available for get_data_by_tag.")
            return {}
        entries = self._context.get_entries_by_tag(tag)
        logger.info(f"Retrieved {len(entries)} entries with tag '{tag}'")
        return entries

    def get_data_by_source(self, source: str) -> Dict[str, Any]:
        """
        Retrieves all data entries from the context that originate from a specific source.
        
        Args:
            source: The source identifier to search for.
        
        Returns:
            A dictionary of key-value pairs from the specified source.
        """
        if not self._context:
            logger.warning("No SharedTaskContext available for get_data_by_source.")
            return {}
        entries = self._context.get_entries_by_source(source)
        logger.info(f"Retrieved {len(entries)} entries from source '{source}'")
        return entries

    async def get_last_history_result(self) -> Optional[Any]:
        """
        Retrieves the result summary of the last executed fragment from the history.

        Returns:
            The result summary of the last entry in the execution history if available,
            otherwise None.
        """
        if not self._context:
            logger.warning("No SharedTaskContext available for get_last_history_result.")
            return None
        
        history = await self._context.get_history()
        if history:
            last_entry = history[-1]
            if len(last_entry) > 1:
                # Assuming history stores (fragment_name, result_summary) tuples
                logger.info(f"Retrieved last history result: {last_entry[1]}")
                return last_entry[1]
            else:
                logger.warning(f"Last history entry has unexpected format: {last_entry}")
                return None
        else:
            logger.info("Execution history is empty.")
            return None 