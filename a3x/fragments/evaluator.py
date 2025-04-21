import asyncio
import logging
import json
from typing import Optional, Dict

from .base import BaseFragment
from a3x.core.context import FragmentContext

logger = logging.getLogger(__name__)

REFACTOR_REWARD = 10
MUTATION_REWARD = 5

class EvaluatorFragment(BaseFragment):
    """A fragment that observes system messages and provides rewards for successful actions."""

    async def get_purpose(self, context: Optional[Dict] = None) -> str:
        """Returns a description of this fragment's purpose."""
        return "Observe system messages (like refactor_result, mutation_attempt) and issue rewards for successful actions to reinforce positive behaviors."

    async def handle_realtime_chat(self, message: dict, context: FragmentContext):
        """Processes incoming messages and posts rewards for successes."""
        message_type = message.get("type")
        sender = message.get("sender")
        content = message.get("content")

        if not isinstance(content, dict):
            logger.debug(f"[{self.get_name()}] Ignoring message with non-dict content from {sender}.")
            return
        
        target_fragment = None
        reason = ""
        reward_amount = 0
        status = content.get("status")

        # --- Evaluate different message types --- 

        if message_type == "refactor_result":
            if status == "success":
                target_fragment = sender
                target_path = content.get("target")
                original_action = content.get("original_action", "refactor action")
                reason = f"Successful '{original_action}' on target '{target_path}'"
                reward_amount = REFACTOR_REWARD
            elif status == "error":
                target_path = content.get('target')
                logger.info(f"[{self.get_name()}] Noted error result from {sender} for target {target_path}")
            else:
                target_path = content.get('target')
                logger.debug(f"[{self.get_name()}] Ignoring refactor_result from {sender} for target {target_path} with status: {status}")

        elif message_type == "mutation_attempt":
            if status == "success":
                target_fragment = sender
                target_path = content.get("target")
                reason = f"Successful mutation applied to '{target_path}'"
                reward_amount = MUTATION_REWARD
            elif status == "no_change":
                target_path = content.get('target')
                logger.info(f"[{self.get_name()}] Noted 'no_change' mutation attempt from {sender} for target {target_path}")
            elif status == "error":
                target_path = content.get('target')
                logger.info(f"[{self.get_name()}] Noted error mutation attempt from {sender} for target {target_path}")
            else:
                target_path = content.get('target')
                logger.debug(f"[{self.get_name()}] Ignoring mutation_attempt from {sender} for target {target_path} with status: {status}")

        elif message_type == "anomaly":
            summary = content.get('summary')
            logger.warning(f"[{self.get_name()}] Observed anomaly from {sender}: {summary}")

        # --- Post Reward if applicable --- 
        if target_fragment and reason and reward_amount > 0:
            reward_content = {
                "target": target_fragment,
                "amount": reward_amount,
                "reason": reason
            }
            logger.info(f"[{self.get_name()}] Posting reward for {target_fragment}: {reward_amount} coins. Reason: {reason}")
            try:
                await self.post_chat_message(
                    message_type="reward",
                    content={
                        "recipient": sender,
                        "reward_amount": reward_amount,
                        "reason": reason,
                        "original_status": status,
                        "original_target": target_path,
                        "subtask_id": context.subtask_id
                    },
                    target_fragment=target_fragment
                )
                logger.info(f"[{self.get_name()}] Posted reward of {reward_amount} to {sender} for status '{status}' on target '{target_path}'. Reason: {reason}")
            except Exception as e:
                logger.error(f"[{self.get_name()}] Failed to post reward message: {e}", exc_info=True)
        else:
            if message_type in ["refactor_result", "mutation_attempt"] and status != "success":
                logger.debug(f"[{self.get_name()}] No reward: Message type '{message_type}' from {sender}, status: {status}")

    # TODO: Add methods for tracking overall system performance or adjusting rewards? 