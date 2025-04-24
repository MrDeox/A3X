
from a3x.fragments.base import BaseFragment
from typing import Any

class ValidTestFragment(BaseFragment):
    async def execute(self, ctx: Any):
        return {"status": "success", "message": "Executed valid fragment"}
