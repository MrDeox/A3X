
from a3x.fragments.base import BaseFragment
from a3x.context import Context

class ValidTestFragment(BaseFragment):
    def execute(self, ctx: Context):
        return {"status": "success", "message": "Executed valid fragment"}
