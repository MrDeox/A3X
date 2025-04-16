from fastapi import APIRouter

router = APIRouter()
 
@router.get("", tags=["General Status"])
async def get_api_status():
    """Returns the current status of the API itself."""
    return {"status": "running", "message": "A³X API is operational."} 