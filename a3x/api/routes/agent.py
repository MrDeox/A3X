from fastapi import APIRouter, Query, Depends, HTTPException, Body, BackgroundTasks
from typing import List, Optional
import asyncio
import sys
import os
from pydantic import BaseModel

# Assuming AgentState and get_log_events are correctly importable
from a3x.api.state import AGENT_STATE
from a3x.api.log_buffer import get_log_events

router = APIRouter()

# --- Agent State Endpoint --- #

@router.get("/state", 
            tags=["Agent State"], 
            summary="Get Current Agent State",
            response_description="The current live state of the A³X agent")
async def get_agent_state():
    """Returns the current live state of the agent, including task, status, recent actions, etc."""
    # Use __dict__ for easy serialization, maybe filter later if needed
    state_dict = AGENT_STATE.__dict__.copy()
    # Remove the internal singleton instance reference if present
    state_dict.pop('_instance', None)
    return state_dict

# --- Log Buffer Endpoint --- #

@router.get("/logs", 
            tags=["Logs"], 
            summary="Get Recent Log Events",
            response_description="A list of recent structured log events from the agent")
async def get_logs(limit: int = Query(default=50, 
                                     ge=1, 
                                     le=AGENT_STATE._instance.LOG_EVENTS.maxlen if AGENT_STATE._instance and hasattr(AGENT_STATE._instance, 'LOG_EVENTS') else 200, 
                                     description="Maximum number of log events to return.")):
    """Returns recent log events from the circular log buffer. Events are returned newest first."""
    events = get_log_events(limit)
    return {
        "requested_limit": limit,
        "actual_count": len(events),
        "events": events
    }

# --- Future Endpoints (Examples) --- #

# @router.get("/fragments", tags=["Agent Configuration"])
# async def list_fragments():
#     """Lists available fragments and their status."""
#     # Logic to get fragment info from FragmentRegistry
#     # Example: return FragmentRegistry.get_fragment_details()
#     return {"message": "Not implemented yet"} 