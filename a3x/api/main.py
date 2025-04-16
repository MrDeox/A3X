from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import the routers
from a3x.api.routes import status, agent # Assuming you have these files

app = FastAPI(
    title="Painel A³X – API",
    description="API for interacting with and monitoring the A³X agent.",
    version="0.1.0",
    docs_url="/api/docs",  # Set custom path for Swagger UI
    openapi_url="/api/openapi.json" # Set custom path for OpenAPI schema
)

# --- CORS Configuration ---
# List of origins that are allowed to make requests
# In production, restrict this to your actual frontend domain
origins = [
    "http://localhost:3000", # Default Next.js dev server
    "http://127.0.0.1:3000",
    # Add other origins if needed (e.g., deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Allows cookies (if needed)
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)
# --- End CORS Configuration ---

# Include routers with prefixes
# Note: The path in the router function (e.g., @router.get("")) is appended to the prefix
app.include_router(status.router, prefix="/api/v1/status") 
app.include_router(agent.router, prefix="/api/v1/agent")

# Optional: Add a root endpoint for basic API discoverability
@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "Welcome to the A³X Agent API", "docs": "/api/docs"}

# Example of how to run this directly (for testing)
if __name__ == "__main__":
    # Use the module string format for Uvicorn when running directly
    uvicorn.run("a3x.api.main:app", host="0.0.0.0", port=8000, reload=True) 