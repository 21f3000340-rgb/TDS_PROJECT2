from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent
from dotenv import load_dotenv
import os
import time

from shared_store import url_time, BASE64_STORE

# Load .env values
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

@app.get("/healthz")
def healthz():
    """Simple liveness check."""
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME)
    }


@app.post("/solve")
async def solve(request: Request, background_tasks: BackgroundTasks):
    """
    Start the autonomous agent in the background.
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not data:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Extract incoming fields
    url = data.get("url")
    secret = data.get("secret")

    # Validate minimal fields
    if not url or not secret:
        raise HTTPException(status_code=400, detail="Missing url or secret")

    # Secret check
    if secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Clear state stores
    url_time.clear()
    BASE64_STORE.clear()

    print("Verified request. Starting quiz-solving agent...")

    # Pass FULL data (email, secret, url, etc.) to the agent
    background_tasks.add_task(run_agent, data)

    # Immediate response
    return JSONResponse({"status": "ok"}, status_code=200)


# Local dev mode (Render uses CMD in Dockerfile)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
