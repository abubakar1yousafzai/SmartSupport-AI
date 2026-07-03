"""
FastAPI Backend Server
======================
Exposes endpoints for:
  - Fetching issues, active issues, and repeat issues from SQLite
  - Fetching dashboard metrics
  - Updating issue status (triggering pipeline)
  - Submitting customer complaints (generating support reply, saving, and triggering pipeline)
"""

import os
import sys
import json
import asyncio

# ─────────────────────────────────────────────
# Step 1: Add project root to sys.path
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────
# Step 2: Load environment variables
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ─────────────────────────────────────────────
# Step 3: Import FastAPI and required modules
# ─────────────────────────────────────────────
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Import project modules
from agents.support_agent import draft_support_reply
from mcp_tools.data_tool import (
    fetch_all_issues,
    fetch_active_issues,
    get_repeat_issues,
    update_issue_status,
    save_issue,
    # Session / conversation management tools (new)
    create_session,
    get_conversation_history,
    save_conversation_message,
)
from orchestrator.orchestrator_agent import run_pipeline

# Frontend directory location
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

app = FastAPI(title="Customer Insight API Server")

# Serve static files from frontend directory
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))

@app.get("/customer")
def serve_customer():
    return FileResponse(os.path.join(FRONTEND_DIR, "customer.html"))

# ─────────────────────────────────────────────
# Step 4: CORS Middleware configuration
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Step 5: Request/Response Schemas
# ─────────────────────────────────────────────
class UpdateStatusRequest(BaseModel):
    issue_id: int
    status: str

class SubmitIssueRequest(BaseModel):
    complaint_text: str
    # Optional session tracking fields (new)
    session_id: str = None        # Session ID from GET /api/session
    is_followup: bool = False     # True when customer is replying to a previous answer

# ─────────────────────────────────────────────
# Step 6: Endpoints
# ─────────────────────────────────────────────

@app.get("/api/session")
def get_session():
    """
    Creates a new conversation session and returns its unique session_id.

    The session_id must be sent back in every subsequent
    POST /api/submit-issue-stream request so the server can link
    conversation turns and issues together.

    Returns:
        dict: {"session_id": "sess_xxxxxxxx"}
    """
    session_id = create_session()
    return {"session_id": session_id}

@app.get("/api/issues")
def get_issues():
    """
    Fetch all issues from SQLite and return as JSON.
    """
    try:
        raw_data = fetch_all_issues()
        return json.loads(raw_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch issues: {str(e)}")

@app.get("/api/active-issues")
def get_active_issues():
    """
    Fetch active (Open/In Progress) issues from SQLite and return as JSON.
    """
    try:
        raw_data = fetch_active_issues()
        return json.loads(raw_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch active issues: {str(e)}")

@app.get("/api/dashboard")
def get_dashboard():
    """
    Read data/dashboard_data.json and return as JSON.
    Returns an empty dict if the file is not found.
    """
    dashboard_path = os.path.join(PROJECT_ROOT, "data", "dashboard_data.json")
    if not os.path.exists(dashboard_path):
        return {}
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dashboard data: {str(e)}")

@app.get("/api/repeat-issues")
def get_recurring_issues():
    """
    Fetch repeat/recurring issues from SQLite and return as JSON.
    """
    try:
        raw_data = get_repeat_issues()
        return json.loads(raw_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repeat issues: {str(e)}")

@app.post("/api/update-status")
async def update_status(payload: UpdateStatusRequest, background_tasks: BackgroundTasks):
    """
    Update the status of an issue and trigger the pipeline.
    """
    try:
        result_message = update_issue_status(payload.issue_id, payload.status)
        if "Success" in result_message:
            # Trigger pipeline in the background to avoid blocking the API response
            background_tasks.add_task(run_pipeline)
            return {"success": True, "message": result_message}
        else:
            return {"success": False, "message": result_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")

@app.post("/api/submit-issue")
async def submit_issue(payload: SubmitIssueRequest, background_tasks: BackgroundTasks):
    """
    Submit a customer complaint.
    1. Drafts a support reply using support_agent.
    2. Saves the issue to SQLite using save_issue().
    3. Triggers run_pipeline() after saving.
    Returns agent reply and issue ID.
    """
    try:
        # 1. Draft the support reply from support_agent
        reply_dict = await draft_support_reply(payload.complaint_text)
        if "error" in reply_dict:
            raise HTTPException(status_code=500, detail=reply_dict["error"])

        agent_reply = reply_dict.get("draft_reply", "Thank you for reaching out.")
        
        # 2. Derive fields for SQLite storage
        # Derive category from suggested_department
        category = reply_dict.get("suggested_department", "General").strip()
        
        # Derive sentiment from tone
        tone = reply_dict.get("tone", "").strip()
        sentiment_mapping = {
            "Empathetic": "Negative",
            "Apologetic": "Negative",
            "Informative": "Neutral",
        }
        sentiment = sentiment_mapping.get(tone, "Neutral")
        
        # Get priority
        priority = reply_dict.get("priority")

        # Get status
        status = reply_dict.get("status", "Open")

        # 3. Save issue to SQLite
        save_result = save_issue(
            complaint_text=payload.complaint_text,
            agent_reply=agent_reply,
            category=category,
            sentiment=sentiment,
            priority=priority,
            status=status,
        )
        
        save_data = json.loads(save_result)
        if "error" in save_data:
            raise HTTPException(status_code=500, detail=save_data["error"])
            
        issue_id = save_data.get("issue_id")
        
        # 4. Trigger pipeline in the background only if escalate is True
        escalate = reply_dict.get("escalate") == True
        if escalate:
            background_tasks.add_task(run_pipeline)
        
        return {
            "agent_reply": agent_reply,
            "issue_id": issue_id,
            "escalate": escalate,
            "classification": reply_dict.get("classification"),
            "status": status
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to submit issue: {str(e)}")

# ─────────────────────────────────────────────
# Step 7: Streaming Endpoint (SSE)
# ─────────────────────────────────────────────

@app.post("/api/submit-issue-stream")
async def submit_issue_stream(payload: SubmitIssueRequest, background_tasks: BackgroundTasks):
    """
    Streaming endpoint for customer complaint submission using Server-Sent Events (SSE).

    Stream order:
      1. 'metadata' event  – classification details (category, sentiment, priority, etc.)
      2. 'token'   events  – reply text word-by-word with a 30 ms delay between each word
      3. 'done'    event   – issue_id after the record is persisted to SQLite

    Conversation context (new):
      - If payload.session_id is provided, the full conversation history for that
        session is fetched and prepended to the complaint before calling the agent.
      - The customer message and agent reply are saved back to the conversations table.
      - The saved issue is linked to the session via session_id.

    If escalate=True, run_pipeline() is triggered as a background task after streaming ends.
    """

    async def event_generator():
        try:
            # Unpack request fields for convenience
            complaint_text = payload.complaint_text
            session_id     = payload.session_id      # may be None

            # ── Step A: Fetch conversation history for this session ───────────
            # Only attempted when the caller passes a session_id.
            history = None
            if session_id:
                history_raw = get_conversation_history(session_id)
                history = json.loads(history_raw)

            # ── Step B: Build complaint text enriched with conversation context ─
            # When prior turns exist we prepend them so the agent has full context.
            if history and len(history.get("messages", [])) > 0:
                context = "[CONVERSATION HISTORY]\n"
                for msg in history["messages"]:
                    role = "Customer" if msg["role"] == "customer" else "Agent"
                    context += f"{role}: {msg['message']}\n"

                # If the last issue is still waiting on the customer and they are
                # replying, signal to the agent that the previous fix didn't work.
                last_issue = history.get("last_issue")
                if last_issue and last_issue["status"] == "Pending Customer Action":
                    context += "\n[CURRENT SITUATION]\n"
                    context += f"Previous issue #{last_issue['issue_id']} "
                    context += f"was marked '{last_issue['status']}'\n"
                    context += "Customer is following up — previous fix did NOT work.\n"
                    context += "Classify as: followup_unresolved\n"

                complaint_with_context = f"{context}\n[NEW MESSAGE]\nCustomer: {complaint_text}"
            else:
                # No prior history – treat as a fresh complaint
                complaint_with_context = complaint_text

            # ── Step C: Call the AI agent with the enriched complaint ─────────
            reply_dict = await draft_support_reply(complaint_with_context)
            if "error" in reply_dict:
                # Surface the error as an SSE error event so the client can react
                error_payload = json.dumps({"type": "error", "message": reply_dict["error"]})
                yield f"data: {error_payload}\n\n"
                return

            agent_reply_text = reply_dict.get("draft_reply", "Thank you for reaching out.")
            category         = reply_dict.get("suggested_department", "General").strip()

            # Map tone → human-readable sentiment
            tone = reply_dict.get("tone", "").strip()
            sentiment_mapping = {
                "Empathetic":  "Negative",
                "Apologetic":  "Negative",
                "Informative": "Neutral",
            }
            sentiment = sentiment_mapping.get(tone, "Neutral")
            priority  = reply_dict.get("priority", "Medium")
            status    = reply_dict.get("status", "Open")
            escalate  = reply_dict.get("escalate") is True

            # ── 1. Stream metadata first ──────────────────────────────────────
            metadata_payload = json.dumps({
                "type":      "metadata",
                "category":  category,
                "sentiment": sentiment,
                "priority":  priority,
                "escalate":  escalate,
                "status":    status,
            })
            yield f"data: {metadata_payload}\n\n"

            # ── Step D: Persist the customer's raw message to conversations ───
            # Saved BEFORE streaming the reply so the DB is updated promptly.
            if session_id:
                save_conversation_message(session_id, "customer", complaint_text)

            # ── 2. Stream reply text word-by-word ─────────────────────────────
            words = agent_reply_text.split()
            for word in words:
                token_payload = json.dumps({"type": "token", "word": word})
                yield f"data: {token_payload}\n\n"
                # Small delay for a smooth typewriter effect (30 ms)
                await asyncio.sleep(0.03)

            # ── Step E: Persist issue to SQLite (now linked to the session) ───
            save_result = save_issue(
                complaint_text=complaint_text,
                agent_reply=agent_reply_text,
                category=category,
                sentiment=sentiment,
                priority=priority,
                status=status,
                session_id=session_id,   # links issue ↔ session
            )
            save_data = json.loads(save_result)
            issue_id  = save_data.get("issue_id")

            done_payload = json.dumps({"type": "done", "issue_id": issue_id})
            yield f"data: {done_payload}\n\n"

            # ── Step F: Persist agent reply to conversations ──────────────────
            if session_id:
                save_conversation_message(
                    session_id,
                    "agent",
                    agent_reply_text,
                    issue_id=issue_id,
                )

            # ── 3. Background pipeline if escalation is required ──────────────
            if escalate:
                background_tasks.add_task(run_pipeline)

        except Exception as e:
            # Yield a terminal error event so the client can display a message
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Prevent intermediate proxies / browsers from buffering the stream
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ─────────────────────────────────────────────
# Step 8: Main Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
