"""
Support Agent
=============
This agent receives an individual customer complaint and drafts a professional,
empathetic support reply using Google ADK and Gemini 2.5 Flash.

The reply is:
  - Sorry for the inconvenience
  - Solution focused
  - Naturally written by Gemini based on the complaint content

No reply templates or hardcoded phrases are used.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# ─────────────────────────────────────────────
# Step 1: Load environment variables from .env
# ─────────────────────────────────────────────
# Loads GEMINI_API_KEY from the .env file at the project root.
# This ensures no API keys are ever hardcoded in source code.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Please set it in your .env file."
    )

# Set the API key as an environment variable for the Google SDK to pick up
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ─────────────────────────────────────────────
# Step 2: Define the system instruction prompt
# ─────────────────────────────────────────────
# Instructs Gemini to write a reply naturally from the complaint content.
# No templates, phrases, or tone scripts are hardcoded here.
SYSTEM_PROMPT = """
You are a Senior Customer Support Specialist who decides how to 
respond to each complaint based on its type.

CLASSIFY the input as one of:
- "self_fixable": customer can likely fix it themselves 
  (slow loading, minor glitches, login confusion, cache issues)
- "technical_escalation": a backend/system problem the customer 
  CANNOT fix (payment failures, billing errors, data loss, 
  security issues, repeated crashes, account lockouts)
- "followup_unresolved": customer is replying that a previous 
  suggested fix did NOT work

RESPOND based on classification:

self_fixable (first report):
- Apologize briefly, give 2-4 clear numbered troubleshooting steps
- End with: "If this doesn't help, reply and I'll escalate to our 
  technical team right away."
- escalate = false, status = "Pending Customer Action"

technical_escalation (first report):
- Apologize sincerely, do NOT give troubleshooting steps
- State clearly: "I've forwarded this to our technical team for 
  urgent investigation."
- escalate = true, status = "Open"

followup_unresolved:
- Apologize the steps didn't help
- State: "I've now escalated this directly to our technical team."
- escalate = true, status = "Open"

STRICT OUTPUT RULES:
- Return valid JSON only. No markdown, no extra text.
- Use exactly this structure:

{
  "original_complaint": "<input complaint>",
  "draft_reply": "<full reply text>",
  "classification": "<self_fixable | technical_escalation | followup_unresolved>",
  "escalate": <true | false>,
  "status": "<Pending Customer Action | Open>",
  "tone": "<Empathetic | Apologetic | Informative>",
  "suggested_department": "<department derived from context>",
  "priority": "<High | Medium | Low>"
}

- Never ask more than one round of clarifying steps before 
  resolving or escalating.
- Be warm, human, and reassuring.
"""

# ─────────────────────────────────────────────
# Step 3: Create the ADK LlmAgent
# ─────────────────────────────────────────────
support_agent = LlmAgent(
    name="SupportAgent",
    model="gemini-2.5-flash",
    instruction=SYSTEM_PROMPT,
    description=(
        "Drafts professional customer support replies, classifies "
        "issues, and decides whether to escalate to the technical team."
    ),
)

# ─────────────────────────────────────────────
# Step 4: Define the main analysis function
# ─────────────────────────────────────────────
async def draft_support_reply(complaint: str) -> dict:
    """
    Analyzes a customer complaint and drafts a support reply,
    classifying it and deciding whether to escalate.

    Args:
        complaint (str): The customer's complaint text.

    Returns:
        dict: Parsed JSON with draft_reply, classification, 
              escalate, status, tone, suggested_department, priority.
    """
    # ── 4a. Setup in-memory session service ──────────────────────────────────
    # InMemorySessionService stores conversation state temporarily in RAM.
    # Suitable for stateless, single-call agents like this one.
    session_service = InMemorySessionService()

    # Unique identifiers for this app and session
    APP_NAME   = "customer_insight_app"
    USER_ID    = "internal_pipeline"
    SESSION_ID = "support_reply_session"

    # Create a fresh session to hold the conversation context
    # NOTE: create_session is async in ADK 1.3.0 — must be awaited
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # ── 4b. Create the Runner ─────────────────────────────────────────────────
    # Runner orchestrates the agent — sends the message and collects the reply.
    runner = Runner(
        agent=support_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ── 4c. Prepare the user message ──────────────────────────────────────────
    # We pass the complaint directly with a brief instruction.
    user_message_text = f"""
Please draft a professional customer support reply for the following complaint
and return the result in the JSON format specified in your instructions.

CUSTOMER COMPLAINT:
{complaint}
"""

    # Build the ADK-compatible Content object
    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message_text)],
    )

    # ── 4d. Run the agent and collect the final response ──────────────────────
    # runner.run_async() is an async iterator in ADK 1.3.0 — use async for
    final_response_text = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_message,
    ):
        # We only care about the final model response
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()
            break

    # ── 4e. Parse and return the JSON response ────────────────────────────────
    if not final_response_text:
        return {"error": "Agent returned an empty response. Please try again."}

    try:
        # Strip potential markdown code fences if the model wraps JSON in ```json
        cleaned = final_response_text
        if cleaned.startswith("```"):
            # Remove the opening fence (```json or ```)
            cleaned = cleaned.split("\n", 1)[-1]
            # Remove the closing fence
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0].strip()

        # Parse and return as a Python dict
        result = json.loads(cleaned)
        return result

    except json.JSONDecodeError as e:
        # Return raw text alongside the error to help with debugging
        return {
            "error": f"Failed to parse agent response as JSON: {str(e)}",
            "raw_response": final_response_text,
        }


# ─────────────────────────────────────────────
# Step 5: Standalone test entry point
# ─────────────────────────────────────────────
# Run this file directly to test the agent with sample complaints.
if __name__ == "__main__":

    # Test with three different complaint types to verify tone and priority routing
    SAMPLE_COMPLAINTS = [
        "I've been charged twice for my subscription this month and no one from support has responded to my emails in 5 days. This is completely unacceptable!",
        "The mobile app keeps crashing every time I try to open the dashboard on my Android phone. It's been happening since the last update.",
        "I can't figure out how to export my reports to PDF. Is there a tutorial or a way to do this from the settings?",
    ]

    for i, complaint in enumerate(SAMPLE_COMPLAINTS, 1):
        print("=" * 60)
        print(f"  Test {i}: Drafting support reply...")
        print("=" * 60)

        # Call the async function from a synchronous context
        result = asyncio.run(draft_support_reply(complaint))

        # Pretty-print the JSON output
        print(json.dumps(result, indent=2))
        print()
