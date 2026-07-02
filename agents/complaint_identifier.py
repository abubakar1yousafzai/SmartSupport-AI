"""
Complaint Identifier Agent
==========================
This agent analyzes customer reviews and identifies the top 5 most repeated
complaints using Google ADK and Gemini 2.5 Flash.

Categories are NOT predefined — they emerge naturally from the review data itself,
allowing Gemini to discover whatever themes are actually present in the feedback.
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
# Loads GEMINI_API_KEY from the .env file located at the project root.
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
# The system prompt sets the role and expected behavior of the agent.
# Categories are NOT predefined — Gemini discovers them organically from the data.
SYSTEM_PROMPT = """
You are a Customer Complaint Analysis Expert.

Your task is to analyze customer reviews provided to you as input text.
You must:
1. Identify all complaints mentioned across the reviews.
2. Automatically determine the most appropriate category for each complaint
   based solely on the content of the reviews themselves. Do NOT use any
   predefined or fixed list of categories. Let the categories emerge naturally
   from the data — name them in a way that best reflects what customers are
   actually complaining about.
3. Count how many times each unique complaint appears (or is semantically similar).
4. Return ONLY the TOP 5 most repeated complaints sorted by count (highest first).

STRICT OUTPUT RULES:
- Your response MUST be valid JSON only. No extra text, no markdown, no explanation.
- Use exactly this structure:

{
  "top_complaints": [
    {
      "rank": 1,
      "category": "<a descriptive category name derived from the reviews>",
      "complaint": "<short summary of the complaint>",
      "count": <integer number of occurrences>
    },
    ...
  ],
  "total_reviews_analyzed": <integer>,
  "analysis_note": "<brief one-line summary of overall customer sentiment>"
}

- If a review is positive with no complaint, skip it.
- Always return exactly 5 items in "top_complaints". If fewer than 5 distinct
  complaint types exist, fill remaining ranks with count: 0 and complaint: "No additional complaints found".
"""

# ─────────────────────────────────────────────
# Step 3: Create the ADK LlmAgent
# ─────────────────────────────────────────────
# LlmAgent is the core ADK agent class that wraps an LLM with a system prompt.
# We use Gemini 2.5 Flash for fast, cost-efficient analysis.
complaint_agent = LlmAgent(
    name="ComplaintIdentifierAgent",
    model="gemini-2.5-flash",                  # Gemini 2.5 Flash model
    instruction=SYSTEM_PROMPT,                  # Sets the agent's behavior
    description=(
        "Analyzes customer reviews to identify and rank the top 5 "
        "complaints by category and frequency."
    ),
)

# ─────────────────────────────────────────────
# Step 4: Define the main analysis function
# ─────────────────────────────────────────────
async def identify_complaints(reviews_text: str) -> dict:
    """
    Analyzes customer reviews and returns the top 5 most repeated complaints
    grouped by category in JSON format.

    Args:
        reviews_text (str): A string containing all customer reviews.
                            Can be plain text or a JSON string of review objects.

    Returns:
        dict: Parsed JSON dictionary containing top_complaints, 
              total_reviews_analyzed, and analysis_note.
              Returns an error dict if analysis fails.
    
    Example:
        >>> result = asyncio.run(identify_complaints("App crashes often. Billing is wrong."))
        >>> print(result["top_complaints"][0]["category"])
        'Performance'
    """

    # ── 5a. Setup in-memory session service ──────────────────────────────────
    # InMemorySessionService stores conversation state temporarily in RAM.
    # This is suitable for stateless, single-call agents like this one.
    session_service = InMemorySessionService()

    # Define unique IDs for the app and user session
    APP_NAME    = "customer_insight_app"
    USER_ID     = "internal_pipeline"
    SESSION_ID  = "complaint_analysis_session"

    # Create a new session to hold the conversation context
    # NOTE: create_session is async in ADK 1.3.0 — must be awaited
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # ── 5b. Create the Runner ─────────────────────────────────────────────────
    # Runner orchestrates the agent — it sends messages and receives responses.
    runner = Runner(
        agent=complaint_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ── 5c. Prepare the user message ──────────────────────────────────────────
    # We wrap the reviews text inside a clear instruction for the model.
    user_message_text = f"""
Please analyze the following customer reviews and return the top 5 complaints
in the JSON format specified in your instructions.

CUSTOMER REVIEWS:
{reviews_text}
"""

    # Build the ADK-compatible message object
    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message_text)],
    )

    # ── 5d. Run the agent and collect the final response ──────────────────────
    # runner.run_async() is the async iterator in ADK 1.3.0 — must use async for
    final_response_text = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=user_message,
    ):
        # Check if this event contains the final model response text
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()
            break

    # ── 5e. Parse and return the JSON response ────────────────────────────────
    if not final_response_text:
        return {"error": "Agent returned an empty response. Please try again."}

    try:
        # Strip potential markdown code fences if model wraps JSON in ```json
        cleaned = final_response_text
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            cleaned = cleaned.split("\n", 1)[-1]
            # Remove closing fence
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0].strip()

        # Parse and return as Python dict
        result = json.loads(cleaned)
        return result

    except json.JSONDecodeError as e:
        # Return raw text alongside the error for debugging
        return {
            "error": f"Failed to parse agent response as JSON: {str(e)}",
            "raw_response": final_response_text,
        }


# ─────────────────────────────────────────────
# Step 5: Standalone test entry point
# ─────────────────────────────────────────────
# Run this file directly to test the agent with sample reviews.
if __name__ == "__main__":

    # Sample customer reviews for testing
    SAMPLE_REVIEWS = """
    1. The app crashes every time I try to view my bill. Very frustrating!
    2. I was charged twice for the same month. Billing department didn't help.
    3. The mobile app is so slow, it takes forever to load.
    4. Customer support took 5 days to respond to my ticket. Unacceptable.
    5. I wish there was a dark mode feature in the app.
    6. My internet speed is way below what I'm paying for.
    7. The app crashed again during checkout. Please fix this bug.
    8. I've been overcharged for 3 months in a row. No resolution yet.
    9. Support agents are rude and unhelpful.
    10. The mobile app keeps logging me out randomly.
    11. Performance is terrible during peak hours.
    12. I can't find a bulk download feature anywhere — please add it.
    13. Billing errors keep happening and no one fixes it.
    14. App crashes on Android devices specifically.
    15. Customer support never picks up the phone.
    """

    print("=" * 60)
    print("  Running Complaint Identifier Agent...")
    print("=" * 60)

    # Call the async function using asyncio.run() from sync context
    result = asyncio.run(identify_complaints(SAMPLE_REVIEWS))

    # Pretty-print the JSON output
    print(json.dumps(result, indent=2))
