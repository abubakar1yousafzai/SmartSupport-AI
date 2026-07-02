"""
Insight Reporter Agent
======================
This agent receives the combined output of the three analysis agents
(complaint identifier, sentiment analyzer, trend analyzer) and generates
actionable business recommendations using Google ADK and Gemini 2.5 Flash.

It produces:
  - 3–5 prioritized recommendations (High / Medium / Low impact)
  - The department responsible for each action
  - An executive summary (2–3 lines)
  - The single most urgent top-priority action

Gemini derives all recommendations naturally from the input data.
No rules, templates, or department lists are hardcoded here.
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
# Instructs Gemini to synthesize complaint, sentiment, and trend data
# into clear, prioritized, department-specific business recommendations.
SYSTEM_PROMPT = """
You are a Senior Customer Experience Strategist.

You will receive a combined JSON input with three analyses:
  1. Complaint data   — top complaints by category and frequency
  2. Sentiment data   — monthly sentiment scores, best/worst months
  3. Trend data       — monthly complaint volumes, recurring and worsening issues

Synthesize all three and produce 3–5 actionable business recommendations. Each must:
- Be grounded in specific evidence from the data (no generic advice).
- Have a priority: "High", "Medium", or "Low" based on impact and urgency.
- Name the department best suited to act (derive from context — do not use a fixed list).
- Include a concrete, specific action.

Also write a 2–3 sentence executive summary and identify the single most urgent action.

STRICT OUTPUT RULES:
- Return valid JSON only. No markdown, no extra text.
- Use exactly this structure:

{
  "recommendations": [
    {
      "rank": <int>,
      "priority": "<High | Medium | Low>",
      "department": "<department name>",
      "issue": "<problem this addresses>",
      "action": "<specific action to take>",
      "expected_impact": "<one sentence>"
    }
  ],
  "executive_summary": "<2–3 sentences on the overall customer situation>",
  "top_priority_action": {
    "department": "<department>",
    "action": "<most urgent action>",
    "reason": "<one sentence>"
  }
}

- Sort recommendations High → Medium → Low.
- Generate exactly 3–5 recommendations.
"""

# ─────────────────────────────────────────────
# Step 3: Create the ADK LlmAgent
# ─────────────────────────────────────────────
# LlmAgent wraps the Gemini model with the system prompt above.
# We use Gemini 2.5 Flash for fast, cost-efficient analysis.
insight_agent = LlmAgent(
    name="InsightReporterAgent",
    model="gemini-2.5-flash",               # Gemini 2.5 Flash model
    instruction=SYSTEM_PROMPT,              # Sets the agent's role and output rules
    description=(
        "Synthesizes complaint, sentiment, and trend data into prioritized "
        "business recommendations with an executive summary."
    ),
)

# ─────────────────────────────────────────────
# Step 4: Define the main report generation function
# ─────────────────────────────────────────────
async def generate_insights(
    complaint_data: dict,
    sentiment_data: dict,
    trend_data: dict,
) -> dict:
    """
    Generates actionable business recommendations from the combined analysis data.

    Args:
        complaint_data (dict): Output from the Complaint Identifier Agent.
                               Contains top_complaints, total_reviews_analyzed, etc.
        sentiment_data (dict): Output from the Sentiment Analyzer Agent.
                               Contains reviews, monthly_summary, best/worst month, etc.
        trend_data     (dict): Output from the Trend Analyzer Agent.
                               Contains monthly_volumes, recurring_issues, flagged_issues, etc.

    Returns:
        dict: Parsed JSON dictionary containing:
              - recommendations    : 3–5 prioritized actions with department and impact
              - executive_summary  : 2–3 sentence overview of the customer situation
              - top_priority_action: the single most urgent action to take
              Returns an error dict if the analysis fails.

    Example:
        >>> result = asyncio.run(generate_insights(complaints, sentiment, trends))
        >>> print(result["top_priority_action"]["action"])
        'Deploy a hotfix for the Android crash within 48 hours.'
    """

    # ── 4a. Setup in-memory session service ──────────────────────────────────
    # InMemorySessionService stores conversation state temporarily in RAM.
    # Suitable for stateless, single-call agents like this one.
    session_service = InMemorySessionService()

    # Unique identifiers for this app and session
    APP_NAME   = "customer_insight_app"
    USER_ID    = "internal_pipeline"
    SESSION_ID = "insight_report_session"

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
        agent=insight_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ── 4c. Prepare the user message ──────────────────────────────────────────
    # We combine all three analysis outputs into a single structured prompt.
    combined_input = {
        "complaint_analysis": complaint_data,
        "sentiment_analysis": sentiment_data,
        "trend_analysis":     trend_data,
    }

    user_message_text = f"""
Please analyze the following combined customer analysis data and generate
actionable business recommendations in the JSON format specified in your instructions.

COMBINED ANALYSIS DATA:
{json.dumps(combined_input, indent=2)}
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
# Run this file directly to test the agent with realistic sample data.
if __name__ == "__main__":

    # --- Sample complaint data (as produced by complaint_identifier.py) ---
    SAMPLE_COMPLAINT_DATA = {
        "top_complaints": [
            {"rank": 1, "category": "App Stability",    "complaint": "App crashes frequently on mobile devices",       "count": 6},
            {"rank": 2, "category": "Billing Errors",   "complaint": "Customers overcharged or billed incorrectly",    "count": 5},
            {"rank": 3, "category": "Support Delays",   "complaint": "Support response times are too slow",            "count": 4},
            {"rank": 4, "category": "Performance",      "complaint": "App is slow, especially during peak hours",      "count": 3},
            {"rank": 5, "category": "Broken Features",  "complaint": "Dashboard and export features stopped working",  "count": 2},
        ],
        "total_reviews_analyzed": 15,
        "analysis_note": "App stability and billing are the dominant pain points.",
    }

    # --- Sample sentiment data (as produced by sentiment_analyzer.py) ---
    SAMPLE_SENTIMENT_DATA = {
        "monthly_summary": [
            {"month": "January 2024",  "average_score": 52.33, "review_count": 4, "positive_count": 1, "negative_count": 2, "neutral_count": 1},
            {"month": "February 2024", "average_score": 48.75, "review_count": 5, "positive_count": 1, "negative_count": 3, "neutral_count": 1},
            {"month": "March 2024",    "average_score": 38.17, "review_count": 6, "positive_count": 1, "negative_count": 4, "neutral_count": 1},
        ],
        "best_month":  {"month": "January 2024",  "average_score": 52.33},
        "worst_month": {"month": "March 2024",    "average_score": 38.17},
        "overall_average_score": 46.08,
        "total_reviews_analyzed": 15,
        "analysis_note": "Sentiment is declining steadily month over month.",
    }

    # --- Sample trend data (as produced by trend_analyzer.py) ---
    SAMPLE_TREND_DATA = {
        "monthly_volumes": [
            {"month": "January 2024",  "complaint_count": 4, "growth_rate": None},
            {"month": "February 2024", "complaint_count": 5, "growth_rate": 25.0},
            {"month": "March 2024",    "complaint_count": 6, "growth_rate": 20.0},
        ],
        "recurring_issues": [
            {"issue": "App crashes",    "months_present": ["January 2024", "February 2024", "March 2024"], "total_mentions": 3},
            {"issue": "Billing errors", "months_present": ["January 2024", "February 2024", "March 2024"], "total_mentions": 3},
            {"issue": "Slow support",   "months_present": ["January 2024", "February 2024", "March 2024"], "total_mentions": 3},
        ],
        "flagged_issues": [
            {"issue": "App crashes",    "reason": "Mentions increased from 1 to 2 to 3 across months.", "trend": "worsening"},
            {"issue": "Billing errors", "reason": "Mentions increased each month with no resolution.",  "trend": "worsening"},
        ],
        "total_complaints_analyzed": 15,
        "months_analyzed": 3,
        "trend_summary": "Complaint volume is rising 20–25% month over month with no signs of improvement.",
    }

    print("=" * 60)
    print("  Running Insight Reporter Agent...")
    print("=" * 60)

    # Call the async function from a synchronous context
    result = asyncio.run(
        generate_insights(
            complaint_data=SAMPLE_COMPLAINT_DATA,
            sentiment_data=SAMPLE_SENTIMENT_DATA,
            trend_data=SAMPLE_TREND_DATA,
        )
    )

    # Pretty-print the full JSON output
    print(json.dumps(result, indent=2))
