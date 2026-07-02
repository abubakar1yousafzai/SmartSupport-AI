"""
Trend Analyzer Agent
====================
This agent analyzes customer reviews to surface complaint trends over time
using Google ADK and Gemini 2.5 Flash.

It identifies:
  - Monthly complaint volume (how many complaints per month)
  - Month-over-month growth rate (% change in complaint volume)
  - Recurring issues (same issue appearing in 2+ months)
  - Flagged issues (issues whose frequency is increasing over time)

Gemini derives all issue labels and counts directly from the review content.
No categories, keywords, or rules are hardcoded here.
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
# Instructs Gemini to identify trends naturally from review content.
# No category lists or complaint types are imposed — Gemini decides.
SYSTEM_PROMPT = """
You are a Customer Trend Analysis Expert.

You will receive a list of customer reviews. Each review includes a month label
(e.g., "January 2024", "2024-01") and a complaint or feedback text.

Your task is to analyze complaint trends over time. Specifically:
Analyze the provided customer reviews (each tagged with a month) and:
1. Count complaints per month (skip positive/neutral reviews with no complaint).
2. Calculate month-over-month growth rate (% change). Round to 2 decimal places.
   Formula: ((current - previous) / previous) * 100. Set null for the first month.
3. Identify recurring issues: themes appearing as complaints in 2+ distinct months.
   Name them naturally from the review content — no fixed list.
4. Flag worsening issues: recurring issues whose mention count increases over time.

STRICT OUTPUT RULES:
- Return valid JSON only. No markdown, no extra text.
- Use exactly this structure:

{
  "monthly_volumes": [
    { "month": "<label>", "complaint_count": <int>, "growth_rate": <float | null> }
  ],
  "month_over_month_growth": [
    { "from_month": "<label>", "to_month": "<label>", "growth_rate_percent": <float> }
  ],
  "recurring_issues": [
    {
      "issue": "<issue name>",
      "months_present": ["<month1>", "<month2>"],
      "total_mentions": <int>,
      "monthly_breakdown": [ { "month": "<label>", "mentions": <int> } ]
    }
  ],
  "flagged_issues": [
    { "issue": "<issue name>", "reason": "<one sentence>", "trend": "worsening" }
  ],
  "total_complaints_analyzed": <int>,
  "months_analyzed": <int>,
  "trend_summary": "<one sentence describing the overall complaint trend>"
}

- Sort monthly_volumes and month_over_month_growth chronologically.
- Sort recurring_issues by total_mentions descending.
- Return empty arrays if no recurring or flagged issues exist.
"""

# ─────────────────────────────────────────────
# Step 3: Create the ADK LlmAgent
# ─────────────────────────────────────────────
# LlmAgent wraps the Gemini model with the system prompt above.
# We use Gemini 2.5 Flash for fast, cost-efficient analysis.
trend_agent = LlmAgent(
    name="TrendAnalyzerAgent",
    model="gemini-2.5-flash",               # Gemini 2.5 Flash model
    instruction=SYSTEM_PROMPT,              # Sets the agent's role and output rules
    description=(
        "Analyzes customer reviews to surface complaint trends: monthly volumes, "
        "month-over-month growth, recurring issues, and flagged worsening issues."
    ),
)

# ─────────────────────────────────────────────
# Step 4: Define the main analysis function
# ─────────────────────────────────────────────
async def analyze_trends(reviews_text: str) -> dict:
    """
    Analyzes customer reviews for complaint trends over time.

    Each review in the input should include the month it was written so that
    monthly grouping and trend detection can be performed.

    Args:
        reviews_text (str): A string containing all customer reviews.
                            Each review should mention the month it was written.

    Returns:
        dict: Parsed JSON dictionary containing:
              - monthly_volumes          : complaint count per month with growth rate
              - month_over_month_growth  : % change between consecutive months
              - recurring_issues         : issues appearing in 2+ months
              - flagged_issues           : recurring issues that are getting worse
              - total_complaints_analyzed
              - months_analyzed
              - trend_summary            : one-line overall trend description
              Returns an error dict if analysis fails.

    Example:
        >>> result = asyncio.run(analyze_trends(sample_reviews))
        >>> print(result["trend_summary"])
        'Complaint volume is rising month over month, driven by billing and app issues.'
    """

    # ── 4a. Setup in-memory session service ──────────────────────────────────
    # InMemorySessionService stores conversation state temporarily in RAM.
    # Suitable for stateless, single-call agents like this one.
    session_service = InMemorySessionService()

    # Unique identifiers for this app and session
    APP_NAME   = "customer_insight_app"
    USER_ID    = "internal_pipeline"
    SESSION_ID = "trend_analysis_session"

    # Create a fresh session to hold the conversation context
    # NOTE: create_session is async in ADK 1.3.0 — must be awaited
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # ── 4b. Create the Runner ─────────────────────────────────────────────────
    # Runner orchestrates the agent — sends the user message and collects the reply.
    runner = Runner(
        agent=trend_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ── 4c. Prepare the user message ──────────────────────────────────────────
    # We wrap the raw reviews text in a clear instruction for the model.
    user_message_text = f"""
Please analyze the following customer reviews for complaint trends and return
the full trend report in the JSON format specified in your instructions.

CUSTOMER REVIEWS:
{reviews_text}
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
# Run this file directly to test the agent with sample reviews.
if __name__ == "__main__":

    # Sample reviews spanning three months — includes recurring and worsening issues
    SAMPLE_REVIEWS = """
    1. [January 2024] The app crashes constantly. I've lost work three times already.
    2. [January 2024] I was charged twice for my subscription. Billing is a mess.
    3. [January 2024] Customer support took 4 days to respond. Very disappointed.
    4. [January 2024] The app is really slow during peak hours.

    5. [February 2024] App keeps crashing on my iPhone. This is getting ridiculous.
    6. [February 2024] Another billing error — overcharged again this month!
    7. [February 2024] Support response time is even worse now, waited 6 days.
    8. [February 2024] Slow performance again. Nothing has improved.
    9. [February 2024] New update broke the dashboard. Can't access my reports.

    10. [March 2024] Three crashes today alone. The app is unusable.
    11. [March 2024] Billed the wrong amount for the third month in a row. Unacceptable.
    12. [March 2024] Still waiting for support to reply after 8 days. No one cares.
    13. [March 2024] Performance is getting worse every single week.
    14. [March 2024] Dashboard is still broken after the February update.
    15. [March 2024] Export feature stopped working after the latest patch.
    """

    print("=" * 60)
    print("  Running Trend Analyzer Agent...")
    print("=" * 60)

    # Call the async function from a synchronous context
    result = asyncio.run(analyze_trends(SAMPLE_REVIEWS))

    # Pretty-print the full JSON output
    print(json.dumps(result, indent=2))
