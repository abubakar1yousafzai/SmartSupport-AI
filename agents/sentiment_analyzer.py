"""
Sentiment Analyzer Agent
========================
This agent analyzes customer reviews and produces a detailed sentiment report
using Google ADK and Gemini 2.5 Flash.

For each review the agent:
  - Assigns a sentiment label  : Positive / Negative / Neutral
  - Assigns a sentiment score  : 0 (most negative) → 100 (most positive)

It then aggregates the per-review scores to:
  - Calculate the average sentiment score per month
  - Identify the best month  (highest average score)
  - Identify the worst month (lowest average score)

Gemini derives all labels and scores directly from the review content.
No sentiment rules, thresholds, or category lists are hardcoded here.
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
# The system prompt defines the agent's role and strict output format.
# Gemini is given full freedom to evaluate sentiment naturally from the text —
# no rules, thresholds, or keyword lists are imposed here.
SYSTEM_PROMPT = """
You are a Customer Sentiment Analysis Expert.

Analyze the provided customer reviews and for each one:
1. Assign a sentiment label: "Positive", "Negative", or "Neutral" — based purely on the review content.
2. Assign a sentiment score (0–100): 0 = extremely negative, 50 = neutral, 100 = extremely positive.
3. Write one short reason explaining the score.

Then aggregate by month:
- Compute the average score per month (rounded to 2 decimal places).
- Identify the best month (highest average) and worst month (lowest average).
- If months are tied, prefer the earlier one chronologically.

STRICT OUTPUT RULES:
- Return valid JSON only. No markdown, no extra text.
- Use exactly this structure:

{
  "reviews": [
    {
      "review_id": <id>,
      "month": "<month label from input>",
      "sentiment": "<Positive | Negative | Neutral>",
      "score": <0–100>,
      "reason": "<one sentence>"
    }
  ],
  "monthly_summary": [
    {
      "month": "<month label>",
      "average_score": <float>,
      "review_count": <int>,
      "positive_count": <int>,
      "negative_count": <int>,
      "neutral_count": <int>
    }
  ],
  "best_month":  { "month": "<label>", "average_score": <float> },
  "worst_month": { "month": "<label>", "average_score": <float> },
  "overall_average_score": <float>,
  "total_reviews_analyzed": <int>,
  "analysis_note": "<one-line summary of overall sentiment trend>"
}

- Do not skip any review.
- Sort monthly_summary chronologically (earliest first).
"""

# ─────────────────────────────────────────────
# Step 3: Create the ADK LlmAgent
# ─────────────────────────────────────────────
# LlmAgent wraps the Gemini model with the system prompt above.
# We use Gemini 2.5 Flash for fast, cost-efficient analysis.
sentiment_agent = LlmAgent(
    name="SentimentAnalyzerAgent",
    model="gemini-2.5-flash",               # Gemini 2.5 Flash model
    instruction=SYSTEM_PROMPT,              # Sets the agent's role and output rules
    description=(
        "Analyzes customer reviews to assign sentiment labels and scores, "
        "compute monthly averages, and identify the best and worst months."
    ),
)

# ─────────────────────────────────────────────
# Step 4: Define the main analysis function
# ─────────────────────────────────────────────
async def analyze_sentiment(reviews_text: str) -> dict:
    """
    Analyzes customer reviews and returns a detailed sentiment report in JSON.

    Each review should include its month so that monthly aggregation is possible.
    The input can be plain text, a numbered list, or a JSON string — Gemini will
    parse it either way.

    Args:
        reviews_text (str): A string containing all customer reviews.
                            Each review should mention the month it was written.

    Returns:
        dict: Parsed JSON dictionary containing:
              - reviews               : per-review sentiment label, score, and reason
              - monthly_summary       : aggregated stats per month
              - best_month            : month with the highest average score
              - worst_month           : month with the lowest average score
              - overall_average_score : average score across all reviews
              - total_reviews_analyzed
              - analysis_note         : one-line overall sentiment summary
              Returns an error dict if the analysis fails.

    Example:
        >>> result = asyncio.run(analyze_sentiment(sample_reviews))
        >>> print(result["best_month"]["month"])
        'January 2024'
    """

    # ── 4a. Setup in-memory session service ──────────────────────────────────
    # InMemorySessionService stores conversation state temporarily in RAM.
    # Suitable for stateless, single-call agents like this one.
    session_service = InMemorySessionService()

    # Unique identifiers for this app and session
    APP_NAME   = "customer_insight_app"
    USER_ID    = "internal_pipeline"
    SESSION_ID = "sentiment_analysis_session"

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
        agent=sentiment_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ── 4c. Prepare the user message ──────────────────────────────────────────
    # We wrap the raw reviews text in a clear instruction for the model.
    user_message_text = f"""
Please analyze the following customer reviews and return the full sentiment
report in the JSON format specified in your instructions.

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

    # Sample customer reviews spanning three months for testing
    SAMPLE_REVIEWS = """
    1. [January 2024] The app is blazing fast and the new UI is gorgeous. Love it!
    2. [January 2024] I was double-charged this month. Support hasn't replied in a week. Furious.
    3. [January 2024] Works fine overall. Nothing special to report.
    4. [February 2024] Major improvement in speed since the last update. Very happy!
    5. [February 2024] The mobile app crashes every time I open it on my Android phone.
    6. [February 2024] Billing issue was resolved quickly. Support team was helpful.
    7. [February 2024] Average experience. The app does what it should, nothing more.
    8. [March 2024] Absolutely terrible. Three billing errors in one month, no response from support.
    9. [March 2024] The new features are fantastic — exactly what I was waiting for!
    10. [March 2024] Performance has degraded noticeably after the latest update.
    11. [March 2024] Customer support resolved my issue in under an hour. Impressed!
    12. [March 2024] App is okay but could use a dark mode. Not complaining, just suggesting.
    """

    print("=" * 60)
    print("  Running Sentiment Analyzer Agent...")
    print("=" * 60)

    # Call the async function from a synchronous context
    result = asyncio.run(analyze_sentiment(SAMPLE_REVIEWS))

    # Pretty-print the full JSON output
    print(json.dumps(result, indent=2))
