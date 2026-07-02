"""
Orchestrator Agent
==================
This is the main coordinator for the Customer Insight Agent pipeline.
It calls all sub-agents in sequence, aggregates their results, and produces
a final markdown report + dashboard JSON saved via MCP tools.

Pipeline order:
  Step 1 : Fetch active (Open/In Progress) issues via SQLite MCP tool (fetch_active_issues)
  Step 2 : Run complaint_identifier + sentiment_analyzer in PARALLEL
  Step 3 : Run trend_analyzer (uses issues text)
  Step 4 : Run insight_reporter (uses all Step 2 + Step 3 results)
  Step 5 : Build dashboard JSON and save via MCP (save_dashboard_data)
  Step 6 : Build markdown report and save via MCP (save_report)
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Step 0a: Resolve project root so imports work
# ─────────────────────────────────────────────
# When running this file directly, Python needs to find the `agents` package.
# We add the project root (one level above this file) to sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────
# Step 0b: Load environment variables from .env
# ─────────────────────────────────────────────
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Please set it in your .env file."
    )
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ─────────────────────────────────────────────
# Step 0c: Import sub-agent functions
# ─────────────────────────────────────────────
from agents.complaint_identifier import identify_complaints
from agents.sentiment_analyzer   import analyze_sentiment
from agents.trend_analyzer       import analyze_trends
from agents.insight_reporter     import generate_insights

# ─────────────────────────────────────────────
# Step 0d: Import MCP tool functions directly
# ─────────────────────────────────────────────
# We call the MCP tool functions directly (not via MCP transport) since the
# orchestrator and tools live in the same Python process.
from mcp_tools.data_tool import fetch_active_issues, save_dashboard_data


# ══════════════════════════════════════════════════════════════
# Helper: Format issues list → plain text string for agents
# ══════════════════════════════════════════════════════════════
def issues_to_text(issues: list) -> str:
    """
    Converts the structured issues list from SQLite into a plain numbered
    text string that all sub-agents can process uniformly.

    Each line format:
      "{i}. [created_at month] complaint: {complaint_text} category: {category} sentiment: {sentiment}"

    Args:
        issues (list): List of issue dicts from SQLite containing issue_id,
                       complaint_text, category, sentiment, created_at, etc.

    Returns:
        str: Formatted plain text string of all issues.
    """
    lines = []
    for i, issue in enumerate(issues, 1):
        # Format the timestamp into a month-year representation (e.g. "June 2026")
        created_at_str = issue.get("created_at", "")
        month_label = "Unknown Month"
        if created_at_str:
            try:
                # Expecting format "YYYY-MM-DD HH:MM:SS"
                dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                month_label = dt.strftime("%B %Y")
            except ValueError:
                # If format differs, fallback to string representation or parts of it
                month_label = created_at_str

        complaint_text = issue.get("complaint_text", "").replace("\n", " ").strip()
        category = issue.get("category", "General")
        sentiment = issue.get("sentiment", "Neutral")
        
        lines.append(
            f'{i}. [{month_label}] complaint: "{complaint_text}" '
            f'category: {category} sentiment: {sentiment}'
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Helper: Build the final markdown report (excluding support replies)
# ══════════════════════════════════════════════════════════════
def build_markdown_report(
    complaint_data: dict,
    sentiment_data: dict,
    trend_data:     dict,
    insight_data:   dict,
) -> str:
    """
    Assembles all sub-agent outputs into a single structured markdown report.

    Args:
        complaint_data   : Output from identify_complaints()
        sentiment_data   : Output from analyze_sentiment()
        trend_data       : Output from analyze_trends()
        insight_data     : Output from generate_insights()

    Returns:
        str: Full markdown report as a string.
    """
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("# Customer Insight Report")
    lines.append(f"\n> **Generated:** {now}\n")

    # ── Section 1: Executive Summary ──────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Executive Summary\n")
    summary = insight_data.get("executive_summary", "_No summary available._")
    lines.append(summary)

    top_action = insight_data.get("top_priority_action", {})
    if top_action:
        lines.append(f"\n**🚨 Top Priority Action**")
        lines.append(f"- **Department:** {top_action.get('department', 'N/A')}")
        lines.append(f"- **Action:** {top_action.get('action', 'N/A')}")
        lines.append(f"- **Reason:** {top_action.get('reason', 'N/A')}")

    # ── Section 2: Sentiment Analysis ─────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Sentiment Analysis\n")

    overall = sentiment_data.get("overall_average_score", "N/A")
    best    = sentiment_data.get("best_month",  {})
    worst   = sentiment_data.get("worst_month", {})
    total   = sentiment_data.get("total_reviews_analyzed", "N/A")

    lines.append(f"- **Total Reviews/Issues Analyzed:** {total}")
    lines.append(f"- **Overall Average Sentiment Score:** {overall} / 100")
    lines.append(f"- **Best Month:** {best.get('month', 'N/A')} (Score: {best.get('average_score', 'N/A')})")
    lines.append(f"- **Worst Month:** {worst.get('month', 'N/A')} (Score: {worst.get('average_score', 'N/A')})")
    lines.append(f"\n*{sentiment_data.get('analysis_note', '')}*")

    monthly = sentiment_data.get("monthly_summary", [])
    if monthly:
        lines.append("\n### Monthly Breakdown\n")
        lines.append("| Month | Avg Score | Issues | Positive | Negative | Neutral |")
        lines.append("|-------|-----------|---------|----------|----------|---------|")
        for m in monthly:
            lines.append(
                f"| {m.get('month')} "
                f"| {m.get('average_score')} "
                f"| {m.get('review_count')} "
                f"| {m.get('positive_count')} "
                f"| {m.get('negative_count')} "
                f"| {m.get('neutral_count')} |"
            )

    # ── Section 3: Top Repeated Complaints ────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Top Repeated Complaints\n")

    complaints = complaint_data.get("top_complaints", [])
    if complaints:
        lines.append("| Rank | Category | Complaint | Count |")
        lines.append("|------|----------|-----------|-------|")
        for c in complaints:
            lines.append(
                f"| {c.get('rank')} "
                f"| {c.get('category')} "
                f"| {c.get('complaint')} "
                f"| {c.get('count')} |"
            )
    else:
        lines.append("_No complaints identified._")

    lines.append(f"\n*{complaint_data.get('analysis_note', '')}*")

    # ── Section 4: Monthly Trends ─────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Monthly Trends\n")

    volumes = trend_data.get("monthly_volumes", [])
    if volumes:
        lines.append("### Complaint Volume\n")
        lines.append("| Month | Complaints | MoM Growth |")
        lines.append("|-------|------------|------------|")
        for v in volumes:
            growth = v.get("growth_rate")
            growth_str = f"{growth:+.2f}%" if growth is not None else "—"
            lines.append(f"| {v.get('month')} | {v.get('complaint_count')} | {growth_str} |")

    recurring = trend_data.get("recurring_issues", [])
    if recurring:
        lines.append("\n### Recurring Issues (2+ Months)\n")
        for issue in recurring:
            months = ", ".join(issue.get("months_present", []))
            lines.append(f"- **{issue.get('issue')}** — {issue.get('total_mentions')} mentions across: {months}")

    flagged = trend_data.get("flagged_issues", [])
    if flagged:
        lines.append("\n### ⚠️ Flagged Issues (Getting Worse)\n")
        for f in flagged:
            lines.append(f"- **{f.get('issue')}:** {f.get('reason')}")

    lines.append(f"\n*{trend_data.get('trend_summary', '')}*")

    # ── Section 5: Business Recommendations ───────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Business Recommendations\n")

    recommendations = insight_data.get("recommendations", [])
    priority_icons  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

    for rec in recommendations:
        icon = priority_icons.get(rec.get("priority", ""), "⚪")
        lines.append(f"### {icon} Rank {rec.get('rank')} — {rec.get('priority')} Priority | {rec.get('department')}\n")
        lines.append(f"**Issue:** {rec.get('issue')}")
        lines.append(f"\n**Action:** {rec.get('action')}")
        lines.append(f"\n**Expected Impact:** {rec.get('expected_impact')}\n")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append("\n---")
    lines.append(f"\n*Report generated automatically by the Customer Insight Agent pipeline.*")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Main orchestration function
# ══════════════════════════════════════════════════════════════
async def run_pipeline() -> None:
    """
    Executes the full Customer Insight Agent pipeline end-to-end:
      Step 1 → Fetch active (Open/In Progress) issues from SQLite MCP tool
      Step 2 → Run complaint + sentiment agents in parallel
      Step 3 → Run trend agent
      Step 4 → Run insight reporter
      Step 5 → Build and save dashboard JSON via MCP
      Step 6 → Build and save markdown report via MCP
    """

    print("\n" + "═" * 60)
    print("   Customer Insight Agent — Starting Pipeline (SQLite Active Issues)")
    print("═" * 60)

    # ─────────────────────────────────────────────────────────
    # Step 1: Fetch active issues via SQLite tool
    # ─────────────────────────────────────────────────────────
    print("\n[Step 1] Fetching active issues via SQLite MCP tool...")

    raw_json = fetch_active_issues()
    
    # Parse fetch result
    try:
        issues_list = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse active issues JSON: {str(e)}")
        return

    # Check for error message inside returned JSON
    if isinstance(issues_list, dict) and "error" in issues_list:
        print(f"  ✗ Failed to fetch active issues: {issues_list['error']}")
        return

    # If no active issues, print message and exit
    if not issues_list:
        print("  ✓ No active issues found. Pipeline stopped.")
        return

    print(f"  ✓ Fetched {len(issues_list)} active issues.")

    # ─────────────────────────────────────────────────────────
    # Step 2: Convert issues to text format for sub-agents
    # ─────────────────────────────────────────────────────────
    issues_text = issues_to_text(issues_list)

    # ─────────────────────────────────────────────────────────
    # Step 3: Run complaint_identifier + sentiment_analyzer in PARALLEL
    # ─────────────────────────────────────────────────────────
    print("\n[Step 3] Running complaint identifier & sentiment analyzer in parallel...")

    complaint_task = identify_complaints(issues_text)
    sentiment_task = analyze_sentiment(issues_text)

    # asyncio.gather runs both coroutines concurrently
    complaint_data, sentiment_data = await asyncio.gather(complaint_task, sentiment_task)

    if "error" in complaint_data:
        print(f"  ✗ Complaint identifier failed: {complaint_data['error']}")
    else:
        top = len(complaint_data.get("top_complaints", []))
        print(f"  ✓ Complaint identifier done — {top} top complaints identified.")

    if "error" in sentiment_data:
        print(f"  ✗ Sentiment analyzer failed: {sentiment_data['error']}")
    else:
        score = sentiment_data.get("overall_average_score", "N/A")
        print(f"  ✓ Sentiment analyzer done — Overall score: {score}/100.")

    # ─────────────────────────────────────────────────────────
    # Step 4: Run trend_analyzer
    # ─────────────────────────────────────────────────────────
    print("\n[Step 4] Running trend analyzer...")

    trend_data = await analyze_trends(issues_text)

    if "error" in trend_data:
        print(f"  ✗ Trend analyzer failed: {trend_data['error']}")
    else:
        months  = trend_data.get("months_analyzed", "N/A")
        flagged = len(trend_data.get("flagged_issues", []))
        print(f"  ✓ Trend analyzer done — {months} months analyzed, {flagged} flagged issues.")

    # ─────────────────────────────────────────────────────────
    # Step 5: Run insight_reporter (uses all previous results)
    # ─────────────────────────────────────────────────────────
    print("\n[Step 5] Running insight reporter...")

    insight_data = await generate_insights(
        complaint_data=complaint_data,
        sentiment_data=sentiment_data,
        trend_data=trend_data,
    )

    if "error" in insight_data:
        print(f"  ✗ Insight reporter failed: {insight_data['error']}")
    else:
        recs = len(insight_data.get("recommendations", []))
        print(f"  ✓ Insight reporter done — {recs} recommendations generated.")

    # ─────────────────────────────────────────────────────────
    # Step 6: Build dashboard JSON and save via MCP
    # ─────────────────────────────────────────────────────────
    print("\n[Step 6] Building and saving dashboard data...")

    dashboard_data = {
        "generated_at":      datetime.now().isoformat(),
        "total_reviews":     len(issues_list),
        "complaint_data":    complaint_data,
        "sentiment_data":    sentiment_data,
        "trend_data":        trend_data,
        "insight_data":      insight_data,
    }

    dash_status = save_dashboard_data(json.dumps(dashboard_data))
    print(f"  {dash_status}")

    # ─────────────────────────────────────────────────────────
    # Step 7: Build markdown report and save directly
    # ─────────────────────────────────────────────────────────
    print("\n[Step 7] Building and saving markdown report...")

    report_md = build_markdown_report(
        complaint_data=complaint_data,
        sentiment_data=sentiment_data,
        trend_data=trend_data,
        insight_data=insight_data,
    )

    try:
        report_file = os.path.join(PROJECT_ROOT, "output", "report.md")
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"  Success: Business report saved to {report_file}")
    except Exception as e:
        print(f"  Failure: Failed to save report: {str(e)}")

    # ─────────────────────────────────────────────────────────
    # Done
    # ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("   Pipeline Complete ✓")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(run_pipeline())
