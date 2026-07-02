"""
MCP Server for Customer Insight Agent.
Exposes tools to read customer reviews, write analytics reports,
dashboard data, and manage customer issues via SQLite.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Customer Insight Data Tool")

# Setup absolute file paths relative to this file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_DATA_FILE = os.path.join(BASE_DIR, "data", "dashboard_data.json")
ISSUES_DB = os.path.join(BASE_DIR, "data", "issues.db")

# Note: Data now comes from SQLite via customer portal chat

@mcp.tool()
def save_dashboard_data(data_json: str) -> str:
    """
    Saves the aggregated analysis results in JSON format for the dashboard to render.
    
    Args:
        data_json (str): JSON formatted string of insights, trends, and statistics.
        
    Returns:
        str: A status message indicating success or failure.
    """
    try:
        os.makedirs(os.path.dirname(DASHBOARD_DATA_FILE), exist_ok=True)
        
        # Verify it's valid JSON
        parsed_data = json.loads(data_json)
        
        with open(DASHBOARD_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2)
            
        return f"Success: Dashboard data saved to {DASHBOARD_DATA_FILE}"
    except json.JSONDecodeError:
        return "Failure: Invalid JSON data provided."
    except Exception as e:
        return f"Failure: Failed to save dashboard data: {str(e)}"

# ---------------------------------------------------------------------------
# Issue Tracking Tools (SQLite-backed)
# ---------------------------------------------------------------------------

def _get_issues_connection() -> sqlite3.Connection:
    """
    Opens a connection to the issues SQLite database and ensures the
    'issues' table exists with the correct schema.
    The database file is created automatically if it doesn't exist.
    """
    # Ensure the data/ directory exists before connecting
    os.makedirs(os.path.dirname(ISSUES_DB), exist_ok=True)

    conn = sqlite3.connect(ISSUES_DB)
    conn.row_factory = sqlite3.Row  # Allows column access by name

    # Create the issues table if it has not been created yet
    conn.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            issue_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_text TEXT    NOT NULL,
            agent_reply    TEXT    NOT NULL,
            category       TEXT    NOT NULL,
            sentiment      TEXT    NOT NULL,
            priority       TEXT    NOT NULL,
            status         TEXT    NOT NULL DEFAULT 'Open',
            created_at     TEXT    NOT NULL,
            resolved_at    TEXT    DEFAULT NULL
        )
    """)
    conn.commit()
    return conn


@mcp.tool()
def save_issue(
    complaint_text: str,
    agent_reply: str,
    category: str,
    sentiment: str,
    priority: str,
    status: str = "Open",
) -> str:
    """
    Saves a new customer issue to the SQLite issues database.

    Args:
        complaint_text (str): The original complaint text from the customer.
        agent_reply    (str): The AI agent's reply / suggested resolution.
        category       (str): Issue category (e.g. 'Billing', 'Delivery').
        sentiment      (str): Detected sentiment (e.g. 'Negative', 'Neutral').
        priority       (str): Priority level (e.g. 'High', 'Medium', 'Low').
        status         (str): Initial issue status (e.g. 'Open', 'Pending Customer Action').

    Returns:
        str: JSON string with the newly created issue_id, or an error message.
    """
    try:
        conn = _get_issues_connection()

        # Capture the current UTC timestamp for created_at
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        cursor = conn.execute(
            """
            INSERT INTO issues
                (complaint_text, agent_reply, category, sentiment, priority,
                 status, created_at, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (complaint_text, agent_reply, category, sentiment, priority, status, created_at),
        )
        conn.commit()

        # Retrieve the auto-assigned primary key
        issue_id = cursor.lastrowid
        conn.close()

        return json.dumps({"success": True, "issue_id": issue_id})
    except Exception as e:
        return json.dumps({"error": f"Failed to save issue: {str(e)}"})


@mcp.tool()
def fetch_all_issues() -> str:
    """
    Retrieves all customer issues from the SQLite database, ordered newest first.

    Returns:
        str: JSON string containing a list of all issue records, or an error message.
    """
    try:
        conn = _get_issues_connection()

        # Fetch every row, most recently created first
        cursor = conn.execute(
            "SELECT * FROM issues ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        # Convert sqlite3.Row objects to plain dicts for JSON serialisation
        issues = [dict(row) for row in rows]
        return json.dumps(issues, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch issues: {str(e)}"})


@mcp.tool()
def update_issue_status(issue_id: int, status: str) -> str:
    """
    Updates the status of an existing issue.
    Automatically sets resolved_at when the status is changed to 'Resolved'.

    Args:
        issue_id (int): The ID of the issue to update.
        status   (str): New status — must be one of: 'Open', 'In Progress', 'Resolved'.

    Returns:
        str: A message indicating success or failure.
    """
    # Validate the provided status value
    allowed_statuses = {"Open", "In Progress", "Resolved"}
    if status not in allowed_statuses:
        return (
            f"Failure: Invalid status '{status}'. "
            f"Allowed values: {', '.join(sorted(allowed_statuses))}."
        )

    try:
        conn = _get_issues_connection()

        # If resolving, record the resolution timestamp; otherwise clear it
        resolved_at = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if status == "Resolved"
            else None
        )

        cursor = conn.execute(
            """
            UPDATE issues
            SET    status      = ?,
                   resolved_at = ?
            WHERE  issue_id    = ?
            """,
            (status, resolved_at, issue_id),
        )
        conn.commit()
        conn.close()

        # rowcount == 0 means no row matched the given issue_id
        if cursor.rowcount == 0:
            return f"Failure: No issue found with issue_id = {issue_id}."

        return f"Success: Issue {issue_id} status updated to '{status}'."
    except Exception as e:
        return f"Failure: Failed to update issue status: {str(e)}"


@mcp.tool()
def get_repeat_issues() -> str:
    """
    Identifies categories that have 2 or more issues, indicating recurring problems.

    Groups issues by category and returns only those categories whose issue
    count is >= 2, along with the count and the list of associated issue IDs.

    Returns:
        str: JSON string containing a list of repeat-category summaries,
             or an error message.
    """
    try:
        conn = _get_issues_connection()

        # Aggregate by category and filter to categories with multiple issues
        cursor = conn.execute(
            """
            SELECT   category,
                     COUNT(*)          AS issue_count,
                     GROUP_CONCAT(issue_id ORDER BY issue_id) AS issue_ids
            FROM     issues
            GROUP BY category
            HAVING   COUNT(*) >= 2
            ORDER BY issue_count DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()

        # Build a clean list of dicts; split the comma-separated IDs into a list
        repeat_issues = [
            {
                "category": row["category"],
                "issue_count": row["issue_count"],
                "issue_ids": [
                    int(i) for i in row["issue_ids"].split(",")
                ],
            }
            for row in rows
        ]

        return json.dumps(repeat_issues, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get repeat issues: {str(e)}"})


@mcp.tool()
def fetch_active_issues() -> str:
    """
    Retrieves all active (Open or In Progress) customer issues from the SQLite database.
    Ordered by created_at DESC (newest first).

    Returns:
        str: JSON string containing a list of active issues, or an error message if issues database fails.
    """
    try:
        # Check if database file exists before connecting
        if not os.path.exists(ISSUES_DB):
            return json.dumps({"error": f"Database file not found at: {ISSUES_DB}"})

        conn = _get_issues_connection()
        cursor = conn.execute(
            """
            SELECT * FROM issues 
            WHERE status = 'Open' OR status = 'In Progress' 
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()

        # Convert sqlite3.Row objects to plain dicts for JSON serialization
        active_issues = [dict(row) for row in rows]
        return json.dumps(active_issues, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch active issues: {str(e)}"})


if __name__ == "__main__":
    # Run server on stdio transport for MCP host connection
    mcp.run(transport="stdio")

