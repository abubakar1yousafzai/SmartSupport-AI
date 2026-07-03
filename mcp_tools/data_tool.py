"""
MCP Server for Customer Insight Agent.
Exposes tools to read customer reviews, write analytics reports,
dashboard data, and manage customer issues via SQLite.

New in this version:
  - create_session()             : Generate a unique session ID and ensure
                                   the conversations table exists.
  - get_conversation_history()   : Retrieve chat history for a session.
  - save_conversation_message()  : Persist a single chat turn to the DB.
  - save_issue() updated         : Now accepts an optional session_id so
                                   issues can be linked to a conversation.
"""

import os
import json
import uuid
import sqlite3
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Customer Insight Data Tool")

# ---------------------------------------------------------------------------
# Absolute file paths – everything is anchored to the project root (BASE_DIR)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_DATA_FILE = os.path.join(BASE_DIR, "data", "dashboard_data.json")
ISSUES_DB = os.path.join(BASE_DIR, "data", "issues.db")

# Note: Data now comes from SQLite via customer portal chat

# ---------------------------------------------------------------------------
# Internal helpers
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


def _ensure_session_id_column(conn: sqlite3.Connection) -> None:
    """
    Adds the session_id column to the issues table if it does not already exist.
    Uses ALTER TABLE … ADD COLUMN which is a safe, idempotent-like operation
    (wrapped in try/except so duplicate-column errors are silently ignored).
    """
    try:
        conn.execute("ALTER TABLE issues ADD COLUMN session_id TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists – nothing to do
        pass


def _get_conversations_connection() -> sqlite3.Connection:
    """
    Opens a connection to the shared issues SQLite database and ensures the
    'conversations' table exists.  Reuses ISSUES_DB so that sessions and
    issues live in the same file and can be joined easily.
    """
    os.makedirs(os.path.dirname(ISSUES_DB), exist_ok=True)

    conn = sqlite3.connect(ISSUES_DB)
    conn.row_factory = sqlite3.Row

    # Create conversations table if it does not exist yet
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER  PRIMARY KEY AUTOINCREMENT,
            session_id TEXT     NOT NULL,
            role       TEXT     NOT NULL,
            message    TEXT     NOT NULL,
            issue_id   INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


# ===========================================================================
# TOOL 1 (existing) – save_dashboard_data
# ===========================================================================

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

# ===========================================================================
# TOOL 2 (existing, updated) – save_issue
#   Change: added optional session_id parameter; the issues table gets the
#           session_id column added if it is missing (safe, idempotent).
# ===========================================================================

@mcp.tool()
def save_issue(
    complaint_text: str,
    agent_reply: str,
    category: str,
    sentiment: str,
    priority: str,
    status: str = "Open",
    session_id: str = None,
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
        session_id     (str): Optional session ID to link the issue to a conversation.

    Returns:
        str: JSON string with the newly created issue_id, or an error message.
    """
    try:
        conn = _get_issues_connection()

        # Ensure the session_id column exists (safe even if already present)
        _ensure_session_id_column(conn)

        # Capture the current UTC timestamp for created_at
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        cursor = conn.execute(
            """
            INSERT INTO issues
                (complaint_text, agent_reply, category, sentiment, priority,
                 status, created_at, resolved_at, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (complaint_text, agent_reply, category, sentiment, priority,
             status, created_at, session_id),
        )
        conn.commit()

        # Retrieve the auto-assigned primary key
        issue_id = cursor.lastrowid
        conn.close()

        return json.dumps({"success": True, "issue_id": issue_id})
    except Exception as e:
        return json.dumps({"error": f"Failed to save issue: {str(e)}"})


# ===========================================================================
# TOOL 3 (existing) – fetch_all_issues
# ===========================================================================

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


# ===========================================================================
# TOOL 4 (existing) – update_issue_status
# ===========================================================================

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


# ===========================================================================
# TOOL 5 (existing) – get_repeat_issues
# ===========================================================================

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


# ===========================================================================
# TOOL 6 (existing) – fetch_active_issues
# ===========================================================================

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


# ===========================================================================
# TOOL 7 (new) – create_session
# ===========================================================================

@mcp.tool()
def create_session() -> str:
    """
    Generates a unique session ID and ensures the conversations table exists
    in the database so that subsequent save_conversation_message() calls work
    without any further setup.

    Session ID format: "sess_" + first 8 hex characters of a UUID4.
    Example: "sess_a1b2c3d4"

    Returns:
        str: The newly created session_id string.
    """
    # Build a short, human-readable session identifier
    session_id = "sess_" + str(uuid.uuid4()).replace("-", "")[:8]

    # Opening the connection triggers CREATE TABLE IF NOT EXISTS for conversations
    conn = _get_conversations_connection()
    conn.close()

    return session_id


# ===========================================================================
# TOOL 8 (new) – get_conversation_history
# ===========================================================================

@mcp.tool()
def get_conversation_history(session_id: str) -> str:
    """
    Retrieves the full conversation history for a given session, plus the
    most recent issue linked to that session (if any).

    Queries:
        1. SELECT all rows from conversations WHERE session_id = ?
           ORDER BY created_at ASC
        2. SELECT issue_id, status, category FROM issues
           WHERE session_id = ?
           ORDER BY created_at DESC LIMIT 1

    Args:
        session_id (str): The session identifier returned by create_session().

    Returns:
        str: JSON string with the shape:
             {
               "session_id": "sess_abc123",
               "messages": [
                 {"role": "customer", "message": "...", "created_at": "..."},
                 {"role": "agent",    "message": "...", "created_at": "..."}
               ],
               "last_issue": {
                 "issue_id": 7,
                 "status": "Pending Customer Action",
                 "category": "Technical Support"
               }
             }
             If no history is found, messages will be [] and last_issue null.
    """
    try:
        conn = _get_conversations_connection()

        # ------------------------------------------------------------------
        # 1. Fetch all conversation turns for this session, oldest first
        # ------------------------------------------------------------------
        msg_cursor = conn.execute(
            """
            SELECT role, message, created_at
            FROM   conversations
            WHERE  session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        messages = [dict(row) for row in msg_cursor.fetchall()]

        # ------------------------------------------------------------------
        # 2. Fetch the most recent issue linked to this session (if any)
        #    Guard against the session_id column not existing yet on older DBs.
        # ------------------------------------------------------------------
        last_issue = None
        try:
            issue_cursor = conn.execute(
                """
                SELECT issue_id, status, category
                FROM   issues
                WHERE  session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = issue_cursor.fetchone()
            if row:
                last_issue = dict(row)
        except sqlite3.OperationalError:
            # session_id column doesn't exist in older DB – treat as no issue
            last_issue = None

        conn.close()

        return json.dumps(
            {
                "session_id": session_id,
                "messages": messages,
                "last_issue": last_issue,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": f"Failed to get conversation history: {str(e)}"})


# ===========================================================================
# TOOL 9 (new) – save_conversation_message
# ===========================================================================

@mcp.tool()
def save_conversation_message(
    session_id: str,
    role: str,
    message: str,
    issue_id: int = None,
) -> str:
    """
    Persists a single conversation turn (one message) to the conversations table.

    Args:
        session_id (str): The session identifier returned by create_session().
        role       (str): Speaker role, e.g. 'customer' or 'agent'.
        message    (str): The message text to store.
        issue_id   (int): Optional – the issue_id to associate with this turn.

    Returns:
        str: JSON string indicating success (with the new row id) or failure.
             Success: {"success": true, "id": <row_id>}
             Failure: {"success": false, "error": "<reason>"}
    """
    try:
        conn = _get_conversations_connection()

        cursor = conn.execute(
            """
            INSERT INTO conversations (session_id, role, message, issue_id)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, message, issue_id),
        )
        conn.commit()

        # Return the auto-assigned row id as confirmation
        row_id = cursor.lastrowid
        conn.close()

        return json.dumps({"success": True, "id": row_id})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Failed to save message: {str(e)}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run server on stdio transport for MCP host connection
    mcp.run(transport="stdio")
