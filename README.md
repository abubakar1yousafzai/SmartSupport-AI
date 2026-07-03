![SmartSupport AI Banner](docs/banner.png)

<div align="center">

# SmartSupport AI
### Multi-Agent Customer Support & Business Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Google ADK](https://img.shields.io/badge/Google%20ADK-1.3.0-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/adk-docs/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/gemini/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MCP](https://img.shields.io/badge/MCP-FastMCP-FF6B35?style=for-the-badge)](https://github.com/jlowin/fastmcp)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![HTML5](https://img.shields.io/badge/HTML5-CSS3-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Kaggle 5-Day AI Agents Intensive — Capstone Project**
**Track: Agents for Business**

[🚀 Live Demo](#setup) • [📹 Video](#video-demo) • [🏗️ Architecture](#architecture) • [⚙️ Setup](#setup)

</div>

---

## 🎯 Problem Statement

Businesses receive hundreds of customer complaints daily but lack 
the tools to:
- Identify **recurring patterns** before they become crises
- Analyze **sentiment trends** over time
- Automatically **escalate critical issues** to the right team
- Generate **actionable recommendations** from complaint data

Manual review is slow, inconsistent, and expensive. Support teams 
spend hours reading complaints instead of fixing problems.

---

## 💡 Solution

**SmartSupport AI** is a production-ready multi-agent system that:

- 🤖 **Intelligently classifies** every complaint — self-fixable vs technical escalation
- 📊 **Analyzes patterns** across all customer issues automatically
- 🚨 **Alerts business teams** when issues become recurring
- 💬 **Drafts empathetic replies** instantly for support agents
- 📈 **Generates business recommendations** prioritized by impact

> "Instead of a human reading 500 complaints, 
>  6 AI agents analyze them in seconds and tell 
>  you exactly what to fix first."

---

## 🏗️ Architecture
Customer Portal (chat)          Business Dashboard
↓                              ↑
Customer types issue           Team manages issues
↓                              ↑
[Support Agent - ADK]          [Issue Management]
Classifies: self_fixable               ↑
or technical_escalation                ↑
↓ (if escalate=true)           ↑
[SQLite Database]  ←── MCP Tools ─────┘
↓
[Orchestrator Agent - ADK]
↓ (parallel)
[Complaint    [Sentiment   [Trend      [Insight
Identifier]   Analyzer]   Analyzer]   Reporter]
↓
[dashboard_data.json]
↓
[Business Dashboard - Charts + Recommendations]

---

## 🤖 Agents (Google ADK + Gemini 2.5 Flash)

| Agent | Role |
|---|---|
| **Support Agent** | Classifies complaints, drafts replies, decides escalation |
| **Orchestrator Agent** | Coordinates all analysis agents in pipeline |
| **Complaint Identifier** | Finds top repeated complaints, groups by category |
| **Sentiment Analyzer** | Scores each issue, tracks monthly sentiment trends |
| **Trend Analyzer** | Detects growing issues, flags worsening problems |
| **Insight Reporter** | Generates prioritized business recommendations |

---

## 🔧 MCP Tools (FastMCP Server)

| Tool | Purpose |
|---|---|
| `save_issue()` | Save customer issue to SQLite |
| `fetch_all_issues()` | Retrieve all issues |
| `fetch_active_issues()` | Get only Open/In Progress issues |
| `update_issue_status()` | Mark as Resolved/In Progress |
| `get_repeat_issues()` | Detect recurring complaints |
| `save_dashboard_data()` | Save analysis results for dashboard |

---

## ✨ Key Features

### 🧠 Smart Escalation Logic
self_fixable → Agent gives troubleshooting steps
Issue saved as "Pending Customer Action"
Orchestrator NOT triggered yet
technical_escalation → Agent escalates immediately
Issue saved as "Open"
Orchestrator triggered instantly
Business dashboard updated

### 📊 Business Intelligence Dashboard
- Real-time sentiment trend charts
- Top complaint categories bar chart
- Daily issue volume tracking
- AI-generated business recommendations
- Issue management with status updates
- Recurring issue alerts

### 💬 Customer Support Portal
- WhatsApp-style chat interface
- Instant AI replies
- Smart classification badges
- Escalation status indicators

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| AI Framework | Google ADK 1.3.0 |
| LLM | Gemini 2.5 Flash |
| Backend | FastAPI + Uvicorn |
| MCP Server | FastMCP |
| Database | SQLite3 |
| Frontend | HTML5 + CSS3 + JavaScript |
| Charts | Chart.js |
| Package Manager | pip |
| Language | Python 3.13 |

---

## 📁 Project Structure
customer_insight_agent/
├── .env                          # API keys (not in git)
├── requirements.txt              # Dependencies
├── api/
│   └── server.py                 # FastAPI backend
├── agents/
│   ├── complaint_identifier.py   # Finds repeated complaints
│   ├── sentiment_analyzer.py     # Analyzes sentiment trends
│   ├── trend_analyzer.py         # Monthly trend analysis
│   ├── insight_reporter.py       # Business recommendations
│   └── support_agent.py          # Smart reply + escalation
├── orchestrator/
│   └── orchestrator_agent.py     # Pipeline coordinator
├── mcp_tools/
│   └── data_tool.py              # MCP Server (6 tools)
├── frontend/
│   ├── dashboard.html            # Business dashboard
│   └── customer.html             # Customer chat portal
├── data/
│   └── issues.db                 # SQLite database
└── README.md

---

## ⚙️ Setup

### Prerequisites
- Python 3.13+
- Gemini API Key from [Google AI Studio](https://aistudio.google.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/customer_insight_agent.git
cd customer_insight_agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Start the server
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# 5. Open in browser
# Business Dashboard: http://localhost:8000
# Customer Portal:    http://localhost:8000/customer
# API Docs:           http://localhost:8000/docs
```

---

## 📹 Video Demo

[![Watch Demo](https://img.shields.io/badge/YouTube-Watch%20Demo-FF0000?style=for-the-badge&logo=youtube)](YOUR_YOUTUBE_LINK)

---

## 🎓 Course Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Multi-Agent System (ADK) | 6 specialized agents with orchestrator |
| MCP Server | FastMCP with 6 tools |
| Security | API keys in .env, CORS, input validation |
| Deployability | FastAPI + Uvicorn, one command setup |
| Antigravity IDE | Used for development and agent building |

---

## 📄 License

MIT License — feel free to use and modify.

---

<div align="center">
Built with ❤️ using Google ADK + Gemini 2.5 Flash
<br>
Kaggle 5-Day AI Agents Intensive Capstone 2026
</div>