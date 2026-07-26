"""
Xen AI — Claude Mind Starter trial backend.

Runs the Planner -> Executor -> Planner loop server-side using OUR
Anthropic API key, so trial users need zero setup.

Trial model: each trial_id gets 7 runs (one per day of the 7-day trial).
State is in memory per the spec — a redeploy resets counters, which is
acceptable for a free trial gate.
"""
import json
import os
import re
import threading
import datetime
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODEL = os.getenv("CLAUDE_MIND_MODEL", "claude-3-5-sonnet-20240620")
API_KEY = os.getenv("ANTHROPIC_API_KEY")
BUY_URL = "https://claudemind.gumroad.com/l/zfseds"
MAX_RUNS = 7
COST_PER_M = (3.0, 15.0)  # sonnet $/M tokens (input, output)

# Guardrails so a single request can't burn the shared key
MAX_VAULT_CHARS = 120_000     # total incoming vault text
MAX_FILE_COUNT = 60
TRIAL_ID_RE = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")

app = FastAPI(title="Xen AI — Claude Mind Starter trial backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- trial store
_runs: Dict[str, int] = {}
_lock = threading.Lock()


def consume_run(trial_id: str) -> Optional[int]:
    """Reserve one run. Returns runs used AFTER this run, or None if expired."""
    with _lock:
        used = _runs.get(trial_id, 0)
        if used >= MAX_RUNS:
            return None
        _runs[trial_id] = used + 1
        return used + 1


def refund_run(trial_id: str) -> None:
    """Give the run back if we failed before doing any model work."""
    with _lock:
        if _runs.get(trial_id, 0) > 0:
            _runs[trial_id] -= 1


# ---------------------------------------------------------------- web tools
def search_web(query: str, max_results: int = 3) -> str:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=max_results))
        return json.dumps(
            [{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")} for r in results],
            indent=2,
        )
    except Exception as e:  # search must never kill the round
        return f"Search error: {e}"


def fetch_page(url: str, max_chars: int = 2000) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={"User-Agent": "ClaudeMindStarter/1.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:max_chars]
    except Exception as e:
        return f"Error fetching {url}: {e}"


TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web with DuckDuckGo. Returns JSON results.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "fetch_page",
        "description": "Fetch a web page and return its visible text.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "max_chars": {"type": "integer", "default": 2000}},
            "required": ["url"],
        },
    },
]

TOOL_FUNCS = {"search_web": search_web, "fetch_page": fetch_page}


# ---------------------------------------------------------------- Claude
class CostMeter:
    def __init__(self):
        self.in_tok = 0
        self.out_tok = 0

    def add(self, usage):
        self.in_tok += usage.input_tokens
        self.out_tok += usage.output_tokens

    @property
    def dollars(self) -> float:
        return (self.in_tok / 1e6) * COST_PER_M[0] + (self.out_tok / 1e6) * COST_PER_M[1]


def call_claude(client, system: str, user: str, meter: CostMeter, max_tokens: int = 2000) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    meter.add(resp.usage)
    return resp.content[0].text if resp.content else ""


def run_executor(client, brief: str, meter: CostMeter) -> str:
    """Executor with a real tool-use loop (search + fetch), capped at 6 tool calls."""
    system = ("You are the Executor. Execute the given briefs using the tools when research is needed. "
              "Then write a concise report with cited sources.")
    messages = [{"role": "user", "content": brief}]
    for _ in range(6):
        resp = client.messages.create(
            model=MODEL, max_tokens=2000, system=system,
            messages=messages, tools=TOOLS, tool_choice={"type": "auto"},
        )
        meter.add(resp.usage)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        # answer every tool call in this turn
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                fn = TOOL_FUNCS.get(block.name)
                out = fn(**block.input) if fn else f"Unknown tool {block.name}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})
    return "Executor stopped after the tool-call limit. Partial work only."


def run_round(vault_files: Dict[str, str]) -> dict:
    """The Planner -> Executor -> Planner loop from run_agent.py, statelessly."""
    from anthropic import Anthropic

    client = Anthropic(api_key=API_KEY)
    meter = CostMeter()

    def vf(name_part: str) -> str:
        for path, content in vault_files.items():
            if name_part.lower() in path.lower():
                return content
        return ""

    claude_md = vf("CLAUDE.md")
    tasks = vf("TASKS.md")
    loops = vf("Open Loops")
    agenda = vf("Research Agenda")
    profile = vf("Profile")

    # 1. Planner — briefs
    planner_prompt = f"""
You are the Planner. Read the vault state and produce briefs for the Executor.

Vault:
- CLAUDE.md: {claude_md}
- Tasks: {tasks}
- Open Loops: {loops}
- Research Agenda: {agenda}
- Owner Profile: {profile}

Decide on the top 2-3 actions (tasks and/or research questions). Write briefs in this exact format:

Brief 1: <title>
Description: <what to do>
Outcome: <desired result>
"""
    briefs = call_claude(client, planner_prompt, "Produce today's briefs.", meter, max_tokens=1200)

    # 2. Executor — do the work (may search the web)
    report = run_executor(client, briefs, meter)

    # 3. Planner — review and update the vault
    review_prompt = f"""
You are the Planner (review phase). Review the Executor's report and update the vault.

Executor report: {report}

Current TASKS.md: {tasks}
Current Research Agenda: {agenda}

Actions:
- Mark completed tasks as done; return TASKS.md with only open tasks.
- Remove answered research questions from the agenda.
- List new durable owner facts (or empty string).
- Write a short summary of what was accomplished.

Return ONLY valid JSON, no code fences:
{{"tasks_md": "...", "agenda_md": "...", "profile_update": "...", "summary": "..."}}
"""
    review = call_claude(client, review_prompt, "Review the round and update the vault.", meter, max_tokens=3000)
    try:
        cleaned = re.sub(r"^```(json)?|```$", "", review.strip(), flags=re.M).strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {"tasks_md": tasks, "agenda_md": agenda, "profile_update": "",
                "summary": "Round completed, but the review step returned unparseable output; vault left unchanged."}

    # Build updated files, only for paths that exist in the request
    updated: Dict[str, str] = {}
    for path in vault_files:
        low = path.lower()
        if "tasks.md" in low and data.get("tasks_md"):
            updated[path] = data["tasks_md"]
        elif "research agenda" in low and data.get("agenda_md"):
            updated[path] = data["agenda_md"]
        elif "profile" in low and data.get("profile_update"):
            updated[path] = vault_files[path].rstrip() + "\n\n" + data["profile_update"] + "\n"
    handoff = (f"## Planner -> Executor\n{briefs}\n\n## Executor -> Planner\n{report}\n\n"
               f"*Round of {datetime.date.today().isoformat()}*\n")
    for path in vault_files:
        if "handoff" in path.lower():
            updated[path] = handoff
            break
    else:
        updated["Agents/Handoff.md"] = handoff

    return {
        "summary": data.get("summary", ""),
        "briefs": briefs,
        "report": report,
        "updated_files": updated,
        "tokens": {"input": meter.in_tok, "output": meter.out_tok},
        "estimated_cost_usd": round(meter.dollars, 4),
    }


# ---------------------------------------------------------------- API
class RunRequest(BaseModel):
    trial_id: str
    vault_files: Dict[str, str] = Field(default_factory=dict)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run(req: RunRequest):
    if not TRIAL_ID_RE.match(req.trial_id.strip().lower()):
        raise HTTPException(status_code=400, detail="Invalid trial ID.")
    if not req.vault_files or len(req.vault_files) > MAX_FILE_COUNT:
        raise HTTPException(status_code=400, detail="vault_files must contain 1-60 files.")
    if sum(len(v) for v in req.vault_files.values()) > MAX_VAULT_CHARS:
        raise HTTPException(status_code=413, detail="Vault too large for the trial (120k character limit).")

    trial_id = req.trial_id.strip().lower()
    used = consume_run(trial_id)
    if used is None:
        return JSONResponse(
            status_code=403,
            content={"error": f"Trial expired. Buy the full version at {BUY_URL}", "expired": True},
        )

    if not API_KEY:
        refund_run(trial_id)
        raise HTTPException(status_code=503, detail="Backend not configured (missing API key). Try again later.")

    try:
        result = run_round(req.vault_files)
    except Exception as e:
        refund_run(trial_id)
        raise HTTPException(status_code=502, detail=f"Agent round failed: {e}")

    result["runs_used"] = used
    result["runs_left"] = MAX_RUNS - used
    return result
