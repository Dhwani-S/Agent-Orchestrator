from __future__ import annotations

import sys
import uuid
import asyncio
import subprocess
import time
import os
from datetime import datetime
from pathlib import Path

from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from schemas import Goal
from memory import Memory
from artifacts import ArtifactStore
import perception
import decision
import action

MAX_ITERATIONS = 15
MCP_SERVER_PATH = "mcp_server.py.py"
GATEWAY_URL = os.getenv("LLM_GATEWAY_V7_URL", "http://localhost:8107")
LOGS_DIR = Path("logs")


memory = Memory()
artifact_store = ArtifactStore()

# ---------------------------------------------------------------------------
# Dual logger: prints to stdout AND writes to a log file in logs/
# ---------------------------------------------------------------------------

class DualLogger:
    """Writes every line to both stdout and a log file."""

    def __init__(self, log_path: Path):
        LOGS_DIR.mkdir(exist_ok=True)
        self.file = open(log_path, "w", encoding="utf-8")

    def log(self, msg: str = ""):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


def _query_slug(query: str) -> str:
    """Short slug from query for the log filename."""
    words = query.lower().split()[:5]
    slug = "_".join(w for w in words if w.isalnum())
    return slug[:40] or "query"


def ensure_gateway():
    import httpx
    try:
        httpx.get(f"{GATEWAY_URL}/v1/capabilities", timeout=5)
    except Exception:
        raise RuntimeError(f"Gateway not reachable at {GATEWAY_URL}. Start it with: cd llm_gateway && python main.py")


@asynccontextmanager
async def mcp_session():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_PATH],
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def load_tools(session: ClientSession) -> list[dict]:
    result = await session.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in result.tools
    ]


def final_answer_from(history: list[dict]) -> str:
    answers = [e["text"] for e in history if e.get("kind") == "answer"]
    if answers:
        return answers[-1]
    descriptors = [e.get("result_descriptor", "") for e in history if e.get("kind") == "action"]
    return descriptors[-1] if descriptors else "(no answer produced)"


async def run(query: str) -> str:
    ensure_gateway()
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    # Set up dual logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _query_slug(query)
    log_path = LOGS_DIR / f"{ts}_{slug}.log"
    logger = DualLogger(log_path)
    L = logger.log

    L(f"\n{'='*60}")
    L(f"[agent6] Query: {query}")
    L(f"[agent6] Run ID: {run_id}")
    L(f"[agent6] Timestamp: {datetime.now().isoformat()}")
    L(f"{'='*60}")

    try:
        memory.remember(query, source="user_query", run_id=run_id)
        L("[memory.remember] Query classified and stored.")
    except Exception as e:
        L(f"[agent6] memory.remember warning: {e}")

    async with mcp_session() as session:
        tools = await load_tools(session)
        L(f"[mcp] Connected — {len(tools)} tools available.")

        for it in range(1, MAX_ITERATIONS + 1):
            L(f"\n--- iter {it} ---")

            hits = memory.read(query, history)
            L(f"[memory.read]   {len(hits)} hits")

            obs = perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals

            for g in obs.goals:
                status = "done" if g.done else "open"
                attach = f"  attach={g.attach_artifact_id}" if g.attach_artifact_id else ""
                L(f"[perception]    [{status}] {g.text}{attach}")

            if obs.all_done:
                L(f"\n[done] all {len(obs.goals)} goals satisfied")
                break

            goal = obs.next_unfinished()

            attached = []
            if goal.attach_artifact_id and artifact_store.exists(goal.attach_artifact_id):
                blob = artifact_store.get_bytes(goal.attach_artifact_id)
                attached.append((goal.attach_artifact_id, blob))
                L(f"[attach]        {goal.attach_artifact_id} ({len(blob)} bytes)")

            out = decision.next_step(goal, hits, attached, history, tools, user_query=query)

            if out.is_answer:
                L(f"[decision]      ANSWER: {out.answer[:300]}")
                history.append({
                    "iter": it, "kind": "answer",
                    "goal_id": goal.id, "text": out.answer,
                })
                continue

            L(f"[decision]      TOOL_CALL: {out.tool_call.name}({out.tool_call.arguments})")

            result_text, art_id = await action.execute(session, out.tool_call, artifact_store)
            art_tag = f" [artifact: {art_id}]" if art_id else ""
            L(f"[action]        -> {result_text[:300]}{art_tag}")

            memory.record_outcome(
                tool_call=out.tool_call,
                result_text=result_text,
                artifact_id=art_id,
                run_id=run_id,
                goal_id=goal.id,
            )

            history.append({
                "iter": it, "kind": "action",
                "goal_id": goal.id, "tool": out.tool_call.name,
                "arguments": out.tool_call.arguments,
                "result_descriptor": result_text[:300],
                "artifact_id": art_id,
            })
        else:
            L(f"\n[warning] max iterations ({MAX_ITERATIONS}) reached")

    answer = final_answer_from(history)
    L(f"\n{'='*60}")
    L(f"FINAL: {answer}")
    L(f"Iterations: {min(it, MAX_ITERATIONS)}")
    L(f"{'='*60}\n")
    
    logger.close()
    print(f"\n[log saved] {log_path}")
    return answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent6.py \"<query>\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    asyncio.run(run(query))
