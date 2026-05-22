from __future__ import annotations

import sys
import uuid
import asyncio
import subprocess
import time

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
GATEWAY_URL = "http://localhost:8101"


memory = Memory()
artifact_store = ArtifactStore()


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

    print(f"\n{'='*60}")
    print(f"[agent6] Query: {query}")
    print(f"[agent6] Run ID: {run_id}")
    print(f"{'='*60}")

    try:
        memory.remember(query, source="user_query", run_id=run_id)
    except Exception as e:
        print(f"[agent6] memory.remember warning: {e}")

    async with mcp_session() as session:
        tools = await load_tools(session)

        for it in range(1, MAX_ITERATIONS + 1):
            print(f"\n--- iter {it} ---")

            hits = memory.read(query, history)
            print(f"[memory.read]   {len(hits)} hits")

            obs = perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals

            for g in obs.goals:
                status = "done" if g.done else "open"
                attach = f"  attach={g.attach_artifact_id}" if g.attach_artifact_id else ""
                print(f"[perception]    [{status}] {g.text}{attach}")

            if obs.all_done:
                print("\n[done] all goals satisfied")
                break

            goal = obs.next_unfinished()

            attached = []
            if goal.attach_artifact_id and artifact_store.exists(goal.attach_artifact_id):
                blob = artifact_store.get_bytes(goal.attach_artifact_id)
                attached.append((goal.attach_artifact_id, blob))
                print(f"[attach]        {goal.attach_artifact_id} ({len(blob)} bytes)")

            out = decision.next_step(goal, hits, attached, history, tools)

            if out.is_answer:
                print(f"[decision]      ANSWER: {out.answer[:150]}")
                history.append({
                    "iter": it, "kind": "answer",
                    "goal_id": goal.id, "text": out.answer,
                })
                continue

            print(f"[decision]      TOOL_CALL: {out.tool_call.name}({out.tool_call.arguments})")

            result_text, art_id = await action.execute(session, out.tool_call, artifact_store)
            print(f"[action]        -> {result_text[:150]}")

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

    answer = final_answer_from(history)
    print(f"\n{'='*60}")
    print(f"FINAL: {answer}")
    print(f"{'='*60}\n")
    return answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent6.py \"<query>\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    asyncio.run(run(query))
