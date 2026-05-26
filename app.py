"""
Web frontend for agent6.  Run with:
    python app.py
Then open http://localhost:8501
Requires the gateway running on :8101.
"""
from __future__ import annotations

import asyncio
import json
import uuid
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from memory import Memory
from artifacts import ArtifactStore
from agent6 import mcp_session, load_tools, final_answer_from, ensure_gateway, MAX_ITERATIONS
import perception
import decision
import action

app = FastAPI(title="Agent6 Chat")
app.mount("/static", StaticFiles(directory="static"), name="static")

memory = Memory()
artifact_store = ArtifactStore()


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/api/chat")
async def chat(q: str):
    async def event_stream():
        run_id = uuid.uuid4().hex[:8]
        history: list[dict] = []
        prior_goals = []

        def send(data: dict):
            return f"data: {json.dumps(data)}\n\n"

        yield send({"type": "start", "run_id": run_id, "query": q})

        try:
            ensure_gateway()
        except (SystemExit, RuntimeError) as e:
            yield send({"type": "error", "message": str(e)})
            return

        try:
            memory.remember(q, source="user_query", run_id=run_id)
            yield send({"type": "log", "message": "Query memorised."})
        except Exception as e:
            yield send({"type": "log", "message": f"Memory classify warning: {e}"})

        try:
            async with mcp_session() as session:
                tools = await load_tools(session)
                yield send({"type": "log", "message": f"MCP connected — {len(tools)} tools available."})

                for it in range(1, MAX_ITERATIONS + 1):
                    hits = memory.read(q, history)
                    yield send({"type": "iter_start", "iter": it, "hits": len(hits)})

                    obs = perception.observe(q, hits, history, prior_goals, run_id)
                    prior_goals = obs.goals

                    goals_data = []
                    for g in obs.goals:
                        goals_data.append({
                            "text": g.text,
                            "done": g.done,
                            "attach": g.attach_artifact_id,
                        })
                    yield send({"type": "perception", "iter": it, "goals": goals_data})

                    if obs.all_done:
                        yield send({"type": "all_done", "iter": it})
                        break

                    goal = obs.next_unfinished()

                    attached = []
                    if goal.attach_artifact_id and artifact_store.exists(goal.attach_artifact_id):
                        blob = artifact_store.get_bytes(goal.attach_artifact_id)
                        attached.append((goal.attach_artifact_id, blob))
                        yield send({"type": "attach", "artifact_id": goal.attach_artifact_id, "size": len(blob)})

                    try:
                        out = decision.next_step(goal, hits, attached, history, tools, user_query=q)
                    except Exception as dec_err:
                        err_msg = str(dec_err)
                        if "503" in err_msg or "429" in err_msg or "Service Unavailable" in err_msg:
                            yield send({"type": "log", "message": f"Rate limited — waiting 15s before retry (iter {it})..."})
                            import asyncio as _aio
                            await _aio.sleep(15)
                            try:
                                out = decision.next_step(goal, hits, attached, history, tools, user_query=q)
                            except Exception as retry_err:
                                yield send({"type": "error", "message": f"Gateway still unavailable after retry: {retry_err}"})
                                return
                        else:
                            raise

                    if out.is_answer:
                        yield send({"type": "answer", "iter": it, "goal": goal.text, "text": out.answer})
                        history.append({
                            "iter": it, "kind": "answer",
                            "goal_id": goal.id, "text": out.answer,
                        })
                        continue

                    yield send({
                        "type": "tool_call", "iter": it, "goal": goal.text,
                        "tool": out.tool_call.name,
                        "arguments": out.tool_call.arguments,
                    })

                    result_text, art_id = await action.execute(session, out.tool_call, artifact_store)
                    yield send({
                        "type": "action_result", "iter": it,
                        "result": result_text[:500],
                        "artifact_id": art_id,
                    })

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

            final = final_answer_from(history)
            yield send({"type": "final", "answer": final})

        except BaseException as e:
            traceback.print_exc()
            # Unwrap ExceptionGroup / BaseExceptionGroup to show the real error
            real = e
            while isinstance(real, BaseExceptionGroup) and real.exceptions:
                real = real.exceptions[0]
            yield send({"type": "error", "message": str(real)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
