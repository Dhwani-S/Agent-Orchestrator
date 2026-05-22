from __future__ import annotations

import json
from schemas import Goal, Observation, MemoryItem
from llm_gateway.client import LLM

SYNTHESIS_KEYWORDS = {"synthesise", "synthesize", "extract", "list", "compare",
                      "decide", "choose", "select", "summarize", "summarise",
                      "consolidate", "combine", "analyze", "analyse"}

PERCEPTION_SYSTEM = """You are the Perception layer of a cognitive agent. Your job:

1. FIRST CALL (prior_goals is empty): Decompose the user's query into a list of bounded goals.
   Each goal is a short imperative statement (e.g., "Fetch the Wikipedia page for Claude Shannon").
   Order goals logically — prerequisites first.

2. LATER CALLS (prior_goals is not empty): Review the run history.
   - Mark a goal done=true when the history contains an action that satisfies it.
   - Once done, a goal stays done forever.
   - Do NOT reorder, insert, or drop goals.

3. ARTIFACT ATTACHMENT: For the first unfinished goal, decide if it needs raw bytes
   from a previously fetched artifact. If yes, set artifact_index to the integer index
   of the relevant memory hit (from the MEMORY HITS section). If no, set artifact_index to -1.

4. Keep the same number of goals across iterations. Preserve goal text exactly.

Output JSON with this schema:
{
  "goals": [
    {"text": "...", "done": true/false, "artifact_index": -1}
  ]
}

artifact_index: integer index into MEMORY HITS that have artifacts, or -1 if no attachment needed.
"""

OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "done": {"type": "boolean"},
                    "artifact_index": {"type": "integer"},
                },
                "required": ["text", "done", "artifact_index"],
            },
        },
    },
    "required": ["goals"],
}


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "MEMORY HITS: (none)\n"
    lines = ["MEMORY HITS:"]
    for i, h in enumerate(hits):
        art_tag = f" [artifact: {h.artifact_id}]" if h.artifact_id else ""
        lines.append(f"  [{i}] ({h.kind}) {h.descriptor}{art_tag}")
    return "\n".join(lines) + "\n"


def _format_history(history: list[dict]) -> str:
    if not history:
        return "HISTORY: (none)\n"
    lines = ["HISTORY:"]
    for event in history[-10:]:
        if event.get("kind") == "action":
            lines.append(f"  iter {event['iter']}: TOOL {event['tool']}({event.get('arguments', {})}) -> {event.get('result_descriptor', '')[:150]}")
        elif event.get("kind") == "answer":
            lines.append(f"  iter {event['iter']}: ANSWER for goal {event.get('goal_id', '?')}: {event.get('text', '')[:150]}")
    return "\n".join(lines) + "\n"


def _format_prior_goals(prior_goals: list[Goal]) -> str:
    if not prior_goals:
        return "PRIOR GOALS: (none -- first iteration, decompose the query)\n"
    lines = ["PRIOR GOALS:"]
    for i, g in enumerate(prior_goals):
        status = "done" if g.done else "open"
        lines.append(f"  [{i}] [{status}] {g.text}")
    return "\n".join(lines) + "\n"


def observe(query: str, hits: list[MemoryItem], history: list[dict],
            prior_goals: list[Goal], run_id: str) -> Observation:
    prompt = (
        f"USER QUERY: {query}\n\n"
        f"{_format_hits(hits)}\n"
        f"{_format_history(history)}\n"
        f"{_format_prior_goals(prior_goals)}\n"
        "Now output the goal list as JSON."
    )

    llm = LLM()
    resp = llm.chat(
        prompt=prompt,
        system=PERCEPTION_SYSTEM,
        provider="g",
        temperature=1.0,
        max_tokens=2048,
        response_format={"type": "json_schema", "schema": OBSERVATION_SCHEMA, "name": "observation"},
    )

    parsed = resp.get("parsed") or json.loads(resp["text"])

    # Build artifact index -> handle mapping from hits
    art_map = {}
    for i, h in enumerate(hits):
        if h.artifact_id:
            art_map[i] = h.artifact_id

    goals: list[Goal] = []
    for i, g in enumerate(parsed["goals"]):
        # Positional identity: reuse prior goal IDs if they exist
        goal_id = prior_goals[i].id if i < len(prior_goals) else f"g{i}"

        # Sticky done: once done, stays done
        done = g["done"]
        if i < len(prior_goals) and prior_goals[i].done:
            done = True

        # Map artifact_index to actual handle
        art_idx = g.get("artifact_index", -1)
        attach_id = art_map.get(art_idx) if art_idx >= 0 else None

        goals.append(Goal(id=goal_id, text=g["text"], done=done, attach_artifact_id=attach_id))

    # Force-attach for synthesis goals: if the goal has synthesis keywords
    # and no attachment was set but artifacts exist in hits, attach the most recent one
    for goal in goals:
        if goal.done or goal.attach_artifact_id:
            continue
        goal_words = set(goal.text.lower().split())
        if goal_words & SYNTHESIS_KEYWORDS and art_map:
            goal.attach_artifact_id = list(art_map.values())[-1]
        break  # only process the first unfinished goal

    return Observation(goals=goals)
