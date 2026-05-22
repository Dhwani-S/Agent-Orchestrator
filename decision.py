from __future__ import annotations

import json
from schemas import Goal, MemoryItem, DecisionOutput, ToolCall
from llm_gateway.client import LLM

DECISION_SYSTEM = """You are the Decision layer of a cognitive agent. You receive ONE goal at a time.

Rules:
1. Respond with EXACTLY ONE output: either call a tool OR give a final answer in plain text. Never both.
2. Strings starting with "art:" are internal artifact handles. They are NOT file paths or URLs.
   Do NOT pass them to tools like read_file or fetch_url. If you need artifact content,
   it will appear under ATTACHED ARTIFACTS in your prompt — read it there.
3. When the goal asks for extraction, listing, comparison, or selection, your answer must be
   substantive: at least three sentences or a list of items. Do not give a meta-answer like
   "the page has been fetched" — do the actual work.
"""


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return ""
    lines = ["MEMORY HITS:"]
    for h in hits:
        art_tag = f" [artifact: {h.artifact_id}]" if h.artifact_id else ""
        lines.append(f"  ({h.kind}) {h.descriptor}{art_tag}")
    return "\n".join(lines) + "\n"


def _format_attached(attached: list[tuple[str, bytes]]) -> str:
    if not attached:
        return ""
    lines = ["ATTACHED ARTIFACTS:"]
    for art_id, blob in attached:
        try:
            text = blob.decode("utf-8", errors="replace")[:50_000]
        except Exception:
            text = f"[binary, {len(blob)} bytes]"
        lines.append(f"--- {art_id} ---")
        lines.append(text)
    return "\n".join(lines) + "\n"


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["RECENT HISTORY:"]
    for event in history[-10:]:
        if event.get("kind") == "action":
            lines.append(f"  iter {event['iter']}: TOOL {event['tool']}({event.get('arguments', {})}) -> {event.get('result_descriptor', '')[:150]}")
        elif event.get("kind") == "answer":
            lines.append(f"  iter {event['iter']}: ANSWER: {event.get('text', '')[:150]}")
    return "\n".join(lines) + "\n"


def next_step(goal: Goal, hits: list[MemoryItem],
              attached: list[tuple[str, bytes]],
              history: list[dict],
              mcp_tools: list[dict]) -> DecisionOutput:

    prompt = (
        f"CURRENT GOAL: {goal.text}\n\n"
        f"{_format_hits(hits)}\n"
        f"{_format_attached(attached)}\n"
        f"{_format_history(history)}\n"
        "Decide: call one tool OR give a final answer."
    )

    llm = LLM()
    resp = llm.chat(
        prompt=prompt,
        system=DECISION_SYSTEM,
        auto_route="decision",
        tools=mcp_tools,
        tool_choice="auto",
        max_tokens=4096,
    )

    if resp.get("tool_calls"):
        tc = resp["tool_calls"][0]
        args = tc.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return DecisionOutput(
            tool_call=ToolCall(
                name=tc["name"],
                arguments=args,
            )
        )

    return DecisionOutput(answer=resp.get("text", "(no response)"))
