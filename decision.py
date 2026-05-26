from __future__ import annotations

import json
from schemas import Goal, MemoryItem, DecisionOutput, ToolCall
from llm_gateway.client import LLM

DECISION_SYSTEM = """You are the Decision layer of a cognitive agent. You receive ONE goal at a time.

Think step-by-step before responding:
  Step 1: Identify the reasoning type — is this a fetch, extract, calculate, lookup, or synthesize task?
  Step 2: Check ATTACHED ARTIFACTS and RECENT HISTORY — do you already have enough information to answer?
  Step 3: If yes, give a substantive final answer. If no, pick the single best tool to call.
  Step 4: Verify your choice — does your answer actually address the goal, or are you giving a meta-answer?

Rules:
1. Respond with EXACTLY ONE output: either call a tool OR give a final answer in plain text. Never both.
2. Strings starting with "art:" are internal artifact handles. They are NOT file paths or URLs.
   Do NOT pass them to tools like read_file or fetch_url. If you need artifact content,
   it will appear under ATTACHED ARTIFACTS in your prompt — read it there.
3. When the goal asks for extraction, listing, comparison, or selection, your answer must be
   substantive: at least three sentences or a list of items. Do not give a meta-answer like
   "the page has been fetched" — do the actual work.
4. When the user query or goal mentions a specific URL, use fetch_url with that exact URL.
   Do NOT use web_search to find a page when you already have its URL.
5. SELF-CHECK: Before returning an answer, confirm it directly addresses the goal.
   If your answer would just restate what a tool returned without analysis, do the analysis.
6. ERROR HANDLING: If RECENT HISTORY shows a tool failed (403, timeout, error),
   try a different tool or approach. Do not retry the exact same call.
   If you are unsure, prefer giving a partial answer over making no progress.
7. NO REPEATS: If RECENT HISTORY shows search_knowledge returned empty or no results,
   do NOT call search_knowledge again. Instead, use read_file to get the document content
   directly, or give your answer based on information already in RECENT HISTORY.
8. USE WHAT YOU HAVE: If RECENT HISTORY contains read_file results with document content,
   you already have enough information to extract key points and answer analytical questions.
   Do the analysis from the content you have — do not call more tools.
9. NEVER RE-READ: If RECENT HISTORY already shows a successful read_file for a path,
   NEVER call read_file on that same path again. The content is already available.
   If you need more information, read a DIFFERENT file or synthesize an ANSWER from what you have.
10. CLEAN ANSWERS: Your final answer is shown directly to the user. NEVER mention internal
    tool failures, 403 errors, timeouts, or retries. Do not say "I couldn't fetch the page" or
    "the request was blocked." Just provide the answer using whatever information you gathered.
    The user does not need to know which tools succeeded or failed.
"""


def _format_hits(hits: list[MemoryItem]) -> str:
    """Format memory hits for LLM context.
    
    Transforms vector search results into readable format with artifacts and content snippets.
    
    Args:
        hits: List of memory items from FAISS search
        
    Returns:
        Formatted string for inclusion in LLM prompt, or empty string if no hits
    """
    if not hits:
        return ""
    lines = ["MEMORY HITS:"]
    for h in hits:
        art_tag = f" [artifact: {h.artifact_id}]" if h.artifact_id else ""
        lines.append(f"  ({h.kind}) {h.descriptor}{art_tag}")
        if h.value.get("chunk"):
            lines.append(f"    chunk: {h.value['chunk'][:500]}")
        elif h.value.get("raw"):
            lines.append(f"    raw: {str(h.value['raw'])[:300]}")
        elif h.value.get("value"):
            lines.append(f"    value: {str(h.value['value'])}")
    return "\n".join(lines) + "\n"


def _format_attached(attached: list[tuple[str, bytes]]) -> str:
    """Format attached artifacts for LLM context.
    
    Converts artifact blobs to readable text with size information and truncation.
    Limits output to 8000 characters to prevent overwhelming rate-limited APIs.
    
    Args:
        attached: List of (artifact_id, binary_blob) tuples
        
    Returns:
        Formatted string for inclusion in LLM prompt, or empty string if none attached
    """
    if not attached:
        return ""
    lines = ["ATTACHED ARTIFACTS:"]
    """Format recent agent history for LLM context.
    
    Shows last 10 iterations of tool calls and answers, helping LLM avoid repetition.
    Part of Rule 7-9 to prevent decision loop issues.
    
    Args:
        history: List of event dictionaries from agent execution
        
    Returns:
        Formatted string showing tool calls and answers, or empty string if no history
    """
    for art_id, blob in attached:
        try:
            text = blob.decode("utf-8", errors="replace")[:8_000]
        except Exception:
            text = f"[binary, {len(blob)} bytes]"
        lines.append(f"--- {art_id} ({len(blob)} bytes, showing first 8000 chars) ---")
        lines.append(text)
    return "\n".join(lines) + "\n"


def _format_history(history: list[dict]) -> str:
    """Format recent agent history for LLM context.
    
    Shows last 10 iterations of tool calls and answers, helping LLM avoid repetition.
    Part of Rule 7-9 to prevent decision loop issues.
    
    Args:
        history: Event list from agent execution
        
    Returns:
        Formatted string showing tool calls and answers, or empty string if no history
    """
    if not history:
        return ""
    lines = ["RECENT HISTORY:"]
    for event in history[-10:]:
        if event.get("kind") == "action":
            lines.append(f"  iter {event['iter']}: TOOL {event['tool']}({event.get('arguments', {})}) -> {event.get('result_descriptor', '')}")
        elif event.get("kind") == "answer":
            lines.append(f"  iter {event['iter']}: ANSWER: {event.get('text', '')}")
    return "\n".join(lines) + "\n"


def next_step(goal: Goal, hits: list[MemoryItem],
              attached: list[tuple[str, bytes]],
              history: list[dict],
              mcp_tools: list[dict],
              user_query: str = "") -> DecisionOutput:
    """Execute decision step: choose tool or synthesize answer.
    
    Core decision logic that evaluates context and decides whether to call a tool
    or generate a final answer. Uses DECISION_SYSTEM prompt with Rules 1-9 to
    prevent looping and ensure high-quality reasoning.
    
    Args:
        goal: Current perception goal to fulfill
        hits: Memory items from vector search (context)
        attached: Attached artifact blobs for analysis
        history: Recent iteration history (last 10 actions)
        mcp_tools: Available MCP tools with signatures
        user_query: Original user query for reference
        
    Returns:
        DecisionOutput with either tool_call or answer field populated
    """
    prompt = (
        f"USER QUERY: {user_query}\n\n"
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
