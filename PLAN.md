# Agent Orchestrator — Implementation Plan

## What We Have
- `mcp_server.py.py` — 9-tool MCP server (web_search, fetch_url, get_time, currency_convert, read_file, list_dir, create_file, update_file, edit_file)
- `llm_gateway/` — V3 gateway with auto_route, router pool, provider override, structured output
- `.env` — needs API keys (TAVILY_API_KEY, plus gateway provider keys)
- Python 3.11 venv with all deps installed

## What We Need to Build

### Files to Create
| File | Purpose |
|------|---------|
| `schemas.py` | All Pydantic v2 models: MemoryItem, Artifact, Goal, Observation, ToolCall, DecisionOutput |
| `memory.py` | Typed memory service: read (keyword search), remember (LLM classify), record_outcome, filter. Persists to `state/memory.json` |
| `perception.py` | Orchestrator role: decomposes query → goals, tracks done flags, decides artifact attachment. Pinned to Gemini via `provider="g"` |
| `decision.py` | One-goal-at-a-time action selector: returns answer OR tool_call. Routes via `auto_route="decision"` |
| `action.py` | Pure MCP dispatch + artifact store. No LLM. Handles large payloads → artifacts, blocks `art:` handles in tool args |
| `agent6.py` | Main loop wiring all four roles. CLI entry point |
| `artifacts.py` | Content-addressable artifact store under `state/artifacts/` |
| `state/` | Directory for memory.json + artifacts/ (excluded from git) |
| `pyproject.toml` | uv-compatible project config (assignment requires uv, not manual venv) |

### .gitignore additions
- `state/`

## Architecture (per iteration)

```
agent6 loop:
  1. memory.read(query, history) → hits[]
  2. perception.observe(query, hits, history, prior_goals) → Observation(goals[])
  3. if all goals done → break
  4. goal = first unfinished goal
  5. if goal.attach_artifact_id → load artifact bytes
  6. decision.next_step(goal, hits, attached, history, tools) → DecisionOutput
  7. if answer → append to history, continue
  8. if tool_call → action.execute(session, tool_call) → (descriptor, art_id?)
  9. memory.record_outcome(tool_call, result, art_id)
  10. append to history, continue
```

## Pydantic Models (schemas.py)

```python
MemoryItem:     id, kind(fact|preference|tool_outcome|scratchpad), keywords[], descriptor, value{}, artifact_id?, source, run_id, goal_id?, confidence, created_at
Artifact:       id("art:<sha256>"), content_type, size_bytes, source, descriptor
Goal:           id, text, done, attach_artifact_id?
Observation:    goals[], all_done, next_unfinished()
ToolCall:       name, arguments{}
DecisionOutput: answer? | tool_call?  (exactly one populated)
```

## Four Target Queries

### Query A — Shannon Wikipedia (artifact attach)
- **Input:** Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions
- **Expected iterations:** 3
- **Max allowed:** 6
- **Key test:** artifact attachment path — fetch produces artifact, Perception attaches it to extraction goal

### Query B — Tokyo activities + weather (multi-goal + memory carryover)
- **Input:** Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast and tell me which is most appropriate
- **Expected iterations:** ~6
- **Max allowed:** 12
- **Key test:** 3 goals, weather fact from goal 2 carries into goal 3 via memory

### Query C — Mom's birthday (durable memory across runs)
- **Run 1:** My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day
- **Run 2:** When is mom's birthday?
- **Expected iterations:** Run 1: 4, Run 2: 2
- **Max allowed:** Run 1: 8, Run 2: 4
- **Key test:** memory.remember() classifies fact at start, persists in state/memory.json, found by keyword search in run 2

### Query D — Asyncio research (multi-source synthesis)
- **Input:** Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on
- **Expected iterations:** 5–7
- **Max allowed:** 14
- **Key test:** multi-artifact attachment, synthesis goal with force-attach

## Implementation Phases

### Phase 1 — Foundation (current branch: phase-1) ✅
- [x] Repo init, venv, deps installed
- [x] MCP server in place
- [x] LLM gateway in place
- [x] .gitignore

### Phase 2 — Schemas & Memory
- [ ] Create `schemas.py` with all Pydantic v2 models
- [ ] Create `artifacts.py` (content-addressable store)
- [ ] Create `memory.py` (read/write/remember/record_outcome/filter)
- [ ] Add `state/` to .gitignore
- [ ] Unit-test memory read/write cycle

### Phase 3 — Perception & Decision
- [ ] Create `perception.py` — system prompt, observe() function, Gemini-pinned gateway call
- [ ] Create `decision.py` — system prompt, next_step() function, auto_route="decision"
- [ ] Design Perception prompt with: goal decomposition, done-flag tracking, artifact attachment via indexed references
- [ ] Design Decision prompt with: answer-or-tool rule, artifact handle warning, substantive answer rule

### Phase 4 — Action & Agent Loop
- [ ] Create `action.py` — MCP dispatch, artifact threshold (4KB), art: handle guard
- [ ] Create `agent6.py` — main loop, MCP session management, CLI
- [ ] Wire: memory → perception → decision → action cycle
- [ ] Gateway startup/health check helper

### Phase 5 — Query Testing & Tuning
- [ ] Test Query A (Shannon) — tune artifact attach
- [ ] Test Query B (Tokyo) — tune multi-goal + memory carryover
- [ ] Test Query C (Mom's birthday) — test cross-run persistence
- [ ] Test Query D (Asyncio) — tune multi-artifact synthesis
- [ ] Ensure all queries within 2x expected iteration count

### Phase 6 — Packaging & Deliverables
- [ ] Switch to `uv` for dependency management (pyproject.toml)
- [ ] README with run instructions + terminal output for all 4 queries
- [ ] YouTube demo recording
- [ ] Extract Perception & Decision prompts + PoP validation JSON

## Key Design Decisions

1. **Perception pinned to Gemini** (`provider="g"`) — smaller models can't reliably follow multi-step goal tracking
2. **Temperature 1.0 on Perception** — prevents Gemini looping at low temp
3. **Position-based goal identity** — no goal ID in LLM output, mapped by position in outer loop
4. **Indexed artifact references** — Perception emits `artifact_index: int` not string handles
5. **Force-attach for synthesis goals** — when goal text contains synthesis keywords and artifacts exist, auto-attach
6. **4KB artifact threshold** — payloads > 4KB go to artifact store, Decision gets handle + preview
7. **Art: handle guard in Action** — blocks tool calls with `art:` prefixed args, returns clear error

## Gateway Integration Notes
- All LLM calls go through `http://localhost:8101`
- Perception: `provider="g"`, `response_format` with Observation schema
- Memory.remember: `provider="g"`, `response_format` with classification schema  
- Decision: `auto_route="decision"`, `tools=mcp_tools`, `tool_choice="auto"`
- Memory.relevant (optional): `auto_route="memory"`
- Gateway must be running before agent starts
