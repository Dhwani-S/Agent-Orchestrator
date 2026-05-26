# Agent Orchestrator: System Architecture

## High-Level Overview

The Agent Orchestrator is a multi-layered cognitive agent that decomposes complex queries into bounded goals, executes them iteratively, and synthesizes answers. Session 6 built the core 4-role reasoning engine; Session 7 added vector search retrieval (RAG/FAISS) for context-aware answering.

```
User Query
    ↓
┌─────────────────────────────────────┐
│  Perception Layer (perception.py)  │
│  - Decompose query into goals      │
│  - Track goal satisfaction         │
│  - Manage artifact attachment      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Decision Loop (agent6.py)       │
│  MAX_ITERATIONS = 15                │
│                                     │
│  For each goal:                     │
│    ├─ Search Memory (FAISS+keyword)│
│    ├─ Decision Layer chooses tool  │
│    └─ Action Layer executes        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Final Answer Synthesis            │
│  - Combine all goal results        │
│  - Return to user                  │
└─────────────────────────────────────┘
```

## Layered Architecture

### Layer 1: Perception

**File**: `perception.py`

**Responsibility**: Understand and decompose the query

**Key Functions**:
- `observe(query, hits, history, prior_goals, run_id)` → `Observation`
  - First call: decompose query into 3-5 bounded goals
  - Later calls: track which goals are satisfied based on history
  - Manage artifact attachment for context

**Goal Types**:
- `fetch`: Retrieve data (web search, read file, search memory)
- `extract`: Analyze/extract information from context
- `synthesize`: Compare, decide, or summarize multiple sources

**Key Rules**:
- Goals are ordered logically (prerequisites first)
- Once a goal is marked done, it stays done
- Do not reorder or drop goals between iterations
- Confirm each goal has satisfactory answer before marking done

### Layer 2: Decision

**File**: `decision.py`

**Responsibility**: Reason about the current goal and decide whether to call a tool or synthesize

**Key Functions**:
- `next_step(goal, hits, attached, history, mcp_tools, user_query)` → `DecisionOutput`
  - Receives current goal + context (memory hits, attached artifacts, history)
  - Uses LLM with DECISION_SYSTEM prompt (Rules 1-9)
  - Returns either tool call or final answer

**Decision Rules** (DECISION_SYSTEM prompt):
1. Output is EXACTLY ONE: tool call OR answer
2. Distinguish artifact handles (art:*) from file paths
3. Answers must be substantive (3+ sentences or list)
4. If URL is mentioned, use fetch_url with exact URL
5. Self-check: does answer address the goal?
6. If tool failed, try different approach
7. NO REPEATS: if search_knowledge returned empty, don't call it again
8. USE WHAT YOU HAVE: if read_file content in history, extract from it
9. NEVER RE-READ: if same path already read, don't read again

**Context Provided**:
- `MEMORY HITS`: Top-k vector search results
- `ATTACHED ARTIFACTS`: Raw bytes of relevant documents (max 8KB each)
- `RECENT HISTORY`: Last 10 iterations of tool calls and answers

### Layer 3: Action

**File**: `action.py`

**Responsibility**: Execute the chosen tool

**Available Tools** (from MCP server):
- `search_knowledge` — Vector search in memory
- `read_file` — Read files from sandbox/
- `index_document` — Add document to memory
- `web_search` — Tavily or DuckDuckGo search
- `fetch_url` — Crawl URL with Chromium
- `get_time` — Current datetime
- `currency_convert` — Exchange rates
- `create_file`, `update_file`, `edit_file` — Sandbox file writes

**MCP Configuration**:
- Transport: stdio (in-process)
- Server: `mcp_server.py.py`
- Schema-driven validation

### Layer 4: Memory & Retrieval

**File**: `memory.py`

**Responsibility**: Store and retrieve facts contextually

**Retrieval Strategy** (hybrid):
1. **Vector Search (Primary)**:
   - Embed query using Gemini (768 dims, cosine)
   - FAISS IndexFlatIP for inner product (normalized = cosine)
   - Return top-8 hits with similarity > 0.3

2. **Keyword Search (Fallback)**:
   - If embed fails or vector search empty
   - Tokenize query + recent history
   - Match against item keywords
   - Score by overlap

**Storage**:
- `state/memory.json`: All items (facts, preferences, scratchpad)
- `state/index.faiss`: FAISS binary index
- `state/index.ids.json`: Row → item ID mapping

**Item Types**:
- `fact`: Observable truth (entity + attribute + value)
- `preference`: User-stated likings/dislikings
- `scratchpad`: Temporary notes (not vectorized)

## External Components

### LLM Gateway V7

**Location**: `llm_gatewayV7/` (FastAPI microservice)

**Responsibility**: Multi-provider LLM and embedding routing

**Providers**:
- LLM: Gemini, Groq, OpenAI, Cerebras
- Embeddings: Gemini (primary), Ollama (fallback)

**Rate Limits** (via gateway):
- Gemini: 15 RPM (queries + embeddings)
- Groq/OpenAI/Cerebras: 30 RPM

**Features**:
- Response caching (Redis)
- SQLite persistence
- Provider-specific rate limiting
- Automatic retry with exponential backoff

**Connection**:
- Runs on `localhost:8107`
- HTTP REST API
- Called by `llm_gateway/client.py` (async wrapper)

### MCP Server (Tool Host)

**Location**: `mcp_server.py.py`

**Responsibility**: Expose tools to agent via MCP protocol

**18 Tools**:
- Memory: search_knowledge, index_document
- File I/O: read_file, list_dir, create_file, update_file, edit_file
- Web: web_search, fetch_url
- Utility: get_time, currency_convert, web_search

**Sandboxing**:
- All file ops restricted to `sandbox/`
- Path escape protection
- Usage tracking (Tavily soft cap at 950/mo)

## Data Flow Example: Query "Who is Claude Shannon?"

```
1. User Query
   └─→ "Who is Claude Shannon?"

2. Perception Layer
   └─→ Decompose into goals:
       - fetch: "Search for Claude Shannon biography"
       - extract: "List key achievements"
       - synthesize: "Create summary"

3. Iteration 1
   ├─ Memory search for "Claude Shannon" → hits (papers, docs)
   ├─ Decision: "Call web_search for biography"
   └─ Action: web_search("Claude Shannon") → Wikipedia snippet

4. Iteration 2
   ├─ Memory search with new content
   ├─ Decision: "Call read_file on papers" (fetch attachment)
   └─ Action: read_file("papers/information_theory.md")

5. Iteration 3
   ├─ Memory search for "Shannon achievements"
   ├─ Decision: "Extract key achievements from history"
   └─ Answer: "Claude Shannon was a mathematician who..."

6. Iteration 4
   ├─ Perception: All goals marked done
   ├─ Decision: "Synthesize final answer"
   └─ Answer: Comprehensive summary returned

7. Output
   └─→ Detailed bio with achievements, impact, legacy
```

## Configuration & Tuning

**Key Files**:
- `config.py` — Centralized constants (768 dims, 8 top_k, etc.)
- `.env` — API keys (GEMINI_API_KEY, TAVILY_API_KEY, etc.)
- `llm_gatewayV7/config.yaml` — Provider credentials

**Tunable Parameters**:
- `MAX_ITERATIONS` — Iteration limit (default 15)
- `DECISION_ARTIFACT_MAX_CHARS` — Context window size (default 8KB)
- `FAISS_TOP_K` — Memory results per search (default 8)
- `MEMORY_CHUNK_SIZE` — Document chunk size in words (default 400)
- `GATEWAY_URL` — LLM Gateway endpoint

## Performance Characteristics

**Latency Breakdown** (typical query):
- Perception: ~2s (LLM call)
- Per iteration: ~3-5s average
  - Memory search: 0.5s (vector embed via Gateway)
  - Decision: 2s (LLM reasoning)
  - Action: 1-3s (tool execution)
- Total for 4-6 iterations: 15-30 seconds

**Memory Usage**:
- FAISS index: ~300 MB (109 items × 768 dims × 4 bytes)
- JSON cache: ~5 MB
- Process RAM: ~500 MB

**Network**:
- Gateway HTTP requests: ~5-10 per iteration
- Rate limited by Gemini 15 RPM

## Testing & Validation

**Test Suite** (Session 7):
- Query A-H: 8 base queries validating core functionality
- Query C1-C5: 5 custom corpus queries testing RAG
- All tests PASS

**Coverage**:
- Perception goal decomposition: ✓
- Memory vector search: ✓
- Keyword fallback: ✓
- Decision loop prevention: ✓
- Artifact handling: ✓
- Multi-turn conversation: ✓

## Deployment Notes

**Requirements**:
- Python 3.12+
- Pydantic v2
- MCP SDK v1.27.1
- FAISS CPU
- FastAPI (for Gateway)

**Startup**:
```bash
# Terminal 1: LLM Gateway
cd llm_gatewayV7 && python main.py

# Terminal 2: Agent
python agent6.py
```

**Logs**:
- Queries: `logs/YYYYMMDD_HHMMSS_slug.log`
- Gateway: stdout (FastAPI)

## Future Roadmap

- [ ] Session 8: Multi-turn conversation management
- [ ] Session 9: Fine-tuning decision rules
- [ ] Session 10: Recursive sub-queries
- [ ] Persistent conversation state (DB)
- [ ] Web UI dashboard
- [ ] Analytics on query success rates
