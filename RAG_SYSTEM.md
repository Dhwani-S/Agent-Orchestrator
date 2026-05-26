# RAG System Architecture (Session 7+)

## Overview

This document describes the Retrieval-Augmented Generation (RAG) system integrated into the cognitive agent in Session 7. The system combines FAISS vector search with keyword fallback to provide intelligent memory retrieval for multi-turn queries.

## Components

### 1. Vector Index (FAISS)

**Location**: `state/index.faiss`, `state/index.ids.json`

- **Index Type**: `IndexFlatIP` (Inner Product for cosine similarity)
- **Dimensions**: 768 (Gemini embedding model)
- **Distance Metric**: Cosine similarity (vectors are L2-normalized)
- **Similarity Threshold**: 0.3 (below this triggers keyword fallback)

**Storage**:
- `index.faiss`: Binary FAISS index
- `index.ids.json`: Mapping from FAISS row index to memory item IDs

### 2. Memory Items

**Location**: `state/memory.json`

Each item has:
```json
{
  "id": "abc123...",
  "kind": "fact|preference|scratchpad",
  "descriptor": "Human-readable summary",
  "keywords": ["search", "terms"],
  "value": {"structured": "data"},
  "embedding": [0.1, 0.2, ...],  // 768 floats or null
  "artifact_id": "art:...",        // optional
  "source": "corpus/file.md|web_search|user_query",
  "run_id": "session_uuid",
  "goal_id": "goal_uuid",
  "created_at": "ISO timestamp"
}
```

### 3. Corpus Indexing

**Corpus Location**: `sandbox/corpus/`

**Directory Structure**:
```
corpus/
  ├── ancient/          (11 files) — Ancient India
  ├── medieval/         (10 files) — Medieval period
  ├── colonial/         (10 files) — Colonial era
  ├── freedom/          (10 files) — Independence struggle
  ├── modern/           (10 files) — Modern India
  └── culture/          (4 files) — Culture & science
```

**Batch Indexing**: 
- 55 corpus files indexed with automatic chunking
- Chunk size: 400 words
- Chunk overlap: 80 words (for context continuity)
- Total indexed items: 109 chunks + 5 papers = 114 memory items

### 4. Hybrid Search (memory.read())

**Algorithm**:

1. **Vector Search (Primary)**:
   - Embed query using Gemini embedding model (task_type="retrieval_query")
   - L2-normalize query vector
   - Search FAISS index with cosine similarity
   - Return top-k (default 8) results with distance > 0.3

2. **Keyword Search (Fallback)**:
   - If vector search fails (embed error) or returns empty results
   - Tokenize query + recent history
   - Match against item keywords using set intersection
   - Score by overlap count
   - Return top-k results sorted by score

**Fallback Triggers**:
- Embedding API returns error (503, timeout)
- Vector search returns 0 results or all below threshold (0.3)
- Previous tool calls in history suggest keyword approach

## Integration Points

### Memory Module (memory.py)

- `Memory.read(query, history, kinds, top_k)` — Hybrid search entry point
- `Memory.remember(raw_text, source, run_id)` — LLM-classified write
- `Memory.add_fact(descriptor, value, keywords, source, run_id)` — Direct fact insertion
- `_try_embed(text, task_type)` — Embedding with error handling

### Decision Module (decision.py)

- `Rules 7-9` prevent decision loop by checking RECENT_HISTORY
- Rule 7: Don't re-call search_knowledge if it returned empty
- Rule 8: Use existing read_file results before calling more tools
- Rule 9: Never re-read the same file path

### Perception Module (perception.py)

- First iteration: Decomposes query into bounded goals
- Later iterations: Tracks goal satisfaction using history
- Artifact attachment for first unfinished goal (index into memory hits)

## Performance Tuning

### Embedding Retry Strategy

**File**: `llm_gateway/client.py`

```python
embed(text: str, *, task_type: str, max_retries: int = 3)
```

- **Retry Count**: 3 (reduced from 8 for faster fallback)
- **Delays**: [10s, 15s, 20s] between retries
- **Rate Limit**: Gemini 15 RPM via gateway
- **Fallback**: Keyword search if embed fails

### Artifact Truncation

**File**: `decision.py` → `_format_attached()`

- **Max Size**: 8,000 characters (reduced from 20,000 for rate limit safety)
- **Reason**: Rate-limited APIs (15 RPM Gemini) overwhelmed by large prompts
- **Result**: Query completion time reduced from ~20 min to ~4-6 min

## Corpus Content

The corpus covers Indian history and culture:

- **Ancient** (1500 BCE - 1000 CE): Vedic, Maurya, Gupta, etc.
- **Medieval** (1000-1800 CE): Islamic dynasties, Vijayanagara, etc.
- **Colonial** (1757-1947): British rule, exploitation, resistance
- **Freedom** (1857-1947): Independence struggle, key figures
- **Modern** (1947-present): Post-independence, technology, economy
- **Culture**: Mathematics, astronomy, arts, ISRO

## Example Query Flow

**User Query**: "How did ancient Indian mathematical innovations influence the modern world?"

**Perception**:
1. Decompose into goals:
   - Fetch ancient mathematical innovations
   - Extract modern mathematical applications
   - Synthesize influence chain

**Decision Loop** (4-6 iterations):
1. Call `search_knowledge("ancient Indian mathematics")` → hits
2. Call `read_file("corpus/ancient/mathematics.md")` → artifact
3. Call `search_knowledge("modern mathematics influence")` → hits
4. Synthesize answer from all results

**Memory Storage**:
- Facts about Aryabhata, decimal system, etc. added to memory
- Future queries on "mathematics" or "India" will retrieve these

## Known Limitations

1. **Embedding Latency**: Gemini at 15 RPM means ~4s per embed, so heavy queries take time
2. **Keyword Overlap**: Requires exact keyword match; similar terms may not match
3. **Chunk Boundary Issues**: Answer spans chunk boundaries risk missing content
4. **Update Frequency**: Corpus is static; new documents require manual indexing

## Future Improvements

- [ ] Hierarchical clustering for faster search (HNSW index)
- [ ] Cross-encoder re-ranking to improve result quality
- [ ] Parent chunk retrieval to avoid boundary issues
- [ ] Query expansion using LLM (e.g., synonyms, related terms)
- [ ] Incremental indexing for new documents
- [ ] Caching of common queries
- [ ] Multi-language support

## Configuration

See `config.py` for tunable constants:
- `EMBEDDING_DIMENSION = 768`
- `FAISS_TOP_K = 8`
- `MEMORY_CHUNK_SIZE = 400` (words)
- `MEMORY_CHUNK_OVERLAP = 80` (words)
- `KEYWORD_SEARCH_THRESHOLD = 0.3`
- `DECISION_ARTIFACT_MAX_CHARS = 8_000`
