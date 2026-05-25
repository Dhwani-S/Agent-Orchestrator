from __future__ import annotations

import json
import uuid
import numpy as np
import faiss
from datetime import datetime
from pathlib import Path

from schemas import MemoryItem

MEMORY_PATH = Path(__file__).parent / "state" / "memory.json"
FAISS_INDEX_PATH = Path(__file__).parent/"state"/"index.faiss"
FAISS_IDS_PATH = Path(__file__).parent/"state"/"index.ids.json"
EMBED_DIM = 768

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
             "for", "of", "and", "or", "it", "its", "this", "that", "with", "from",
             "by", "as", "be", "has", "had", "do", "does", "did", "will", "would",
             "can", "could", "should", "may", "might", "i", "me", "my", "we", "you",
             "he", "she", "they", "them", "what", "which", "who", "when", "where",
             "how", "not", "no", "but", "if", "then", "so", "up", "out", "about"}


class Memory:
    def __init__(self):
        self._items: list[MemoryItem] = []
        self._loaded = False

    # persistence
    def _ensure_loaded(self):
        if self._loaded:
            return
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if MEMORY_PATH.exists():
            raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            self._items = [MemoryItem(**item) for item in raw]
        self._loaded = True

    def _save(self):
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = [item.model_dump(mode="json") for item in self._items]
        MEMORY_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _try_embed(self, text: str, task_type: str = "retrieval_document"):
        try:
            from llm_gateway.client import LLM
            return LLM().embed(text, task_type=task_type)
        except Exception as e:
            print(f"[memory] embed failed: {e}")
            return None
        
    def _load_faiss(self):
        if FAISS_INDEX_PATH.exists() and FAISS_IDS_PATH.exists():
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            ids = json.loads(FAISS_IDS_PATH .read_text(encoding="utf-8"))
            return index, ids
        index = faiss.IndexFlatIP(EMBED_DIM)
        return index, []
    
    def _append_to_faiss(self, item_id: str, embedding: list[float]):
        index, ids = self._load_faiss()
        vec = np.array([embedding], dtype=np.float32)
        vec = vec/np.linalg.norm(vec) # L2 normalize -> cosine similarity
        index.add(vec)
        ids.append(item_id)
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        FAISS_IDS_PATH.write_text(json.dumps(ids), encoding="utf-8")

    def _persist_item(self, item: MemoryItem):
        self._items.append(item)
        self._save()
        if item.embedding is not None:
            self._append_to_faiss(item.id, item.embedding)
        return item
        
    #tokenizer
    def _tokenize(self, text: str) -> set[str]:
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = set(text.split())
        return tokens - STOPWORDS
    
    # reads (no LLM)
    def read(self, query: str, history: list[dict], kinds: list[str] | None = None, top_k: int = 8) -> list[MemoryItem]:
        self._ensure_loaded()

        # vector path (first try)
        query_vec = self._try_embed(query, task_type="retrieval_query")
        if query_vec is not None:
            index, ids = self._load_faiss()
            if index.ntotal > 0:
                vec = np.array([query_vec], dtype=np.float32)
                vec = vec/np.linalg.norm(vec)
                distances, indices = index.search(vec, min(top_k, index.ntotal))

                id_to_item = {item.id: item for item in self._items}
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if idx == -1 or dist < 0.3:
                        continue
                    item_id = ids[idx]
                    item = id_to_item.get(item_id)
                    if item and (not kinds or item.kind in kinds):
                        results.append(item)
                if results:
                    return results
                
        # keyword fallback
        query_tokens = self._tokenize(query)
        
        # pull tokens from recent history
        for event in history[-5:]:
            if "result_descriptor" in event:
                query_tokens |= self._tokenize(event["result_descriptor"])
            if "text" in event:
                query_tokens |= self._tokenize(event["text"])

        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items:
            if kinds and item.kind not in kinds:
                continue
            item_tokens = set(item.keywords) | self._tokenize(item.descriptor)
            overlap = len(query_tokens & item_tokens)
            if overlap > 0:
                scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
    
    def filter(self, kinds: list[str] | None = None, goal_id: str | None = None, recent: int | None = None) -> list[MemoryItem]:
        self._ensure_loaded()
        results = self._items
        if kinds:
            results = [i for i in results if i.kind in kinds]
        if goal_id:
            results = [i for i in results if i.goal_id == goal_id]
        if recent:
            results = results[-recent:]
        return results
    
    def add_fact(self, descriptor: str, *, value: dict, keywords: list[str], source: str, run_id: str, goal_id: str | None = None) -> MemoryItem:
        self._ensure_loaded()
        embedding = self._try_embed(descriptor, task_type="retrieval_document")
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind="fact",
            keywords=[k.lower() for k in keywords],
            descriptor=descriptor,
            value=value,
            source=source,
            embedding=embedding,
            run_id=run_id,
            goal_id=goal_id,
        )
        return self._persist_item(item)
    
    # writes
    def remember(self, raw_text: str, source: str, run_id: str, goal_id: str | None = None):
        """LLM-classified write. Sends text to gateway to extract kind, keywords, value."""
        self._ensure_loaded()
        from llm_gateway.client import LLM

        classify_schema = {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["fact", "preference", "scratchpad"]},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "descriptor": {"type": "string"},
                "value": {"type": "object"},
            },
            "required": ["kind", "keywords", "descriptor", "value"],
        }

        system = (
            "You are a memory classifier. Given a user statement, extract:\n"
            "- kind: 'fact' (observable truth with entity+attribute+value), "
            "'preference' (user-stated liking/disliking), or 'scratchpad' (temporary note).\n"
            "- keywords: list of lowercase search terms (entity names, dates, key nouns).\n"
            "- descriptor: one short human-readable sentence summarizing the content.\n"
            "- value: structured dict with the extracted data (e.g. {\"entity\": \"Mom\", \"attribute\": \"birthday\", \"value\": \"2026-05-15\"}).\n"
            "Output valid JSON only."
        )

        llm = LLM()
        resp = llm.chat(
            prompt=raw_text,
            system=system,
            provider="v",
            temperature=0.3,
            response_format={"type": "json_schema", "schema": classify_schema, "name": "memory_classify"},
        )

        parsed = resp.get("parsed") or json.loads(resp["text"])
        embedding = None
        if parsed["kind"] != "scratchpad":
            embedding = self._try_embed(parsed["descriptor"])

        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind=parsed["kind"],
            keywords=parsed["keywords"],
            descriptor=parsed["descriptor"],
            value=parsed["value"],
            source=source,
            run_id=run_id,
            goal_id=goal_id,
            confidence=0.9,
            embedding=embedding,
        )

        return self._persist_item(item)
    
    def record_outcome(self, tool_call, result_text: str, artifact_id: str | None,
                       run_id: str, goal_id: str | None = None):
        """No-LLM write. Records what a tool returned."""
        self._ensure_loaded()
        keywords = [tool_call.name.lower()]
        keywords += list(self._tokenize(" ".join(str(v) for v in tool_call.arguments.values())))
        keywords += list(self._tokenize(result_text[:200]))[:5]

        embedding = self._try_embed(result_text[:200])

        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind="tool_outcome",
            keywords=keywords[:20],
            descriptor=result_text[:200],
            value={"tool": tool_call.name, "arguments": tool_call.arguments, "result_preview": result_text[:500]},
            artifact_id=artifact_id,
            source=f"tool:{tool_call.name}",
            run_id=run_id,
            goal_id=goal_id,
            embedding=embedding,
        )
        return self._persist_item(item)