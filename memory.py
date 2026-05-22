from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from schemas import MemoryItem

MEMORY_PATH = Path(__file__).parent / "state" / "memory.json"

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

    #tokenizer
    def _tokenize(self, text: str) -> set[str]:
        tokens = set(text.lower().split())
        return tokens - STOPWORDS
    
    # reads (no LLM)
    def read(self, query: str, history: list[dict], kinds: list[str] | None = None, top_k: int = 8) -> list[MemoryItem]:
        self._ensure_loaded()
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
        )
        self._items.append(item)
        self._save()
        return item
    
    def record_outcome(self, tool_call, result_text: str, artifact_id: str | None,
                       run_id: str, goal_id: str | None = None):
        """No-LLM write. Records what a tool returned."""
        self._ensure_loaded()
        keywords = [tool_call.name.lower()]
        keywords += list(self._tokenize(" ".join(str(v) for v in tool_call.arguments.values())))
        keywords += list(self._tokenize(result_text[:200]))[:5]

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
        )
        self._items.append(item)
        self._save()
        return item