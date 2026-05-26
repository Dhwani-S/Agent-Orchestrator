from __future__ import annotations

import hashlib
import json
from pathlib import Path

from schemas import Artifact

ARTIFACTS_DIR = Path(__file__).parent / "state"/"artifacts"

class ArtifactStore:
    """File-backed artifact (blob) storage with metadata.
    
    Stores arbitrary binary content with metadata (content_type, source, descriptor).
    Uses SHA256 hash for deduplication and immutability.
    """
    def __init__(self, base: Path = ARTIFACTS_DIR):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)

    def put(self, blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
        """Store a blob and return artifact ID.
        
        Automatically deduplicates via SHA256 hashing. Stores binary in .bin file
        and metadata in .json file.
        
        Args:
            blob: Binary content to store
            content_type: MIME type (e.g., "text/markdown", "text/html")
            source: Origin (e.g., "web_search", "read_file", "corpus/ancient/...")
            descriptor: Human-readable summary
            
        Returns:
            Artifact ID in format "art:HASH" (16-char hex)
        """
        self.base.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(blob).hexdigest()[:16]
        art_id = f"art:{sha}"
        bin_path = self.base / f"{sha}.bin"
        meta_path = self.base / f"{sha}.json"
        if not bin_path.exists():
            bin_path.write_bytes(blob)
            meta = Artifact(
                id=art_id,
                content_type=content_type,
                size_bytes=len(blob),
                source=source,
                descriptor=descriptor,
            )
            meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        return art_id
    
    def get_bytes(self, artifact_id: str) -> bytes:
        """Retrieve raw bytes for an artifact.
        
        Args:
            artifact_id: ID from put() or "art:HASH"
            
        Returns:
            Binary blob
        """
        sha = artifact_id.removeprefix("art:")
        return (self.base / f"{sha}.bin").read_bytes()
    
    def get_meta(self, artifact_id: str) -> Artifact:
        """Retrieve metadata for an artifact.
        
        Args:
            artifact_id: ID from put() or "art:HASH"
            
        Returns:
            Artifact metadata object
        """
        sha = artifact_id.removeprefix("art:")
        raw = json.loads((self.base / f"{sha}.json").read_text(encoding="utf-8"))
        return Artifact(**raw)
    
    def exists(self, artifact_id: str) -> bool:
        """Check if artifact exists.
        
        Args:
            artifact_id: ID from put() or "art:HASH"
            
        Returns:
            True if both binary and metadata exist
        """
        sha = artifact_id.removeprefix("art:")
        return (self.base / f"{sha}.bin").exists()