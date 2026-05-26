"""Configuration constants for Agent Orchestrator Session 6+.

Centralized settings for LLM gateway, RAG system, and agent execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# ============================================================================
# LLM Gateway Configuration
# ============================================================================

GATEWAY_URL = os.getenv("LLM_GATEWAY_V7_URL", "http://localhost:8107")
"""Base URL for LLM Gateway V7 microservice."""

GATEWAY_TIMEOUT = 60.0
"""Request timeout for gateway in seconds."""

GATEWAY_MAX_RETRIES = 3
"""Maximum retry attempts for gateway requests."""

# ============================================================================
# LLM Model Configuration
# ============================================================================

DEFAULT_MODEL = "gemini-2.0-flash-001"
"""Default LLM model for general reasoning."""

EMBEDDING_MODEL = "gemini-embedding-001"
"""Embedding model for RAG indexing."""

EMBEDDING_DIMENSION = 768
"""Vector dimensionality for FAISS index."""

# ============================================================================
# RAG / Memory Configuration
# ============================================================================

FAISS_METRIC = "cosine"
"""Distance metric for vector search (IP = inner product = cosine with L2 norm)."""

FAISS_TOP_K = 8
"""Maximum memory items to retrieve per query."""

MEMORY_CHUNK_SIZE = 400
"""Chunk size in words for corpus text."""

MEMORY_CHUNK_OVERLAP = 80
"""Word overlap between successive chunks."""

KEYWORD_SEARCH_THRESHOLD = 0.3
"""Minimum FAISS distance score; below this falls back to keyword search."""

# ============================================================================
# Agent Execution Configuration
# ============================================================================

MAX_ITERATIONS = 15
"""Maximum iterations before stopping agent loop."""

DECISION_ARTIFACT_MAX_CHARS = 8_000
"""Maximum characters to include in decision prompt for artifacts (rate limit safety)."""

# ============================================================================
# MCP Server Configuration
# ============================================================================

MCP_SERVER_PATH = "mcp_server.py.py"
"""Path to MCP server module."""

SANDBOX_DIR = Path(__file__).parent / "sandbox"
"""Directory for sandboxed file operations."""

# ============================================================================
# API Usage Limits
# ============================================================================

TAVILY_MONTHLY_CAP = 950
"""Soft cap for Tavily API calls per month (actual limit 1000)."""

TAVILY_MAX_RESULTS = 5
"""Maximum search results per Tavily query."""

# ============================================================================
# Rate Limiting (via Gateway)
# ============================================================================

# Gateway handles provider-specific rate limits:
# - Gemini: 15 requests per minute (RPM)
# - Groq: 30 RPM
# - OpenAI: 30 RPM  
# - Cerebras: 30 RPM

GEMINI_RATE_LIMIT = 15
"""Gemini requests per minute via gateway."""

OTHER_RATE_LIMIT = 30
"""Rate limit for Groq, OpenAI, Cerebras via gateway."""
