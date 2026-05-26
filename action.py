from __future__ import annotations

from mcp import ClientSession
from schemas import ToolCall
from artifacts import ArtifactStore

ARTIFACT_THRESHOLD_BYTES = 4096  # 4 KB


async def execute(session: ClientSession, tool_call: ToolCall,
                  artifact_store: ArtifactStore) -> tuple[str, str | None]:
    """Execute tool call via MCP and handle artifact storage.
    
    - Validates that artifact handles are not passed as arguments
    - Calls tool via MCP with arguments
    - If result > 4KB, stores in artifact store and returns compact descriptor
    - If result <= 4KB, returns inline text
    
    Args:
        session: Active MCP client session
        tool_call: Tool name + arguments
        artifact_store: Artifact storage for large results
        
    Returns:
        Tuple of (descriptor_text, artifact_id_or_None)
        - descriptor_text: Result summary or error message (for decision layer)
        - artifact_id: "art:HASH" if result was stored, None otherwise
    """

    # Guard: block artifact handles passed as tool arguments
    for key, val in tool_call.arguments.items():
        if isinstance(val, str) and val.startswith("art:"):
            return (
                f"ERROR: '{key}' contains an artifact handle '{val}'. "
                f"Artifact handles are not file paths or URLs. "
                f"The artifact content is attached to your prompt by Perception — read it there.",
                None,
            )

    # Dispatch via MCP
    result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)

    # Collapse content blocks into a single text string
    text = "\n".join(
        block.text if hasattr(block, "text") else str(block)
        for block in result.content
    )

    # If payload is large, store as artifact and return a short descriptor
    raw_bytes = text.encode("utf-8")
    if len(raw_bytes) > ARTIFACT_THRESHOLD_BYTES:
        art_id = artifact_store.put(
            raw_bytes,
            content_type="text/plain",
            source=f"tool:{tool_call.name}",
            descriptor=f"{tool_call.name}({tool_call.arguments}) -> {len(raw_bytes)} bytes",
        )
        preview = text[:300].replace("\n", " ")
        return f"[artifact {art_id}, {len(raw_bytes)} bytes] preview: {preview}", art_id

    # Small payload — return directly
    return text, None
