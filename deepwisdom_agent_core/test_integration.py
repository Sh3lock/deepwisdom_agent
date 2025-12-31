"""
Quick test script for Memory and MCP integration.
Run this to verify the basic functionality without full agent interaction.
"""

import os
import sys
from pathlib import Path

# Test 1: Memory Store
print("=" * 60)
print("Test 1: Memory Store")
print("=" * 60)

from memory import MemoryStore, MemoryEntry

# Create a test memory store
test_db = "test_memory.sqlite"
if Path(test_db).exists():
    Path(test_db).unlink()

store = MemoryStore(test_db)
print("✓ MemoryStore initialized")

# Test upsert
store.upsert("preference", "language", "Chinese")
store.upsert("profile", "role", "data scientist")
store.upsert("constraint", "tech_stack", "Python 3.11")
store.upsert("fact", "os", "Windows")
print("✓ Upserted 4 memories")

# Test retrieve
results = store.retrieve("Python data", top_k=3)
print(f"✓ Retrieved {len(results)} memories for 'Python data':")
for r in results:
    print(f"  - {r.memory_type}: {r.key} = {r.value}")

# Test format
formatted = store.format_for_prompt(results)
print(f"✓ Formatted for prompt:\n{formatted}")

# Cleanup
Path(test_db).unlink()
print("✓ Test database cleaned up\n")

# Test 2: MCP Client (basic structure check)
print("=" * 60)
print("Test 2: MCP Client Structure")
print("=" * 60)

from mcp_client import MCPClient, load_mcp_tools_sync

# Check if MCP is configured
mcp_command = os.getenv("MCP_SERVER_COMMAND")
if mcp_command:
    print(f"✓ MCP configured: {mcp_command}")
    print("  Attempting to load tools (this may take a moment)...")
    try:
        tools = load_mcp_tools_sync()
        print(f"✓ Loaded {len(tools)} MCP tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"✗ MCP loading failed: {e}")
else:
    print("○ MCP not configured (set MCP_SERVER_COMMAND to test)")
    print("  This is expected if you haven't set up an MCP server")

print()

# Test 3: Graph integration (import check)
print("=" * 60)
print("Test 3: Graph Integration")
print("=" * 60)

try:
    from graph import build_graph, SYSTEM_MESSAGE
    from langchain_openai import ChatOpenAI
    
    print("✓ Graph module imported successfully")
    print("✓ SYSTEM_MESSAGE defined")
    
    # Try building graph (without actually running it)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    graph = build_graph(llm, memory_store=store, extra_tools=[])
    print("✓ Graph built successfully with memory support")
    
except Exception as e:
    print(f"✗ Graph integration test failed: {e}")

print()

# Test 4: Memory extraction (mock test)
print("=" * 60)
print("Test 4: Memory Extraction Structure")
print("=" * 60)

from memory import extract_memories_with_llm

# Note: This would require actual LLM call, so we just check the function exists
print("✓ extract_memories_with_llm function available")
print("  (Skipping actual LLM call to save API usage)")

print()
print("=" * 60)
print("All basic tests completed!")
print("=" * 60)
print()
print("Ready to run: python main.py")
