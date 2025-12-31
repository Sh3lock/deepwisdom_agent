import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from graph import build_graph, SYSTEM_MESSAGE
from memory import MemoryStore, extract_memories_with_llm
from mcp_client import load_mcp_tools_sync


def run():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    
    # Initialize memory store
    memory_store = MemoryStore("memory.sqlite")
    print("[Memory] Initialized SQLite memory store")
    
    # Load MCP tools if configured
    mcp_tools = load_mcp_tools_sync()
    if mcp_tools:
        print(f"[MCP] Loaded {len(mcp_tools)} tools from MCP server")
        for tool in mcp_tools:
            print(f"  - {tool.name}: {tool.description}")
    else:
        print("[MCP] No MCP server configured (set MCP_SERVER_COMMAND to enable)")
    
    # Build graph with memory and MCP tools
    app = build_graph(llm, memory_store=memory_store, extra_tools=mcp_tools)

    messages = [SYSTEM_MESSAGE]
    conversation_buffer = []  # Buffer for memory extraction
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        # Add to conversation buffer for memory extraction
        conversation_buffer.append(f"User: {user_input}")

        state = {"messages": messages + [HumanMessage(content=user_input)]}
        result = app.invoke(state)
        messages = result["messages"]

        reply = next((m.content for m in reversed(messages) if m.type == "ai"), "")
        print(f"Agent> {reply}")
        
        # Add agent response to buffer
        conversation_buffer.append(f"Agent: {reply}")
        
        # Periodic memory extraction (every 3 turns)
        if len(conversation_buffer) >= 6:
            _extract_and_store_memories(llm, memory_store, conversation_buffer)
            conversation_buffer.clear()


def _extract_and_store_memories(llm, memory_store: MemoryStore, conversation_buffer: list):
    """Extract memories from conversation and store them."""
    conversation_text = "\n".join(conversation_buffer[-12:])  # Last 6 exchanges
    
    try:
        entries = extract_memories_with_llm(llm, conversation_text)
        if entries:
            memory_store.upsert_many(entries)
            print(f"[Memory] Extracted and stored {len(entries)} memories")
    except Exception as e:
        print(f"[Memory] Extraction failed: {e}")


if __name__ == "__main__":
    run()