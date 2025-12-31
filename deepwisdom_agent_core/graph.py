from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from search_tool import local_search
from memory import MemoryStore

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a local document search tool. "
    "Call local_search when you need to check facts, recall specifics, or answer "
    "based on local docs. If you are uncertain, use the tool. "
    "When you use results, cite sources by filename in brackets like [source]."
)
SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _trim_messages(messages: list, memory_context: str = "") -> list:
    """
    Trim to last 6 messages. If no system message, inject one.
    Memory context is injected into system message but not counted in the 6-message limit.
    """
    recent = messages[-6:]
    if not any(m.type == "system" for m in recent):
        # Inject system message with memory context
        system_content = SYSTEM_PROMPT
        if memory_context:
            system_content = f"{memory_context}\n\n{SYSTEM_PROMPT}"
        recent = [SystemMessage(content=system_content)] + recent
        while len(recent) > 6:
            recent.pop(1)
    else:
        # Update existing system message with memory context
        if memory_context:
            for i, m in enumerate(recent):
                if m.type == "system":
                    recent[i] = SystemMessage(content=f"{memory_context}\n\n{m.content}")
                    break
    return recent


def build_graph(llm, memory_store: Optional[MemoryStore] = None, extra_tools: list = None):
    """
    Build the agent graph with memory integration and dynamic tool support.
    
    Args:
        llm: The language model
        memory_store: Optional MemoryStore for long-term memory
        extra_tools: Optional list of additional tools (e.g., from MCP)
    """
    # Combine local_search with any extra tools
    all_tools = [local_search]
    if extra_tools:
        all_tools.extend(extra_tools)
    
    llm_with_tools = llm.bind_tools(all_tools)

    def call_model(state: AgentState):
        messages = state["messages"]
        
        # Retrieve relevant memories if available
        memory_context = ""
        if memory_store and messages:
            # Get last user message for memory retrieval
            user_msg = next((m.content for m in reversed(messages) if m.type == "human"), "")
            if user_msg:
                memories = memory_store.retrieve(user_msg, top_k=5)
                memory_context = memory_store.format_for_prompt(memories)
        
        # Trim messages with memory context
        trimmed = _trim_messages(messages, memory_context)
        response = llm_with_tools.invoke(trimmed)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(all_tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()