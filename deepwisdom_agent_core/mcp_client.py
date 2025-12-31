"""
MCP (Model Context Protocol) client integration.
Connects to MCP servers via stdio and wraps their tools as LangChain tools.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    MCP client that connects to a server via stdio and provides tools.
    """
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.server_params: Optional[StdioServerParameters] = None
        self._read_stream = None
        self._write_stream = None
        self._tools_cache: List[Dict[str, Any]] = []
    
    @classmethod
    def from_env(cls, env_prefix: str = "MCP") -> Optional["MCPClient"]:
        """
        Create MCP client from environment variables.
        Expected env vars:
        - MCP_SERVER_COMMAND: e.g., "npx"
        - MCP_SERVER_ARGS: JSON array of args, e.g., '["@modelcontextprotocol/server-everything"]'
        """
        command = os.getenv(f"{env_prefix}_SERVER_COMMAND")
        args_str = os.getenv(f"{env_prefix}_SERVER_ARGS")
        
        if not command:
            return None
        
        args = []
        if args_str:
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = [args_str]
        
        client = cls()
        client.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )
        return client
    
    async def connect(self):
        """Connect to the MCP server and initialize session."""
        if not self.server_params:
            raise ValueError("Server parameters not set")
        
        read, write = await stdio_client(self.server_params)
        self._read_stream = read
        self._write_stream = write
        self.session = ClientSession(read, write)
        
        await self.session.initialize()
        
        # List available tools
        result = await self.session.list_tools()
        self._tools_cache = result.tools if hasattr(result, 'tools') else []
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            # MCP session cleanup
            self.session = None
        self._read_stream = None
        self._write_stream = None
    
    def get_tools_list(self) -> List[Dict[str, Any]]:
        """Get list of available tools from MCP server."""
        return self._tools_cache
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        result = await self.session.call_tool(name, arguments)
        return result
    
    def wrap_as_langchain_tools(self) -> List[StructuredTool]:
        """
        Wrap MCP tools as LangChain StructuredTool objects.
        Each tool is given a function that calls back to this client.
        """
        langchain_tools = []
        
        for tool_info in self._tools_cache:
            tool_name = tool_info.get("name", "unknown")
            tool_desc = tool_info.get("description", "")
            input_schema = tool_info.get("inputSchema", {})
            
            # Create async wrapper for this specific tool
            async def _async_wrapper(name=tool_name, **kwargs):
                return await self.call_tool(name, kwargs)
            
            # Create sync wrapper using asyncio
            def _sync_wrapper(name=tool_name, **kwargs):
                return asyncio.run(_async_wrapper(name=name, **kwargs))
            
            # Parse input schema to get args_schema
            # For simplicity, we'll use a basic dict type
            from pydantic import BaseModel, create_model, Field
            
            # Build pydantic model from JSON schema
            fields = {}
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            for prop_name, prop_info in properties.items():
                prop_type = str  # Default to str
                prop_desc = prop_info.get("description", "")
                is_required = prop_name in required
                
                if is_required:
                    fields[prop_name] = (prop_type, Field(description=prop_desc))
                else:
                    fields[prop_name] = (Optional[prop_type], Field(default=None, description=prop_desc))
            
            # Create dynamic Pydantic model
            if fields:
                ArgsModel = create_model(f"{tool_name}Args", **fields)
            else:
                ArgsModel = None
            
            # Create LangChain tool
            lc_tool = StructuredTool(
                name=tool_name,
                description=tool_desc or f"MCP tool: {tool_name}",
                func=lambda name=tool_name, **kw: _sync_wrapper(name=name, **kw),
                args_schema=ArgsModel,
            )
            
            langchain_tools.append(lc_tool)
        
        return langchain_tools


@asynccontextmanager
async def get_mcp_client(env_prefix: str = "MCP"):
    """
    Context manager to create and manage MCP client lifecycle.
    Usage:
        async with get_mcp_client() as client:
            tools = client.wrap_as_langchain_tools()
    """
    client = MCPClient.from_env(env_prefix)
    if client is None:
        yield None
        return
    
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()


def load_mcp_tools_sync(env_prefix: str = "MCP") -> List[StructuredTool]:
    """
    Synchronous helper to load MCP tools.
    Returns empty list if MCP not configured.
    """
    async def _load():
        async with get_mcp_client(env_prefix) as client:
            if client is None:
                return []
            return client.wrap_as_langchain_tools()
    
    try:
        return asyncio.run(_load())
    except Exception as e:
        # If MCP fails to load, just return empty tools list
        print(f"[MCP] Failed to load: {e}")
        return []
