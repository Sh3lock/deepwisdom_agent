# Example configuration for MCP integration
# Copy these commands to your PowerShell session before running main.py

# ============================================================
# Required: OpenAI API Configuration
# ============================================================
$env:OPENAI_API_KEY="sk-your-api-key-here"
$env:OPENAI_MODEL="gpt-4o-mini"  # Optional, defaults to gpt-4o-mini

# ============================================================
# Optional: MCP Server Configuration
# ============================================================

# Example 1: MCP Server via npx (requires npm package installed)
# Install first: npm install -g @modelcontextprotocol/server-everything
$env:MCP_SERVER_COMMAND="npx"
$env:MCP_SERVER_ARGS='["@modelcontextprotocol/server-everything"]'

# Example 2: Local Node.js MCP server
# $env:MCP_SERVER_COMMAND="node"
# $env:MCP_SERVER_ARGS='["C:\\path\\to\\your\\mcp-server.js"]'

# Example 3: Python MCP server
# $env:MCP_SERVER_COMMAND="python"
# $env:MCP_SERVER_ARGS='["C:\\path\\to\\your\\mcp_server.py"]'

# Example 4: With additional arguments
# $env:MCP_SERVER_COMMAND="node"
# $env:MCP_SERVER_ARGS='["server.js", "--port", "8080", "--verbose"]'

# ============================================================
# To disable MCP (use only local_search tool)
# ============================================================
# Remove-Item Env:MCP_SERVER_COMMAND
# Remove-Item Env:MCP_SERVER_ARGS

# ============================================================
# Verify configuration
# ============================================================
Write-Host "Current configuration:"
Write-Host "  OPENAI_API_KEY: $($env:OPENAI_API_KEY -ne $null)"
Write-Host "  OPENAI_MODEL: $env:OPENAI_MODEL"
Write-Host "  MCP_SERVER_COMMAND: $env:MCP_SERVER_COMMAND"
Write-Host "  MCP_SERVER_ARGS: $env:MCP_SERVER_ARGS"
