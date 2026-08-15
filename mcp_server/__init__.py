"""Model Context Protocol (MCP) server package for os-autofix-engine."""

from mcp_server.server import (
    ACTIVE_SANDBOXES,
    mcp,
    run_mcp_server,
    tool_create_sandbox,
    tool_destroy_sandbox,
    tool_exec_command,
    tool_inject_fault,
    tool_list_scenarios,
    tool_revert_sandbox,
    tool_run_benchmark,
    tool_verify_fix,
)

__all__ = [
    "mcp",
    "run_mcp_server",
    "ACTIVE_SANDBOXES",
    "tool_list_scenarios",
    "tool_create_sandbox",
    "tool_inject_fault",
    "tool_exec_command",
    "tool_verify_fix",
    "tool_revert_sandbox",
    "tool_destroy_sandbox",
    "tool_run_benchmark",
]
