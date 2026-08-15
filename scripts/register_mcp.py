"""Script to generate and register MCP client configuration snippets for Claude Desktop, Open-WebUI, and AI agent frameworks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPO_PATH = Path("/Docs/Programming/GitHub/os-autofix-engine").resolve()
DEFAULT_PYTHON_EXEC = sys.executable or "python3"


def generate_mcp_configs(
    repo_path: Path = DEFAULT_REPO_PATH, python_exec: str = DEFAULT_PYTHON_EXEC
) -> dict[str, Any]:
    """Generate standardized MCP client configuration dictionaries."""
    main_script = str(repo_path / "main.py")

    claude_desktop_config = {
        "mcpServers": {
            "os-autofix": {
                "command": python_exec,
                "args": [main_script, "mcp"],
                "env": {
                    "OLLAMA_BASE_URL": "http://10.0.0.25:11434/v1",
                    "OPEN_WEBUI_BASE_URL": "https://ai.is27.duckdns.org/api",
                },
            }
        }
    }

    open_webui_tool_config = {
        "name": "os-autofix-mcp",
        "description": "Incus VM/Container OS troubleshooting and fault injection harness",
        "command": f"{python_exec} {main_script} mcp",
        "transport": "stdio",
        "tools": [
            "list_scenarios",
            "create_sandbox",
            "inject_fault",
            "exec_command",
            "verify_fix",
            "revert_sandbox",
            "destroy_sandbox",
            "run_benchmark",
        ],
    }

    return {
        "claude_desktop": claude_desktop_config,
        "open_webui": open_webui_tool_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MCP configuration snippets for Claude Desktop and agent frameworks."
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=DEFAULT_REPO_PATH,
        help="Path to os-autofix-engine repository",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=DEFAULT_PYTHON_EXEC,
        help="Python executable path",
    )
    parser.add_argument(
        "--write-claude",
        action="store_true",
        help="Attempt writing to local Claude Desktop config file",
    )
    args = parser.parse_args()

    configs = generate_mcp_configs(repo_path=args.repo_path, python_exec=args.python)

    print("=" * 60)
    print(" Claude Desktop MCP Configuration (~/.config/Claude/claude_desktop_config.json)")
    print("=" * 60)
    print(json.dumps(configs["claude_desktop"], indent=2))
    print()

    print("=" * 60)
    print(" Open-WebUI & Agent Tool Configuration")
    print("=" * 60)
    print(json.dumps(configs["open_webui"], indent=2))
    print()

    if args.write_claude:
        claude_path = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
        try:
            claude_path.parent.mkdir(parents=True, exist_ok=True)
            existing_data: dict[str, Any] = {}
            if claude_path.exists():
                try:
                    existing_data = json.loads(claude_path.read_text(encoding="utf-8"))
                except Exception:
                    existing_data = {}

            existing_data.setdefault("mcpServers", {})["os-autofix"] = configs["claude_desktop"][
                "mcpServers"
            ]["os-autofix"]
            claude_path.write_text(json.dumps(existing_data, indent=2), encoding="utf-8")
            print(f"Successfully wrote os-autofix MCP configuration to {claude_path}")
        except Exception as e:
            print(f"Failed to write Claude config to {claude_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
