#!/usr/bin/env bash
# Sets up mcp-hfspace in Claude Desktop by merging the hfspace server block
# into the existing claude_desktop_config.json without clobbering other MCP servers.

set -euo pipefail

CONFIG_DIR="$HOME/Library/Application Support/Claude"
CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNIPPET="$SCRIPT_DIR/claude_desktop_config.json"

# ── 1. Verify Node / npx ──────────────────────────────────────────────────────
echo "Checking dependencies..."
if ! command -v node &>/dev/null; then
  echo "node not found — installing via Homebrew..."
  if ! command -v brew &>/dev/null; then
    echo "ERROR: Homebrew is required to auto-install node. Install it from https://brew.sh then re-run."
    exit 1
  fi
  brew install node
fi

if ! command -v npx &>/dev/null; then
  echo "ERROR: npx not found even though node is installed. Check your PATH."
  exit 1
fi

echo "  node $(node --version)  npx $(npx --version)  OK"

# ── 2. Ensure Claude config directory exists ──────────────────────────────────
mkdir -p "$CONFIG_DIR"

# ── 3. Merge hfspace block into existing config ───────────────────────────────
echo "Merging hfspace MCP server into $CONFIG_FILE ..."

python3 - <<'PYEOF'
import json, sys, os

config_file = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
snippet_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_desktop_config.json")

# Load existing config or start fresh
if os.path.exists(config_file):
    with open(config_file) as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: {config_file} is not valid JSON — creating backup and starting fresh.")
            import shutil, time
            shutil.copy(config_file, config_file + f".bak.{int(time.time())}")
            existing = {}
else:
    existing = {}

# Load the hfspace snippet
with open(snippet_file) as f:
    snippet = json.load(f)

# Merge mcpServers
existing.setdefault("mcpServers", {})
existing["mcpServers"].update(snippet["mcpServers"])

# Write back
with open(config_file, "w") as f:
    json.dump(existing, f, indent=2)

print(f"  hfspace server added. Total MCP servers: {list(existing['mcpServers'].keys())}")
PYEOF

# ── 4. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "Done! Please restart Claude Desktop to pick up the new MCP server."
echo ""
echo "  Registered Spaces (from spaces_list.json):"
python3 -c "
import json, os
with open(os.path.join(os.path.dirname('$SNIPPET'), 'spaces_list.json')) as f:
    data = json.load(f)
for s in data['spaces']:
    print(f\"    {s['id']:55s}  [{s['category']}]\")
"
