# AI Lab — HuggingFace Spaces Integration

Two complementary ways to wire HuggingFace Gradio Spaces into Royce's AI Lab:

| Path | What it does |
|---|---|
| **MCP (Claude Desktop)** | Registers Spaces as native tools inside Claude Desktop via `mcp-hfspace` |
| **Langflow component** | Adds a reusable node that routes any task to any Space via `gradio_client` |

---

## Prerequisites

- macOS (setup script targets `~/Library/Application Support/Claude/`)
- Node.js ≥ 18 and `npx` (auto-installed by `setup_mcp.sh` via Homebrew if missing)
- Python ≥ 3.10
- Langflow ≥ 1.0 (for the component path)
- Optional: a HuggingFace token for private Spaces / higher rate limits

```bash
pip install -r requirements.txt
```

---

## Path 1 — Claude Desktop (MCP)

`mcp-hfspace` wraps each Space's Gradio API as a tool that Claude can call natively, with streaming progress and file outputs saved locally.

> **60-second timeout caveat:** `mcp-hfspace` enforces a 60 s timeout per tool call. Image generation on cold Spaces can exceed this. If Claude reports a timeout, re-run the prompt — warm Spaces respond in 5–15 s.

### Quick setup

```bash
bash mcp/setup_mcp.sh
# → Restart Claude Desktop when prompted
```

The script:
1. Checks that `node` / `npx` are installed (installs via Homebrew if not)
2. Reads your existing `~/Library/Application Support/Claude/claude_desktop_config.json`
3. Merges the `hfspace` server block — **does not overwrite** other MCP servers
4. Prompts you to restart Claude Desktop

### Manual setup

If you prefer to patch the config yourself, copy the block from `mcp/claude_desktop_config.json` and add it under `"mcpServers"` in your Claude Desktop config.

### Registered Spaces (default)

| Space | Category |
|---|---|
| `black-forest-labs/FLUX.1-schnell` | Image generation |
| `openai/whisper-large-v3` | Speech-to-text |
| `stabilityai/stable-diffusion-3-5-large` | Image generation |
| `facebook/musicgen-small` | Music generation |
| `depth-anything/Depth-Anything-V2-Small` | Depth estimation |
| `hysts/TRELLIS` | 3-D generation |

To add more Spaces, append them to the `args` array in `mcp/claude_desktop_config.json` and re-run `setup_mcp.sh`.  
The full curated list lives in `mcp/spaces_list.json`.

### Example Claude prompts

```
Generate an image of a cyberpunk Tokyo skyline at night using FLUX.
```
```
Transcribe this audio file: /path/to/recording.mp3
```
```
Generate 10 seconds of lo-fi hip-hop music with a rainy day vibe.
```

---

## Path 2 — Langflow Custom Component

`langflow/hf_spaces_component.py` is a self-contained Langflow component that calls any Gradio Space.

### Installation

1. Copy `langflow/hf_spaces_component.py` to your Langflow custom-components folder  
   (default: `~/.langflow/custom_components/`)
2. Restart Langflow — the **HuggingFace Space** node will appear in the sidebar

### Node inputs

| Input | Type | Description |
|---|---|---|
| `space_id` | string | e.g. `black-forest-labs/FLUX.1-schnell` |
| `input_text` | string | Prompt or text passed to the Space |
| `api_name` | string | Gradio API endpoint (default `/predict`) |
| `hf_token` | secret | HF token for private Spaces |
| `input_image` | Data | File path via a Data object's `file_path` key |
| `timeout` | int | Seconds before the call times out (default 60) |

### Output

A `Data` object with fields:

```json
{
  "space_id": "black-forest-labs/FLUX.1-schnell",
  "result": "/tmp/gradio/abc123/image.webp",
  "file_path": "/tmp/gradio/abc123/image.webp"
}
```

### Loading the example flow

Import `langflow/hf_spaces_flow_example.json` via **Flows → Import** in the Langflow UI.  
It wires **Text Input → FLUX.1-schnell → Text Output** as a minimal end-to-end demo.

### Finding the correct `api_name`

Every Gradio Space exposes its API at `https://huggingface.co/spaces/<space_id>?view=api`.  
The endpoint name (e.g. `/infer`, `/predict`, `/run`) varies by Space — check there first.

---

## Running the smoke tests

```bash
# Optional: set your HF token
export HF_TOKEN=hf_your_token_here

python3 scripts/test_spaces.py
```

Expected output:

```
────────────────────────────────────────────────────────────
  Test 1 — Image Generation  (black-forest-labs/FLUX.1-schnell)
────────────────────────────────────────────────────────────
  ✓  Image generated in 8.3s  →  /var/folders/.../image.webp

────────────────────────────────────────────────────────────
  Test 2 — Speech-to-Text  (openai/whisper-large-v3)
────────────────────────────────────────────────────────────
  ✓  Transcription in 12.1s  →  'DIAL-UP MODEM CONNECTING...'
```

---

## Repository layout

```
ai-lab-hf-spaces/
├── README.md
├── requirements.txt
├── mcp/
│   ├── claude_desktop_config.json   # MCP snippet — merge into Claude Desktop config
│   ├── setup_mcp.sh                 # Auto-installer
│   └── spaces_list.json             # Curated Space catalogue
├── langflow/
│   ├── hf_spaces_component.py       # Drop-in Langflow custom component
│   └── hf_spaces_flow_example.json  # Example flow (import into Langflow UI)
└── scripts/
    └── test_spaces.py               # Smoke tests for both integrations
```

---

## Extending

**Add a new Space to MCP:**
```json
// mcp/claude_desktop_config.json → args array
"your-org/your-space"
```
Then `bash mcp/setup_mcp.sh` and restart Claude Desktop.

**Add a new Space to Langflow:**  
Just change the `space_id` field on the node — no code changes needed.

**Private Spaces:**  
Set `HF_TOKEN` in your environment (MCP picks it up automatically; Langflow reads it from the node's `hf_token` field).

---

## License

MIT
