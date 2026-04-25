#!/usr/bin/env python3
"""
Smoke tests for HuggingFace Spaces integrations.

Two phases:
  1. Connectivity — introspect each Space's Gradio API (no token required).
     Confirms the Space exists and the gradio_client can reach it.

  2. Inference — actually call the Space. Requires HF_TOKEN because free-tier
     ZeroGPU Spaces (FLUX, Whisper) rate-limit anonymous requests aggressively.
     NOTE: FLUX.1-schnell and hf-audio/whisper-large-v3 run on ZeroGPU and can
     still return transient server-side errors even with a token; retry if needed.

Usage:
    pip install gradio_client huggingface_hub
    python3 scripts/test_spaces.py                    # connectivity only
    HF_TOKEN=hf_... python3 scripts/test_spaces.py    # full suite
"""

from __future__ import annotations

import os
import sys
import time

# ── dependency check ──────────────────────────────────────────────────────────
try:
    from gradio_client import Client, handle_file
except ImportError:
    sys.exit("gradio_client not found.  Run: pip install gradio_client")

try:
    from huggingface_hub import HfApi
    HF_HUB_OK = True
except ImportError:
    HF_HUB_OK = False

# ── config ────────────────────────────────────────────────────────────────────
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or None

# NOTE: use hf-audio/whisper-large-v3 (community Space), not openai/whisper-large-v3
# (that slug is the model repo, not a public Space).
FLUX_SPACE       = "black-forest-labs/FLUX.1-schnell"   # ZeroGPU — may be transient
WHISPER_SPACE    = "hf-audio/whisper-large-v3"           # ZeroGPU — may be transient
CHATTERBOX_SPACE = "ResembleAI/Chatterbox"               # TTS, stable
BG_REMOVE_SPACE  = "not-lain/background-removal"         # Vision, stable

SPACES_TO_PROBE = [
    FLUX_SPACE,
    WHISPER_SPACE,
    CHATTERBOX_SPACE,
    BG_REMOVE_SPACE,
]

SAMPLE_AUDIO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/1/1f/Dial_up_modem_noises.ogg"
)
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"
SKIP = "\033[33m–\033[0m"


def section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print("─" * 64)


# ── environment ───────────────────────────────────────────────────────────────
section("Environment")
if HF_TOKEN:
    print(f"  {PASS}  HF_TOKEN set ({HF_TOKEN[:8]}...)")
    if HF_HUB_OK:
        try:
            user = HfApi(token=HF_TOKEN).whoami()
            print(f"  {PASS}  Authenticated as: {user['name']}")
        except Exception as exc:
            print(f"  {FAIL}  Token validation failed: {exc}")
else:
    print(f"  {SKIP}  HF_TOKEN not set — inference tests will be skipped")
    print("       Free token at: https://huggingface.co/settings/tokens")


# ── connectivity tests ────────────────────────────────────────────────────────
section("Connectivity — API introspection (no token needed)")

connectivity_ok: dict[str, bool] = {}
clients: dict[str, Client] = {}

for space_id in SPACES_TO_PROBE:
    t0 = time.perf_counter()
    try:
        c = Client(space_id, token=HF_TOKEN, verbose=False)
        endpoints = [ep for ep in c.endpoints if ep]
        elapsed = time.perf_counter() - t0
        print(f"  {PASS}  {space_id:<50}  {elapsed:.1f}s  endpoints={endpoints}")
        connectivity_ok[space_id] = True
        clients[space_id] = c
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        short = str(exc)[:80]
        print(f"  {FAIL}  {space_id:<50}  {elapsed:.1f}s  {short}")
        connectivity_ok[space_id] = False


# ── inference tests ───────────────────────────────────────────────────────────
section("Inference tests (require HF_TOKEN)")

if not HF_TOKEN:
    print(f"  {SKIP}  Skipped — set HF_TOKEN=hf_... to enable")
else:
    # ── 1. FLUX.1-schnell (ZeroGPU — transient failures possible) ────────────
    label = f"FLUX ({FLUX_SPACE})"
    if connectivity_ok.get(FLUX_SPACE):
        t0 = time.perf_counter()
        try:
            c = clients[FLUX_SPACE]
            result = c.predict(
                prompt="a white cat on a neon sign, synthwave aesthetic",
                seed=42,
                randomize_seed=False,
                width=512,
                height=512,
                num_inference_steps=4,
                api_name="/infer",
            )
            elapsed = time.perf_counter() - t0
            img = result[0] if isinstance(result, (list, tuple)) else result
            img_path = img.get("path") if isinstance(img, dict) else img
            print(f"  {PASS}  {label}  {elapsed:.1f}s  →  {img_path}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            msg = str(exc)[:100]
            symbol = WARN if "RuntimeError" in msg or "AppError" in msg else FAIL
            print(f"  {symbol}  {label}  {elapsed:.1f}s  {msg}")
            if symbol == WARN:
                print(f"       ↳ ZeroGPU transient error — retry usually succeeds")
    else:
        print(f"  {SKIP}  {label} (connectivity failed)")

    # ── 2. Whisper (ZeroGPU — transient failures possible) ───────────────────
    label = f"Whisper ({WHISPER_SPACE})"
    if connectivity_ok.get(WHISPER_SPACE):
        t0 = time.perf_counter()
        try:
            c = clients[WHISPER_SPACE]
            result = c.predict(
                inputs=handle_file(SAMPLE_AUDIO_URL),
                task="transcribe",
                api_name="/transcribe",
            )
            elapsed = time.perf_counter() - t0
            text = result if isinstance(result, str) else str(result)
            print(f"  {PASS}  {label}  {elapsed:.1f}s  →  {text[:100]!r}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            msg = str(exc)[:100]
            symbol = WARN if "AppError" in msg or "RuntimeError" in msg else FAIL
            print(f"  {symbol}  {label}  {elapsed:.1f}s  {msg}")
            if symbol == WARN:
                print(f"       ↳ ZeroGPU transient error — retry usually succeeds")
    else:
        print(f"  {SKIP}  {label} (connectivity failed)")

    # ── 3. Chatterbox TTS ─────────────────────────────────────────────────────
    label = f"Chatterbox TTS ({CHATTERBOX_SPACE})"
    if connectivity_ok.get(CHATTERBOX_SPACE):
        t0 = time.perf_counter()
        try:
            c = clients[CHATTERBOX_SPACE]
            result = c.predict(
                "Hello from Royce's AI Lab.",
                api_name="/generate_tts_audio",
            )
            elapsed = time.perf_counter() - t0
            audio = result[0] if isinstance(result, (list, tuple)) else result
            audio_path = audio.get("path") if isinstance(audio, dict) else audio
            print(f"  {PASS}  {label}  {elapsed:.1f}s  →  {audio_path}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  {FAIL}  {label}  {elapsed:.1f}s  {str(exc)[:100]}")
    else:
        print(f"  {SKIP}  {label} (connectivity failed)")

    # ── 4. Background removal ─────────────────────────────────────────────────
    label = f"Background removal ({BG_REMOVE_SPACE})"
    if connectivity_ok.get(BG_REMOVE_SPACE):
        t0 = time.perf_counter()
        try:
            c = clients[BG_REMOVE_SPACE]
            result = c.predict(
                handle_file(SAMPLE_IMAGE_URL),
                api_name="/image",
            )
            elapsed = time.perf_counter() - t0
            # /image returns a tuple (before, after) via Imageslider
            img = result[0] if isinstance(result, (list, tuple)) else result
            img_path = img.get("path") if isinstance(img, dict) else img
            print(f"  {PASS}  {label}  {elapsed:.1f}s  →  {img_path}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  {FAIL}  {label}  {elapsed:.1f}s  {str(exc)[:100]}")
    else:
        print(f"  {SKIP}  {label} (connectivity failed)")


# ── summary ───────────────────────────────────────────────────────────────────
section("Summary")
ok_count = sum(connectivity_ok.values())
total = len(connectivity_ok)
print(f"  Connectivity: {ok_count}/{total} Spaces reachable")
if not HF_TOKEN:
    print("  Inference:    skipped (set HF_TOKEN=hf_... to enable)")
print()
print("  Note: FLUX and Whisper use ZeroGPU (serverless GPU) — inference can return")
print("  transient RuntimeError on cold starts. Re-running the test usually resolves it.")
print()
