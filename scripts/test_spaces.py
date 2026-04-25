#!/usr/bin/env python3
"""
Smoke tests for HuggingFace Spaces integrations.

Connectivity tests (no token needed) — verify we can reach each Space and
introspect its API. These always run.

Inference tests — actually call the Space. These require HF_TOKEN because
free-tier Spaces rate-limit anonymous users aggressively.

Usage:
    pip install gradio_client huggingface_hub
    python3 scripts/test_spaces.py              # connectivity only
    HF_TOKEN=hf_... python3 scripts/test_spaces.py  # full tests
"""

from __future__ import annotations

import os
import sys
import time

# ── dependency check ──────────────────────────────────────────────────────────
try:
    from gradio_client import Client, handle_file
except ImportError:
    sys.exit("gradio_client not found. Run: pip install gradio_client")

try:
    from huggingface_hub import HfApi
    HF_HUB_OK = True
except ImportError:
    HF_HUB_OK = False

# ── config ────────────────────────────────────────────────────────────────────
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or None

# NOTE: use hf-audio/whisper-large-v3 (community Space), not openai/whisper-large-v3
# (that slug is the model repo, not a public Space).
FLUX_SPACE    = "black-forest-labs/FLUX.1-schnell"
WHISPER_SPACE = "hf-audio/whisper-large-v3"

SAMPLE_AUDIO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/1/1f/Dial_up_modem_noises.ogg"
)

SPACES_TO_PROBE = [
    FLUX_SPACE,
    WHISPER_SPACE,
    "facebook/musicgen-small",
    "depth-anything/Depth-Anything-V2-Small",
]

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m–\033[0m"


def section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print("─" * 64)


# ── token / auth check ────────────────────────────────────────────────────────
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
    print("       Set HF_TOKEN to run the full suite (free token at hf.co/settings/tokens)")


# ── connectivity tests ────────────────────────────────────────────────────────
section("Connectivity — API introspection (no token needed)")

connectivity_ok: dict[str, bool] = {}
for space_id in SPACES_TO_PROBE:
    t0 = time.perf_counter()
    try:
        c = Client(space_id, token=HF_TOKEN, verbose=False)
        endpoints = [ep for ep in c.endpoints if ep]
        elapsed = time.perf_counter() - t0
        print(f"  {PASS}  {space_id:<52}  {elapsed:.1f}s  endpoints={endpoints}")
        connectivity_ok[space_id] = True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        short = str(exc)[:80]
        print(f"  {FAIL}  {space_id:<52}  {elapsed:.1f}s  {short}")
        connectivity_ok[space_id] = False


# ── inference tests (token required) ─────────────────────────────────────────
section("Inference tests (require HF_TOKEN)")

if not HF_TOKEN:
    print(f"  {SKIP}  Skipped — set HF_TOKEN to enable")
else:
    # Test 1: FLUX.1-schnell
    if connectivity_ok.get(FLUX_SPACE):
        t0 = time.perf_counter()
        try:
            c = Client(FLUX_SPACE, token=HF_TOKEN, verbose=False)
            result = c.predict(
                prompt="a white cat sitting on a neon sign, synthwave",
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
            print(f"  {PASS}  FLUX image  {elapsed:.1f}s  →  {img_path}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  {FAIL}  FLUX failed after {elapsed:.1f}s: {exc}")
    else:
        print(f"  {SKIP}  FLUX (connectivity failed above)")

    # Test 2: Whisper
    if connectivity_ok.get(WHISPER_SPACE):
        t0 = time.perf_counter()
        try:
            c = Client(WHISPER_SPACE, token=HF_TOKEN, verbose=False)
            result = c.predict(
                inputs=handle_file(SAMPLE_AUDIO_URL),
                task="transcribe",
                api_name="/transcribe",
            )
            elapsed = time.perf_counter() - t0
            text = result if isinstance(result, str) else str(result)
            print(f"  {PASS}  Whisper ASR  {elapsed:.1f}s  →  {text[:100]!r}")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  {FAIL}  Whisper failed after {elapsed:.1f}s: {exc}")
    else:
        print(f"  {SKIP}  Whisper (connectivity failed above)")


# ── summary ───────────────────────────────────────────────────────────────────
section("Summary")
ok_count = sum(connectivity_ok.values())
total = len(connectivity_ok)
print(f"  Connectivity: {ok_count}/{total} Spaces reachable")
if not HF_TOKEN:
    print("  Inference: skipped (set HF_TOKEN=hf_... to enable)")
print()
