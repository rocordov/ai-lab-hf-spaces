#!/usr/bin/env python3
"""
Quick smoke-test for HuggingFace Spaces integrations.

Usage:
    pip install gradio_client huggingface_hub
    HF_TOKEN=hf_... python3 scripts/test_spaces.py   # token optional for public spaces

Tests:
    1. FLUX.1-schnell  — text-to-image
    2. Whisper large v3 — speech-to-text (uses a public sample URL)
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
TIMEOUT = 90  # seconds; MCP itself uses 60s, give the raw client a bit more

FLUX_SPACE = "black-forest-labs/FLUX.1-schnell"
WHISPER_SPACE = "openai/whisper-large-v3"

# A ~3-second public LibriVox clip (public domain, no auth required)
SAMPLE_AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/1/1f/Dial_up_modem_noises.ogg"


# ── helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m  {msg}")


def fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m  {msg}")


# ── token check ───────────────────────────────────────────────────────────────

section("Environment")
if HF_TOKEN:
    ok(f"HF_TOKEN is set ({HF_TOKEN[:8]}...)")
    if HF_HUB_OK:
        try:
            api = HfApi(token=HF_TOKEN)
            user = api.whoami()
            ok(f"Authenticated as: {user['name']}")
        except Exception as exc:
            fail(f"Token validation failed: {exc}")
else:
    print("  ℹ  HF_TOKEN not set — using public (rate-limited) access")


# ── Test 1: FLUX.1-schnell ────────────────────────────────────────────────────

section(f"Test 1 — Image Generation  ({FLUX_SPACE})")
t0 = time.perf_counter()
try:
    client = Client(FLUX_SPACE, hf_token=HF_TOKEN, verbose=False)
    result = client.predict(
        prompt="a white cat sitting on a neon sign, synthwave aesthetic",
        seed=42,
        randomize_seed=False,
        width=512,
        height=512,
        num_inference_steps=4,
        api_name="/infer",
    )
    elapsed = time.perf_counter() - t0
    # result is typically (image_path, seed_used)
    img_path = result[0] if isinstance(result, (list, tuple)) else result
    ok(f"Image generated in {elapsed:.1f}s  →  {img_path}")
except Exception as exc:
    elapsed = time.perf_counter() - t0
    fail(f"FLUX call failed after {elapsed:.1f}s: {exc}")


# ── Test 2: Whisper ───────────────────────────────────────────────────────────

section(f"Test 2 — Speech-to-Text  ({WHISPER_SPACE})")
t0 = time.perf_counter()
try:
    client = Client(WHISPER_SPACE, hf_token=HF_TOKEN, verbose=False)
    result = client.predict(
        inputs=handle_file(SAMPLE_AUDIO_URL),
        task="transcribe",
        api_name="/predict",
    )
    elapsed = time.perf_counter() - t0
    text = result if isinstance(result, str) else str(result)
    ok(f"Transcription in {elapsed:.1f}s  →  {text[:120]!r}")
except Exception as exc:
    elapsed = time.perf_counter() - t0
    fail(f"Whisper call failed after {elapsed:.1f}s: {exc}")


# ── Summary ───────────────────────────────────────────────────────────────────

section("Done")
print("  Add HF_TOKEN to your shell profile to avoid rate-limit errors on public Spaces.")
print("  Example:  export HF_TOKEN=hf_your_token_here")
print()
