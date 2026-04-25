"""
HuggingFace Space — Langflow custom component
Calls any public (or private-with-token) Gradio Space via gradio_client.

Install:
    pip install gradio_client huggingface_hub pillow
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file

try:
    from langflow.custom import Component
    from langflow.io import DataInput, IntInput, Output, SecretStrInput, StrInput
    from langflow.schema import Data
except ImportError:
    # Allows the file to be imported/tested outside a running Langflow instance.
    Component = object  # type: ignore[assignment,misc]
    StrInput = SecretStrInput = IntInput = DataInput = Output = lambda **kw: None  # type: ignore[assignment]

    class Data:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp3", ".wav", ".ogg", ".mp4", ".glb"}


def _is_file_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return Path(value).suffix.lower() in _FILE_EXTENSIONS


def _coerce_result(result: Any) -> dict[str, Any]:
    """Turn whatever gradio_client returns into a serialisable dict."""
    if result is None:
        return {"result": None}
    if isinstance(result, (str, int, float, bool)):
        return {"result": result}
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        items = []
        for item in result:
            items.append(item if isinstance(item, (str, int, float, bool, dict, list, type(None))) else str(item))
        return {"results": items}
    return {"result": str(result)}


# ── Component ─────────────────────────────────────────────────────────────────


class HuggingFaceSpaceComponent(Component):
    """Call any HuggingFace Gradio Space as a Langflow tool."""

    display_name = "HuggingFace Space"
    description = "Call any HuggingFace Gradio Space via gradio_client"
    icon = "HuggingFace"
    name = "HuggingFaceSpaceComponent"

    inputs = [
        StrInput(
            name="space_id",
            display_name="Space ID",
            info="HuggingFace Space slug, e.g. 'black-forest-labs/FLUX.1-schnell'",
            required=True,
        ),
        StrInput(
            name="input_text",
            display_name="Input Text / Prompt",
            info="Primary text input or prompt for the Space",
            required=False,
        ),
        StrInput(
            name="api_name",
            display_name="API Endpoint",
            info="Gradio API endpoint name (default: /predict). Check the Space's API page.",
            value="/predict",
            required=False,
        ),
        SecretStrInput(
            name="hf_token",
            display_name="HuggingFace Token (optional)",
            info="Required for private Spaces or to avoid rate limits",
            required=False,
        ),
        DataInput(
            name="input_image",
            display_name="Input Image / File (optional)",
            info="Pass a Data object whose 'file_path' key points to a local file",
            required=False,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout (seconds)",
            info="Maximum seconds to wait for the Space to respond",
            value=60,
            required=False,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="call_space"),
    ]

    # ── Core method ───────────────────────────────────────────────────────────

    def call_space(self) -> Data:
        space_id: str = self.space_id
        input_text: str = self.input_text or ""
        api_name: str = self.api_name or "/predict"
        hf_token: str | None = self.hf_token or os.getenv("HF_TOKEN") or None
        timeout: int = int(self.timeout or 60)
        input_image = self.input_image  # Data | None

        if not space_id:
            raise ValueError("space_id is required")

        try:
            client = Client(
                space_id,
                token=hf_token,
                verbose=False,
            )
        except Exception as exc:
            return Data(data={"error": f"Failed to connect to Space '{space_id}': {exc}"})

        # Build positional args list
        args: list[Any] = []

        if input_image is not None:
            img_path: str | None = None
            if isinstance(input_image, Data):
                img_path = input_image.data.get("file_path") or input_image.data.get("path")
            elif isinstance(input_image, dict):
                img_path = input_image.get("file_path") or input_image.get("path")

            if img_path and os.path.exists(img_path):
                args.append(handle_file(img_path))
            elif img_path:
                # Treat as URL
                args.append(handle_file(img_path))

        if input_text:
            args.append(input_text)

        try:
            raw = client.predict(*args, api_name=api_name)
        except Exception as exc:
            return Data(data={"error": f"Space call failed: {exc}", "space_id": space_id})

        result_data = _coerce_result(raw)
        result_data["space_id"] = space_id

        # If the result is a local file path, include it explicitly
        flat_result = result_data.get("result") or result_data.get("results", [None])[0]
        if _is_file_path(flat_result):
            result_data["file_path"] = flat_result

        return Data(data=result_data)
