from __future__ import annotations

from typing import Any, Dict

from .external_dialogue_world_v113 import ExternalDialogueWorldV113, load_world_v113


class ExternalDialogueWorldV118(ExternalDialogueWorldV113):
    """
    V118 thin wrapper over the V113 ExternalDialogueWorld implementation.

    The V113 canonical JSONL format and deterministic offsets index are reused to keep
    the world read-only, deterministic, and audit-friendly.
    """

    pass


def load_world_v118(*, manifest_path: str) -> ExternalDialogueWorldV118:
    w = load_world_v113(manifest_path=str(manifest_path))
    # Cast the concrete instance to the V118 alias type.
    return ExternalDialogueWorldV118(canonical_jsonl_path=str(w.canonical_jsonl_path), manifest=dict(w.manifest))

