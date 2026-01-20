from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from atos_core.world_manifest_resolver_v119 import resolve_world_canonical_jsonl_v119


class TestWorldManifestResolverV119(unittest.TestCase):
    def test_resolves_v113_style_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            canon = root / "dialogue_history_canonical_v113.jsonl"
            canon.write_text('{"global_turn_index":0,"role":"user","text":"hi"}\n', encoding="utf-8")
            manifest = root / "dialogue_history_canonical_v113_manifest.json"
            manifest.write_text(
                json.dumps({"schema_version": 113, "paths": {"canonical_jsonl": "dialogue_history_canonical_v113.jsonl"}}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            res = resolve_world_canonical_jsonl_v119(world_manifest_path=str(manifest), default_rel="dialogue_history_canonical_v113.jsonl")
            self.assertTrue(res.get("ok"), res)
            self.assertTrue(Path(str(res.get("canonical_jsonl_resolved"))).exists())

    def test_resolves_v111_style_manifest_in_manifests_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "dialogue_history_canonical").mkdir(parents=True, exist_ok=True)
            canon = root / "dialogue_history_canonical" / "dialogue_history_canonical_v111.jsonl"
            canon.write_text('{"global_turn_index":0,"role":"user","text":"hi"}\n', encoding="utf-8")
            (root / "manifests").mkdir(parents=True, exist_ok=True)
            manifest = root / "manifests" / "world_manifest_v111.json"
            manifest.write_text(
                json.dumps({"schema_version": 111, "paths": {"canonical_jsonl": "dialogue_history_canonical/dialogue_history_canonical_v111.jsonl"}}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            res = resolve_world_canonical_jsonl_v119(world_manifest_path=str(manifest), default_rel="dialogue_history_canonical_v113.jsonl")
            self.assertTrue(res.get("ok"), res)
            self.assertEqual(Path(str(res.get("canonical_jsonl_resolved"))), canon.resolve())


if __name__ == "__main__":
    unittest.main()

