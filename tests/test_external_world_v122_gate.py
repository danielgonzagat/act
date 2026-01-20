import json
import tempfile
import unittest
from pathlib import Path

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.external_world_gate_v122 import (
    EXTERNAL_WORLD_ACTION_SEARCH_V122,
    compute_external_world_chain_hash_v122,
    external_world_access_v122,
    verify_external_world_event_sig_chain_v122,
)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_jsonl(path: Path, objs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(canonical_json_dumps(o))
            f.write("\n")


class TestExternalWorldV122Gate(unittest.TestCase):
    def _make_world(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        root = Path(td.name)
        dialogue = root / "dialogue_history_canonical_v122.jsonl"
        doc_plain = root / "engineering_doc_plain_v122.txt"
        doc_chunks = root / "engineering_doc_chunks_v122.jsonl"
        _write_jsonl(
            dialogue,
            [
                {
                    "turn_index": 0,
                    "conversation_id": "c1",
                    "message_id": "m1",
                    "timestamp": "2026-01-16T00:00:00Z",
                    "role": "user",
                    "text": "hello world",
                    "meta": {},
                    "source": "chatgpt_export_v122",
                }
            ],
        )
        doc_plain.write_text("Projeto ACT\n", encoding="utf-8")
        _write_jsonl(
            doc_chunks,
            [
                {
                    "doc": "Projeto_ACT",
                    "chunk_id": "Projeto_ACT:deadbeefdeadbeef:000000",
                    "heading": "",
                    "text": "Projeto ACT\n",
                    "offset_start": 0,
                    "offset_end": 11,
                    "sha256_text": sha256_hex("Projeto ACT\n".encode("utf-8")),
                }
            ],
        )
        manifest = {
            "schema_version": 122,
            "kind": "external_world_unified_v122",
            "inputs": {},
            "paths": {
                "dialogue_history_canonical_jsonl": str(dialogue.name),
                "engineering_doc_plain_txt": str(doc_plain.name),
                "engineering_doc_chunks_jsonl": str(doc_chunks.name),
            },
            "sha256": {
                "dialogue_history_canonical_v122_jsonl": _sha256_file(dialogue),
                "engineering_doc_plain_v122_txt": _sha256_file(doc_plain),
                "engineering_doc_chunks_v122_jsonl": _sha256_file(doc_chunks),
            },
            "counts": {},
            "command_line": {"argv": ["test"]},
        }
        body = dict(manifest)
        manifest["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
        (root / "manifest_v122.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return root

    def test_deny_by_default(self) -> None:
        root = self._make_world()
        with self.assertRaises(ValueError) as ctx:
            external_world_access_v122(
                allowed=False,
                manifest_path=str(root / "manifest_v122.json"),
                action=EXTERNAL_WORLD_ACTION_SEARCH_V122,
                reason_code="progress_blocked",
                args={"query": "hello", "limit": 1, "source_filter": "dialogue_history"},
                seed=0,
                turn_index=0,
                prev_event_sig="",
            )
        self.assertEqual(str(ctx.exception), "external_world_access_not_allowed")

    def test_invalid_reason_code(self) -> None:
        root = self._make_world()
        with self.assertRaises(ValueError) as ctx:
            external_world_access_v122(
                allowed=True,
                manifest_path=str(root / "manifest_v122.json"),
                action=EXTERNAL_WORLD_ACTION_SEARCH_V122,
                reason_code="invalid_reason_code_x",
                args={"query": "hello", "limit": 1, "source_filter": "dialogue_history"},
                seed=0,
                turn_index=0,
                prev_event_sig="",
            )
        self.assertEqual(str(ctx.exception), "invalid_reason_code")

    def test_allow_search_and_chain(self) -> None:
        root = self._make_world()
        events, evidences, summary = external_world_access_v122(
            allowed=True,
            manifest_path=str(root / "manifest_v122.json"),
            action=EXTERNAL_WORLD_ACTION_SEARCH_V122,
            reason_code="progress_blocked",
            args={"query": "hello", "limit": 1, "source_filter": "dialogue_history", "roles": ["user"]},
            seed=0,
            turn_index=0,
            prev_event_sig="",
        )
        self.assertEqual(len(events), 1)
        ok, reason, _ = verify_external_world_event_sig_chain_v122(events)
        self.assertTrue(ok, msg=str(reason))
        ch = compute_external_world_chain_hash_v122(events)
        self.assertTrue(isinstance(ch, str) and len(ch) > 10)
        self.assertEqual(len(evidences), 1)
        self.assertTrue(isinstance(summary, dict))


if __name__ == "__main__":
    unittest.main()

