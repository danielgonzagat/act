import json
import os
import tempfile
import unittest
from pathlib import Path

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.external_world_v122 import ew_load_and_verify


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


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, objs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(canonical_json_dumps(o))
            f.write("\n")


class TestExternalWorldV122Build(unittest.TestCase):
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
                    "meta": {"title": "t", "model": "x", "version": None},
                    "source": "chatgpt_export_v122",
                },
                {
                    "turn_index": 1,
                    "conversation_id": "c1",
                    "message_id": "m2",
                    "timestamp": "2026-01-16T00:00:01Z",
                    "role": "assistant",
                    "text": "hi",
                    "meta": {"title": "t", "model": "x", "version": None},
                    "source": "chatgpt_export_v122",
                },
            ],
        )
        _write_text(doc_plain, "Projeto ACT\nLinha 2\n")
        _write_jsonl(
            doc_chunks,
            [
                {
                    "doc": "Projeto_ACT",
                    "chunk_id": "Projeto_ACT:deadbeefdeadbeef:000000",
                    "heading": "",
                    "text": "Projeto ACT\nLinha 2\n",
                    "offset_start": 0,
                    "offset_end": 18,
                    "sha256_text": sha256_hex("Projeto ACT\nLinha 2\n".encode("utf-8")),
                }
            ],
        )

        manifest = {
            "schema_version": 122,
            "kind": "external_world_unified_v122",
            "inputs": {"conversations_json_path": "/dev/null", "conversations_json_sha256": "", "projeto_act_rtf_path": "/dev/null", "projeto_act_rtf_sha256": "", "paths_tried": []},
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

    def test_load_verify_search_fetch(self) -> None:
        root = self._make_world()
        world = ew_load_and_verify(manifest_path=str(root / "manifest_v122.json"))
        hits = world.search(query="hello", limit=3, source_filter="dialogue_history", roles=["user"])
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].hit_id.startswith("dlg:"))
        ft = world.fetch(hit_id=hits[0].hit_id, max_chars=100)
        self.assertEqual(ft.source, "dialogue_history")
        self.assertEqual(ft.text, "hello world")

        hits2 = world.search(query="Projeto", limit=3, source_filter="engineering_doc")
        self.assertEqual(len(hits2), 1)
        ft2 = world.fetch(hit_id=hits2[0].hit_id, max_chars=10)
        self.assertTrue(ft2.truncated)

    def test_manifest_tamper_detected(self) -> None:
        root = self._make_world()
        mp = root / "manifest_v122.json"
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["sha256"]["engineering_doc_plain_v122_txt"] = "0" * 64
        body = dict(m)
        body.pop("manifest_sig", None)
        m["manifest_sig"] = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
        mp.write_text(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            ew_load_and_verify(manifest_path=str(mp))
        self.assertEqual(str(ctx.exception), "external_world_manifest_mismatch_v122")


if __name__ == "__main__":
    unittest.main()

