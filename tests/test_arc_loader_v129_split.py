import json
import tempfile
import unittest
from pathlib import Path


def _write_min_arc_task(path: Path) -> None:
    task = {"train": [{"input": [[0]], "output": [[0]]}], "test": [{"input": [[0]], "output": [[0]]}]}
    path.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")


class TestArcLoaderV129Split(unittest.TestCase):
    def test_arc_loader_v129_split_dir_resolution(self) -> None:
        from atos_core.arc_loader_v129 import write_arc_canonical_jsonl_v129

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arc_root"
            (root / "training").mkdir(parents=True)
            (root / "evaluation").mkdir(parents=True)
            _write_min_arc_task(root / "training" / "00000000.json")
            _write_min_arc_task(root / "evaluation" / "11111111.json")

            out_jsonl = Path(tmp) / "out.jsonl"
            manifest = write_arc_canonical_jsonl_v129(arc_root=str(root), out_jsonl_path=str(out_jsonl), split="training")
            self.assertEqual(manifest.get("split"), "training")
            self.assertTrue(str(manifest.get("tasks_root", "")).endswith("training"))
            self.assertEqual(int(manifest.get("tasks_total") or 0), 1)

            # If split is not recognized, loader must fail to avoid mixing.
            out_jsonl2 = Path(tmp) / "out2.jsonl"
            with self.assertRaises(ValueError) as ctx:
                write_arc_canonical_jsonl_v129(arc_root=str(root), out_jsonl_path=str(out_jsonl2), split="sample")
            self.assertIn("arc_split_required", str(ctx.exception))
