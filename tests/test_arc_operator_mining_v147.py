import importlib.util
from pathlib import Path


def _load_miner_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "arc_mine_operator_templates_v147.py"
    spec = importlib.util.spec_from_file_location("arc_mine_operator_templates_v147", str(path))
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_close_operator_seq_typed_suffixes() -> None:
    m = _load_miner_module()
    close = m._close_operator_seq

    # patch -> commit_patch
    assert close(("crop_bbox",), max_len=5) == ("bbox_by_color", "crop_bbox", "commit_patch")

    # bbox -> crop_bbox + commit_patch
    assert close(("bbox_by_color",), max_len=5) == ("bbox_by_color", "crop_bbox", "commit_patch")

    # obj -> obj_bbox + crop_bbox + commit_patch
    assert close(("select_obj",), max_len=5) == ("cc4", "select_obj", "obj_bbox", "crop_bbox", "commit_patch")

    # objset -> select_obj + obj_bbox + crop_bbox + commit_patch
    assert close(("cc4",), max_len=5) == ("cc4", "select_obj", "obj_bbox", "crop_bbox", "commit_patch")


def test_close_operator_seq_respects_max_len() -> None:
    m = _load_miner_module()
    close = m._close_operator_seq
    assert close(("cc4",), max_len=4) is None


def test_close_operator_seq_passthrough_and_unknown() -> None:
    m = _load_miner_module()
    close = m._close_operator_seq

    # Already grid-producing.
    assert close(("reflect_h",), max_len=5) == ("reflect_h",)

    # Unknown op id cannot be closed.
    assert close(("unknown_op_id_zzz",), max_len=5) is None
