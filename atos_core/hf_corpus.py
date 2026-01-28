from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple


HF_KIND = Literal["dataset", "model"]


@dataclass(frozen=True)
class HFResolvedSource:
    source_id: str
    kind: HF_KIND
    dataset_id: str
    dataset_ids: Tuple[str, ...] = ()


def _normalize_hf_id(value: str) -> str:
    v = (value or "").strip()
    if not v:
        raise ValueError("Empty Hugging Face source.")

    v = re.sub(r"^https?://huggingface\.co/", "", v)
    v = re.sub(r"^datasets/", "", v)
    v = re.sub(r"^models/", "", v)
    v = v.split("#", 1)[0].split("?", 1)[0].strip("/")
    if v.count("/") != 1:
        raise ValueError(f"Expected 'owner/repo' or HF URL, got: {value!r}")
    return v


def resolve_hf_source(source: str) -> HFResolvedSource:
    source_id = _normalize_hf_id(source)

    # Keep optional deps out of core import-time.
    from huggingface_hub import HfApi, ModelCard  # type: ignore

    api = HfApi()
    try:
        api.dataset_info(source_id)
        return HFResolvedSource(
            source_id=source_id, kind="dataset", dataset_id=source_id, dataset_ids=()
        )
    except Exception:
        pass

    # If it's not a dataset, treat it as a model repo and resolve the dataset from card metadata.
    card = ModelCard.load(source_id)
    ids = tuple(str(x) for x in (card.data.datasets or []) if str(x).strip())
    if not ids:
        raise ValueError(
            f"HF source {source_id!r} is not a dataset, and its model card has no 'datasets:' metadata."
        )
    return HFResolvedSource(source_id=source_id, kind="model", dataset_id=ids[0], dataset_ids=ids)


def _format_example_auto(example: Dict[str, Any]) -> str:
    # Prefer canonical fields when present.
    txt = example.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # FLAN/Alpaca-style.
    instruction = example.get("instruction")
    output = example.get("output")
    if isinstance(instruction, str) and isinstance(output, str):
        inp = example.get("input")
        if isinstance(inp, str) and inp.strip():
            return (
                "### Instrução\n"
                + instruction.strip()
                + "\n\n### Entrada\n"
                + inp.strip()
                + "\n\n### Resposta\n"
                + output.strip()
            )
        return "### Instrução\n" + instruction.strip() + "\n\n### Resposta\n" + output.strip()

    # Prompt/completion.
    prompt = example.get("prompt")
    completion = example.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str):
        return "### Prompt\n" + prompt.strip() + "\n\n### Resposta\n" + completion.strip()

    # Generic: serialize all string-like fields.
    flat: Dict[str, Any] = {}
    for k, v in example.items():
        if isinstance(v, (str, int, float, bool)) and str(v).strip():
            flat[str(k)] = v
    if flat:
        return json.dumps(flat, ensure_ascii=False, sort_keys=True)
    return json.dumps(example, ensure_ascii=False, sort_keys=True)


def iter_hf_dataset(
    *,
    dataset_id: str,
    split: str,
    name: Optional[str],
    streaming: bool,
    shuffle: bool,
    seed: int,
    shuffle_buffer: int,
) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_id, name=name, split=split, streaming=bool(streaming))
    if shuffle:
        if streaming:
            ds = ds.shuffle(seed=int(seed), buffer_size=int(shuffle_buffer))
        else:
            ds = ds.shuffle(seed=int(seed))
    return ds


def build_hf_corpus(
    *,
    source: str,
    out_path: str,
    split: str = "train",
    dataset: Optional[str] = None,
    name: Optional[str] = None,
    streaming: bool = True,
    shuffle: bool = True,
    seed: int = 0,
    shuffle_buffer: int = 10_000,
    target_bytes: int = 20_000_000,
    max_examples: int = 0,
    max_chars_per_example: int = 8_000,
    force: bool = False,
    log_every: int = 500,
) -> Dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    if out.exists() and not force:
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"out_path": str(out), "skipped": True}

    resolved = resolve_hf_source(source)
    dataset_id = str(dataset or resolved.dataset_id)

    t0 = time.time()
    ds = iter_hf_dataset(
        dataset_id=dataset_id,
        split=split,
        name=name,
        streaming=streaming,
        shuffle=shuffle,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )

    bytes_written = 0
    examples_written = 0

    # Return codes: 1=wrote, 0=skipped(empty), -1=limit_reached.
    def write_record(f, text: str) -> int:
        nonlocal bytes_written, examples_written
        s = (text or "").strip()
        if not s:
            return 0
        if max_chars_per_example > 0 and len(s) > int(max_chars_per_example):
            s = s[: int(max_chars_per_example)].rstrip() + "…"
        rec = f"<DOC>\n{s}\n</DOC>\n\n"
        b = rec.encode("utf-8", errors="replace")
        if target_bytes > 0 and bytes_written + len(b) > int(target_bytes):
            return -1
        f.write(rec)
        bytes_written += len(b)
        examples_written += 1
        return 1

    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for ex in ds:
            if target_bytes > 0 and bytes_written >= int(target_bytes):
                break

            rc = write_record(f, _format_example_auto(ex))
            if rc < 0:
                break

            if max_examples and examples_written >= int(max_examples):
                break
            if log_every and examples_written and (examples_written % int(log_every) == 0):
                elapsed = max(1e-9, time.time() - t0)
                mb = bytes_written / (1024 * 1024)
                rate = mb / elapsed
                print(
                    f"[hf_corpus] {examples_written} ex, {mb:.1f} MiB written ({rate:.1f} MiB/s)"
                )

    meta = {
        "source": str(source),
        "source_id": resolved.source_id,
        "source_kind": resolved.kind,
        "dataset_id": dataset_id,
        "dataset_ids_from_model_card": list(resolved.dataset_ids),
        "split": str(split),
        "config_name": name,
        "streaming": bool(streaming),
        "shuffle": bool(shuffle),
        "seed": int(seed),
        "shuffle_buffer": int(shuffle_buffer),
        "target_bytes": int(target_bytes),
        "max_examples": int(max_examples),
        "max_chars_per_example": int(max_chars_per_example),
        "examples_written": int(examples_written),
        "bytes_written": int(bytes_written),
        "out_path": str(out),
        "created_at_unix_s": time.time(),
        "elapsed_s": max(0.0, time.time() - t0),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta
