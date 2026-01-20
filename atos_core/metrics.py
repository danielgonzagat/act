from __future__ import annotations

import math
import os
import re
import resource
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_RE = re.compile(r"\s+|[^\s]+", flags=re.UNICODE)


def tokenize_text(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def is_space(tok: str) -> bool:
    return bool(tok) and tok.isspace()


def _filtered(tokens: Sequence[str], *, ignore_space: bool) -> List[str]:
    if not ignore_space:
        return [t for t in tokens if t not in {"<BOS>"}]
    return [t for t in tokens if t not in {"<BOS>"} and not is_space(t)]


def distinct_n(tokens: Sequence[str], n: int, *, ignore_space: bool = True) -> float:
    toks = _filtered(tokens, ignore_space=ignore_space)
    if n <= 0:
        return 0.0
    if len(toks) < n:
        return 0.0
    total = len(toks) - n + 1
    s = set(tuple(toks[i : i + n]) for i in range(total))
    return len(s) / total if total > 0 else 0.0


def repeat_ngram_rate(tokens: Sequence[str], n: int, *, ignore_space: bool = True) -> float:
    toks = _filtered(tokens, ignore_space=ignore_space)
    if n <= 0 or len(toks) < n:
        return 0.0
    seen = set()
    repeats = 0
    total = len(toks) - n + 1
    for i in range(total):
        ng = tuple(toks[i : i + n])
        if ng in seen:
            repeats += 1
        else:
            seen.add(ng)
    return repeats / total if total > 0 else 0.0


def loop_rate(
    tokens: Sequence[str], *, n: int = 3, window: int = 128, ignore_space: bool = False
) -> float:
    toks = _filtered(tokens, ignore_space=ignore_space)
    if len(toks) < n:
        return 0.0
    total = len(toks) - n + 1
    hits = 0
    history: Dict[Tuple[str, ...], int] = {}
    for i in range(total):
        ng = tuple(toks[i : i + n])
        last = history.get(ng)
        if last is not None and i - last <= window:
            hits += 1
        history[ng] = i
    return hits / total if total > 0 else 0.0


def fluency_metrics(tokens: Sequence[str]) -> Dict[str, float]:
    return {
        "repeat3": repeat_ngram_rate(tokens, 3, ignore_space=False),
        "distinct2": distinct_n(tokens, 2, ignore_space=False),
        "loop_rate": loop_rate(tokens, n=3, window=128, ignore_space=False),
    }


def ema(prev: float, x: float, *, alpha: float = 0.02) -> float:
    if prev != prev:  # NaN guard
        return x
    return (1.0 - alpha) * prev + alpha * x


def rss_bytes_best_effort() -> int:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss = int(getattr(ru, "ru_maxrss", 0))
        # On macOS ru_maxrss is bytes; on Linux it's kilobytes.
        if sys.platform.startswith("darwin"):
            return rss
        return rss * 1024
    except Exception:
        return 0


def safe_log2(x: float) -> float:
    return math.log(x + 1e-12, 2)
