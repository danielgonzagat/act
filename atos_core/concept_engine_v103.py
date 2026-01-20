from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex
from .concept_model_v103 import InducedRuleV103, make_csg_rule_v103, rule_features_from_csg_v103
from .intent_grammar_v92 import tokenize_user_text_v92


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def normalize_example_text_v103(text: str) -> str:
    """
    Deterministic example normalization:
      - uses V92 tokenization (lower + accent fold + punctuation strip)
      - joins tokens with single spaces
    """
    toks = tokenize_user_text_v92(str(text or ""))
    return " ".join([str(t) for t in toks if str(t)]).strip()


def example_sig_v103(example_text_norm: str) -> str:
    return sha256_hex(canonical_json_dumps({"t": str(example_text_norm or "")}).encode("utf-8"))


def _marker_features_from_tokens(tokens: Sequence[str]) -> Set[str]:
    feats: Set[str] = set()
    toks = [str(t) for t in tokens]
    toks_set = set(toks)
    # Irony markers: "sqn" and phrase "so que nao" become the same marker feature.
    if "sqn" in toks_set:
        feats.add("m:ironia_sq")
    for i in range(0, max(0, len(toks) - 2)):
        if toks[i : i + 3] == ["so", "que", "nao"]:
            feats.add("m:ironia_sq")
    return feats


def extract_features_v103(text: str) -> Tuple[List[str], Set[str]]:
    """
    Deterministic discrete feature extractor (no embeddings):
      - token features: t:<tok>
      - bigram features: b:<tok_i>_<tok_{i+1}>
      - marker features: m:<...> (hand-written, explicit)
    Returns:
      tokens_raw: token list
      features: set[str]
    """
    toks = tokenize_user_text_v92(str(text or ""))
    tokens = [str(t) for t in toks if str(t)]
    feats: Set[str] = set()
    for t in tokens:
        feats.add(f"t:{t}")
    for i in range(0, max(0, len(tokens) - 1)):
        feats.add(f"b:{tokens[i]}_{tokens[i+1]}")
    feats |= _marker_features_from_tokens(tokens)
    return tokens, feats


def _predict_from_features(*, rule_features: Sequence[str], ex_features: Set[str]) -> bool:
    rf = [str(x) for x in list(rule_features) if str(x)]
    if not rf:
        return False
    return all(f in ex_features for f in rf)


def _mdl_bits_for_errors(*, total: int, errors: int) -> int:
    """
    Deterministic, discrete data cost:
      - each correct label costs 1 bit
      - each error costs 64 bits
    """
    e = int(errors)
    n = int(total)
    e = max(0, min(e, n))
    return int((n - e) * 1 + e * 64)


def _feature_preference_key_v103(feature: str) -> str:
    """
    Deterministic tie-break preference for induced rules.

    We prefer explicit marker features ("m:...") over token ("t:...") and bigram
    ("b:...") features, because marker features are designed to transfer across
    surface forms (e.g., "sqn" vs "só que não").
    """
    f = str(feature or "")
    if f.startswith("m:"):
        rank = 0
    elif f.startswith("t:"):
        rank = 1
    elif f.startswith("b:"):
        rank = 2
    else:
        rank = 3
    return f"{rank}:{f}"


def _candidate_tiebreak_key_v103(features: Sequence[str]) -> List[str]:
    return [str(_feature_preference_key_v103(f)) for f in list(features)]


def induce_rule_v103(
    *,
    pos_examples: Sequence[str],
    neg_examples: Sequence[str],
    max_k: int = 2,
) -> Tuple[Optional[InducedRuleV103], Dict[str, Any]]:
    """
    Induce a simple conjunction rule (RULE_MATCH_ALL) using MDL selection.
    Deterministic candidate generation + scoring; no probability, no ML.
    """
    pos = [str(x) for x in pos_examples if str(x)]
    neg = [str(x) for x in neg_examples if str(x)]
    debug: Dict[str, Any] = {"schema_version": 103}
    if len(pos) < 2 or len(neg) < 1:
        return None, dict(debug, reason="insufficient_examples")

    pos_tokens_sets: List[Set[str]] = []
    pos_feat_sets: List[Set[str]] = []
    for t in pos:
        tokens, feats = extract_features_v103(str(t))
        pos_tokens_sets.append(set(tokens))
        pos_feat_sets.append(set(feats))

    neg_feat_sets: List[Set[str]] = []
    for t in neg:
        _tokens, feats = extract_features_v103(str(t))
        neg_feat_sets.append(set(feats))

    # Universe: prefer features common to all positives to avoid trivial false negatives.
    common_pos: Set[str] = set(pos_feat_sets[0]) if pos_feat_sets else set()
    for s in pos_feat_sets[1:]:
        common_pos &= set(s)
    universe = sorted([str(x) for x in common_pos if str(x)], key=str)
    if not universe:
        # fallback to union of positive features
        uni: Set[str] = set()
        for s in pos_feat_sets:
            uni |= set(s)
        universe = sorted([str(x) for x in uni if str(x)], key=str)

    # Baseline: always predict False.
    total = int(len(pos) + len(neg))
    baseline_errors = int(len(pos))  # all positives wrong
    mdl_baseline_bits = _mdl_bits_for_errors(total=total, errors=baseline_errors)

    best: Optional[Tuple[int, List[str], int]] = None  # (total_bits, features, errors)
    candidates_scored: List[Dict[str, Any]] = []

    # Generate deterministic combinations of size 1..max_k from universe.
    feats_list = list(universe)
    max_k = int(max(1, min(int(max_k), 3)))
    # k=1
    for f1 in feats_list:
        cand = [str(f1)]
        errors = 0
        # pos labels True
        for ex in pos_feat_sets:
            if not _predict_from_features(rule_features=cand, ex_features=set(ex)):
                errors += 1
        for ex in neg_feat_sets:
            if _predict_from_features(rule_features=cand, ex_features=set(ex)):
                errors += 1
        data_bits = _mdl_bits_for_errors(total=total, errors=errors)
        model_bits = 32 + 16 * len(cand)
        total_bits = int(model_bits + data_bits)
        candidates_scored.append({"features": list(cand), "errors": int(errors), "total_bits": int(total_bits)})
        key = (int(total_bits), int(len(cand)), _candidate_tiebreak_key_v103(cand))
        if best is None or key < (best[0], len(best[1]), _candidate_tiebreak_key_v103(best[1])):
            best = (int(total_bits), list(cand), int(errors))

    if max_k >= 2:
        for i in range(0, len(feats_list)):
            for j in range(i + 1, len(feats_list)):
                cand = [str(feats_list[i]), str(feats_list[j])]
                errors = 0
                for ex in pos_feat_sets:
                    if not _predict_from_features(rule_features=cand, ex_features=set(ex)):
                        errors += 1
                for ex in neg_feat_sets:
                    if _predict_from_features(rule_features=cand, ex_features=set(ex)):
                        errors += 1
                data_bits = _mdl_bits_for_errors(total=total, errors=errors)
                model_bits = 32 + 16 * len(cand)
                total_bits = int(model_bits + data_bits)
                candidates_scored.append({"features": list(cand), "errors": int(errors), "total_bits": int(total_bits)})
                key = (int(total_bits), int(len(cand)), _candidate_tiebreak_key_v103(cand))
                if best is None or key < (best[0], len(best[1]), _candidate_tiebreak_key_v103(best[1])):
                    best = (int(total_bits), list(cand), int(errors))

    if best is None:
        return None, dict(debug, reason="no_candidates")

    best_total_bits, best_feats, best_errors = best
    # Require at least one feature and a strictly positive delta vs baseline to accept.
    mdl_delta_bits = int(mdl_baseline_bits - best_total_bits)
    if not best_feats:
        return None, dict(debug, reason="empty_rule")
    if mdl_delta_bits <= 0:
        return None, dict(debug, reason="no_mdl_improvement", mdl_delta_bits=int(mdl_delta_bits))

    # Seed tokens are the union of positive tokens (canonical).
    seed_tokens: Set[str] = set()
    for s in pos_tokens_sets:
        seed_tokens |= set([str(t) for t in s if str(t)])
    seed_tokens_canon = sorted(seed_tokens, key=str)

    # Recompute model/data bits for best.
    mdl_model_bits = int(32 + 16 * len(best_feats))
    mdl_data_bits = int(best_total_bits - mdl_model_bits)

    rule = InducedRuleV103(
        features=list(best_feats),
        mdl_baseline_bits=int(mdl_baseline_bits),
        mdl_model_bits=int(mdl_model_bits),
        mdl_data_bits=int(mdl_data_bits),
        mdl_delta_bits=int(mdl_delta_bits),
        seed_tokens=list(seed_tokens_canon),
    )
    debug2 = dict(
        debug,
        candidates_scored=sorted(
            candidates_scored,
            key=lambda d: (
                int(d.get("total_bits", 0)),
                len(d.get("features") or []),
                _candidate_tiebreak_key_v103(d.get("features") or []),
            ),
        ),
        chosen_features=list(rule.features),
        chosen_total_bits=int(best_total_bits),
        chosen_errors=int(best_errors),
        mdl_baseline_bits=int(mdl_baseline_bits),
        mdl_delta_bits=int(mdl_delta_bits),
    )
    return rule, debug2


def match_csg_v103(*, csg: Dict[str, Any], text: str) -> Dict[str, Any]:
    tokens, feats = extract_features_v103(str(text))
    rule_feats = rule_features_from_csg_v103(dict(csg))
    matched = _predict_from_features(rule_features=list(rule_feats), ex_features=set(feats))
    evidence = [f for f in rule_feats if f in feats]
    return {
        "matched": bool(matched),
        "evidence": list(sorted([str(x) for x in evidence if str(x)], key=str)),
        "tokens": list(tokens),
        "features": sorted([str(x) for x in feats if str(x)], key=str),
    }


def nearest_concepts_v103(*, concepts: List[Dict[str, Any]], text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    _tokens, feats = extract_features_v103(str(text))
    fset = set(feats)
    out: List[Dict[str, Any]] = []
    for c in concepts:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("concept_id") or "")
        rf = set([str(x) for x in rule_features_from_csg_v103(c) if str(x)])
        if not cid or not rf:
            continue
        inter = len(rf & fset)
        union = len(rf | fset)
        sim = 0.0 if union == 0 else float(inter) / float(union)
        out.append({"concept_id": str(cid), "similarity": round(sim, 6), "shared_features": sorted(list(rf & fset), key=str)[:5]})
    out.sort(key=lambda d: (-float(d.get("similarity", 0.0)), str(d.get("concept_id") or "")))
    return out[: max(0, int(top_k))]


def domain_distance_v103(*, seed_tokens: Sequence[str], tokens: Sequence[str]) -> float:
    """
    domain_distance = 1 - Jaccard(tokens_seed, tokens_input).
    """
    a = set([str(x) for x in list(seed_tokens) if str(x)])
    b = set([str(x) for x in list(tokens) if str(x)])
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    jac = 0.0 if union == 0 else float(inter) / float(union)
    return float(1.0 - jac)


def parse_teach_concept_raw_v103(text: str) -> Dict[str, Any]:
    """
    Parses: teach_concept: <NAME> += <TEXT>
           teach_concept: <NAME> -= <TEXT>
    Accepts PT alias: ensine_conceito:
    """
    raw = str(text or "")
    # Keep raw parsing; do not run V92 punctuation strip here.
    m = re.match(r"^\\s*(teach_concept|ensine_conceito)\\s*:\\s*(.+?)\\s*([+\\-]=)\\s*(.+?)\\s*$", raw, flags=re.IGNORECASE)
    if not m:
        return {"recognized": False, "ok": False, "reason": "no_match"}
    name = str(m.group(2) or "").strip()
    op = str(m.group(3) or "").strip()
    ex = str(m.group(4) or "").strip()
    if ex.startswith('\"') and ex.endswith('\"') and len(ex) >= 2:
        ex = ex[1:-1]
    if name == "" or ex == "":
        return {"recognized": True, "ok": False, "reason": "empty_name_or_text", "name": name, "op": op, "text": ex}
    polarity = "+" if op == "+=" else "-" if op == "-=" else ""
    if polarity not in {"+", "-"}:
        return {"recognized": True, "ok": False, "reason": "bad_op", "name": name, "op": op, "text": ex}
    return {"recognized": True, "ok": True, "reason": "ok", "name": name, "polarity": polarity, "text": ex}
