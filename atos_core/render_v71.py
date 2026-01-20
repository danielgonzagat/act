from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .act import canonical_json_dumps, sha256_hex


def _sha256_text(s: str) -> str:
    return sha256_hex(str(s).encode("utf-8"))


def _index_graph(graph_snapshot: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    nodes = graph_snapshot.get("nodes") if isinstance(graph_snapshot.get("nodes"), list) else []
    edges = graph_snapshot.get("edges") if isinstance(graph_snapshot.get("edges"), list) else []
    node_by_id: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("ato_id") or "")
        if not nid:
            continue
        node_by_id[nid] = dict(n)
    edge_list = [dict(e) for e in edges if isinstance(e, dict)]
    return node_by_id, edge_list


def _reachable_subgraph(
    *,
    graph_snapshot: Dict[str, Any],
    root_ids: Sequence[str],
    max_depth: int,
    allowed_edge_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    node_by_id, edges = _index_graph(graph_snapshot)

    adj: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        src = str(e.get("src") or "")
        if not src:
            continue
        adj.setdefault(src, []).append(e)

    for src in list(adj.keys()):
        adj[src].sort(
            key=lambda e: (
                str(e.get("edge_type") or ""),
                str(e.get("dst") or ""),
                str(e.get("edge_sig") or ""),
            )
        )

    roots = [str(r) for r in root_ids if isinstance(r, str) and str(r)]
    roots = sorted(set(roots))

    seen: Set[str] = set()
    q: List[Tuple[str, int]] = []
    for r in roots:
        if r in node_by_id:
            seen.add(r)
            q.append((r, 0))
    q.sort(key=lambda x: (x[1], x[0]))

    active_edges: List[Dict[str, Any]] = []
    while q:
        cur, d = q.pop(0)
        if d >= int(max_depth):
            continue
        for e in adj.get(cur, []):
            et = str(e.get("edge_type") or "")
            if allowed_edge_types is not None and et not in allowed_edge_types:
                continue
            dst = str(e.get("dst") or "")
            if not dst:
                continue
            if dst not in node_by_id:
                continue
            active_edges.append(dict(e))
            if dst not in seen:
                seen.add(dst)
                q.append((dst, d + 1))
        q.sort(key=lambda x: (x[1], x[0]))

    active_nodes = [node_by_id[nid] for nid in sorted(seen)]
    active_edges.sort(
        key=lambda e: (
            str(e.get("src") or ""),
            str(e.get("dst") or ""),
            str(e.get("edge_type") or ""),
            str(e.get("edge_sig") or ""),
        )
    )
    return {"schema_version": 1, "roots": roots, "nodes": active_nodes, "edges": active_edges}


def render_projection(
    *,
    graph_snapshot: Dict[str, Any],
    root_ids: Sequence[str],
    max_depth: int,
    bindings: Dict[str, Any],
    goals: Sequence[Dict[str, Any]],
    plan_state: Dict[str, Any],
    style: str = "v1",
) -> Dict[str, Any]:
    """
    Deterministic projection: language is a renderable view of the active subgraph.
    Must not mutate the graph.
    """
    st = str(style or "v1")
    if st not in {"v1", "v2"}:
        raise ValueError(f"unknown_style:{st}")

    active = _reachable_subgraph(graph_snapshot=graph_snapshot, root_ids=root_ids, max_depth=int(max_depth))
    active_sig = sha256_hex(canonical_json_dumps(active).encode("utf-8"))

    inputs = {
        "schema_version": 1,
        "active_graph_sig": str(active_sig),
        "root_ids": sorted(set(str(r) for r in root_ids if str(r))),
        "max_depth": int(max_depth),
        "bindings": bindings if isinstance(bindings, dict) else {},
        "goals": list(goals) if isinstance(goals, (list, tuple)) else [],
        "plan_state": plan_state if isinstance(plan_state, dict) else {},
        "style": str(st),
    }
    inputs_sig = sha256_hex(canonical_json_dumps(inputs).encode("utf-8"))

    nodes = active.get("nodes") if isinstance(active.get("nodes"), list) else []
    edges = active.get("edges") if isinstance(active.get("edges"), list) else []

    if st == "v1":
        lines: List[str] = []
        lines.append("V71_RENDER_STYLE_V1")
        lines.append(f"active_graph_sig={active_sig}")
        lines.append(f"roots={canonical_json_dumps(active.get('roots'))}")
        lines.append(f"bindings={canonical_json_dumps(inputs['bindings'])}")
        lines.append(f"plan_state={canonical_json_dumps(inputs['plan_state'])}")
        lines.append(f"goals={canonical_json_dumps(inputs['goals'])}")
        lines.append("nodes:")
        for n in nodes:
            if not isinstance(n, dict):
                continue
            nid = str(n.get("ato_id") or "")
            nt = str(n.get("ato_type") or "")
            ns = str(n.get("ato_sig") or "")
            lines.append(f"- {nt} {nid} sig={ns}")
        lines.append("edges:")
        for e in edges:
            if not isinstance(e, dict):
                continue
            src = str(e.get("src") or "")
            dst = str(e.get("dst") or "")
            et = str(e.get("edge_type") or "")
            es = str(e.get("edge_sig") or "")
            lines.append(f"- {src} -{et}-> {dst} sig={es}")
        text = "\n".join(lines) + "\n"
    else:
        # v2 is structurally the same view, but rendered differently (substitutable projection).
        body = {
            "render_style": "V71_RENDER_STYLE_V2",
            "active_graph_sig": str(active_sig),
            "roots": active.get("roots"),
            "nodes": [
                {
                    "id": str(n.get("ato_id") or ""),
                    "type": str(n.get("ato_type") or ""),
                    "sig": str(n.get("ato_sig") or ""),
                }
                for n in nodes
                if isinstance(n, dict)
            ],
            "edges": [
                {
                    "src": str(e.get("src") or ""),
                    "dst": str(e.get("dst") or ""),
                    "type": str(e.get("edge_type") or ""),
                    "sig": str(e.get("edge_sig") or ""),
                }
                for e in edges
                if isinstance(e, dict)
            ],
            "bindings": inputs["bindings"],
            "plan_state": inputs["plan_state"],
            "goals": inputs["goals"],
        }
        text = canonical_json_dumps(body) + "\n"

    render_sig = _sha256_text(text)
    return {"text": str(text), "render_sig": str(render_sig), "inputs_sig": str(inputs_sig)}

