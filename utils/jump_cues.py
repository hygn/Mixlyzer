from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from utils.jumpcue_colors import get_jumpcue_pair_color


_VALID_JUMPCUE_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

_GRAPH_BLOCK_REQUIRED = [
    "cue_id",
    "cue_label",
    "cue_comment",
    "cue_start",
    "cue_end",
    "cue_point",
    "graph_cue_id",
    "graph_idx",
    "graph_id",
    "graph_color_r",
    "graph_color_g",
    "graph_color_b",
]


def _cue_sort_key(cue: dict[str, float | str]) -> tuple[float, float, float, str]:
    point = float(cue.get("point", cue.get("start", 0.0)) or 0.0)
    start = float(cue.get("start", 0.0) or 0.0)
    end = float(cue.get("end", start) or start)
    label = str(cue.get("label", "") or "")
    return (point, start, end, label)


def _cue_timing_key(cue: dict[str, float | str]) -> tuple[float, float, float]:
    return (
        float(cue.get("start", 0.0) or 0.0),
        float(cue.get("end", cue.get("start", 0.0)) or 0.0),
        float(cue.get("point", cue.get("start", 0.0)) or 0.0),
    )


def _cue_label_sort_key(cue: dict[str, float | str]) -> tuple[str, float, float, float]:
    label = str(cue.get("label", "") or "").strip().upper()
    return (label, *_cue_timing_key(cue))


def _pair_display_sort_key(pair: dict) -> tuple[str, str, float, float, float, float]:
    forward = pair.get("forward") or {}
    backward = pair.get("backward") or {}
    f_label = str(forward.get("label", "") or "").strip().upper()
    b_label = str(backward.get("label", "") or "").strip().upper()
    f_point = float(forward.get("point", forward.get("start", 0.0)) or 0.0)
    b_point = float(backward.get("point", backward.get("start", 0.0)) or 0.0)
    return (
        min(f_label, b_label),
        max(f_label, b_label),
        min(f_point, b_point),
        max(f_point, b_point),
        f_point,
        b_point,
    )


def _link_display_sort_key(link: dict[str, Any]) -> tuple[str, str, str, float, float]:
    return (
        str(link.get("source_label", "") or "").strip().upper(),
        str(link.get("kind", "") or "").strip().upper(),
        str(link.get("target_label", "") or "").strip().upper(),
        float(link.get("source_point", 0.0) or 0.0),
        float(link.get("target_point", 0.0) or 0.0),
    )


def _normalized_cue(cue: dict | None) -> dict[str, float | str]:
    cue = cue or {}
    start = float(cue.get("start", 0.0) or 0.0)
    label = str(cue.get("label", "") or "")
    return {
        "label": label,
        "comment": str(cue.get("comment", label) or label),
        "start": start,
        "end": float(cue.get("end", start) or start),
        "point": float(cue.get("point", start) or start),
    }


def _label_from_index(idx: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 <= idx < len(alphabet):
        return alphabet[idx]
    raise ValueError("JumpCUE labels are limited to unique A-Z.")


def _cue_identity_key(cue: dict[str, float | str]) -> tuple[str, float, float, float]:
    return (
        str(cue.get("label", "") or "").strip().upper(),
        float(cue.get("start", 0.0) or 0.0),
        float(cue.get("end", cue.get("start", 0.0)) or 0.0),
        float(cue.get("point", cue.get("start", 0.0)) or 0.0),
    )


def _cue_point_key(cue: dict[str, float | str]) -> float:
    return float(cue.get("point", cue.get("start", 0.0)) or 0.0)


def _cue_merge_priority(cue: dict[str, float | str]) -> tuple[int, str, str]:
    label = str(cue.get("label", "") or "").strip().upper()
    comment = str(cue.get("comment", label) or label).strip()
    return (0 if label else 1, label, comment)


def _validate_edit_labels(sorted_cues: list[dict[str, float | str]]) -> None:
    labels: list[str] = []
    for cue in sorted_cues:
        label = str(cue.get("label", "") or "").strip().upper()
        if label not in _VALID_JUMPCUE_LABELS:
            raise ValueError("JumpCUE labels must be unique single letters A-Z.")
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise ValueError("JumpCUE labels must be unique single letters A-Z.")


def _merge_coincident_cues(
    pairs: List[dict],
) -> tuple[
    list[dict[str, float | str]],
    list[dict[str, Any]],
    dict[tuple[str, float, float, float], float],
]:
    point_to_cue: dict[float, dict[str, float | str]] = {}
    identity_to_point: dict[tuple[str, float, float, float], float] = {}
    raw_pairs: list[dict[str, Any]] = []

    for pair in pairs:
        forward = _normalized_cue(pair.get("forward", {}))
        backward = _normalized_cue(pair.get("backward", {}))
        for cue in (forward, backward):
            identity = _cue_identity_key(cue)
            point = _cue_point_key(cue)
            identity_to_point[identity] = point
            prev = point_to_cue.get(point)
            if prev is None or _cue_merge_priority(cue) < _cue_merge_priority(prev):
                point_to_cue[point] = cue
        raw_pairs.append(
            {
                "forward": forward,
                "backward": backward,
                "lag_beats": float(pair.get("lag_beats", 0.0)),
                "lag_sec": float(pair.get("lag_sec", 0.0)),
                "score": float(pair.get("score", 0.0)),
                "confidence": float(pair.get("confidence", 0.0)),
            }
        )

    return list(point_to_cue.values()), raw_pairs, identity_to_point


def _extract_jump_cue_block(features: Dict[str, Any]) -> dict[str, Any]:
    if not isinstance(features, dict):
        return {}
    block = features.get("jump_cues_np")
    if isinstance(block, dict) and block:
        return block
    prefix = "jump_cues_np."
    return {
        key[len(prefix) :]: value
        for key, value in features.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def _empty_jump_cues_np() -> Dict[str, np.ndarray]:
    return {
        "cue_id": np.asarray([], dtype=np.int32),
        "cue_label": np.asarray([], dtype="U1"),
        "cue_comment": np.asarray([], dtype="U1"),
        "cue_start": np.asarray([], dtype=np.float32),
        "cue_end": np.asarray([], dtype=np.float32),
        "cue_point": np.asarray([], dtype=np.float32),
        "graph_cue_id": np.asarray([], dtype=np.int32),
        "graph_idx": np.asarray([], dtype=np.int32),
        "graph_id": np.asarray([], dtype=np.int32),
        "graph_color_r": np.asarray([], dtype=np.uint8),
        "graph_color_g": np.asarray([], dtype=np.uint8),
        "graph_color_b": np.asarray([], dtype=np.uint8),
    }


def _build_cues_and_groups_from_pairs(
    pairs: List[dict] | None,
    *,
    canonicalize_labels: bool = False,
    merge_coincident: bool = False,
) -> tuple[list[dict[str, float | str | int]], dict[int, int]]:
    input_pairs = pairs if isinstance(pairs, list) else []
    if not input_pairs:
        return [], {}

    if merge_coincident:
        merged_cues, raw_pairs, identity_to_point = _merge_coincident_cues(input_pairs)
        sorted_cues = sorted(merged_cues, key=_cue_sort_key) if canonicalize_labels else list(merged_cues)
        if not canonicalize_labels:
            _validate_edit_labels(sorted_cues)

        point_to_id: dict[float, int] = {}
        cues: list[dict[str, float | str | int]] = []
        for idx, cue in enumerate(sorted_cues):
            label = _label_from_index(idx) if canonicalize_labels else str(cue.get("label", "") or "").strip().upper()
            if not label:
                label = _label_from_index(idx)
            point = _cue_point_key(cue)
            point_to_id[point] = idx
            cues.append(
                {
                    "id": idx,
                    "label": label,
                    "comment": str(cue.get("comment", label) or label),
                    "start": float(cue.get("start", point) or point),
                    "end": float(cue.get("end", cue.get("start", point)) or point),
                    "point": float(point),
                }
            )

        adjacency: dict[int, set[int]] = {int(cue["id"]): set() for cue in cues}
        for pair in raw_pairs:
            a_id = int(point_to_id[identity_to_point[_cue_identity_key(pair["forward"])]])
            b_id = int(point_to_id[identity_to_point[_cue_identity_key(pair["backward"])]])
            if a_id == b_id:
                continue
            adjacency[a_id].add(b_id)
            adjacency[b_id].add(a_id)
        return cues, _build_group_membership(cues, adjacency)

    cue_map: dict[tuple[str, float, float, float], dict[str, float | str]] = {}
    raw_pairs: list[dict[str, Any]] = []
    for pair in input_pairs:
        forward = _normalized_cue(pair.get("forward", {}))
        backward = _normalized_cue(pair.get("backward", {}))
        cue_map[_cue_identity_key(forward)] = forward
        cue_map[_cue_identity_key(backward)] = backward
        raw_pairs.append({"forward": forward, "backward": backward})

    sorted_cues = sorted(cue_map.values(), key=_cue_sort_key) if canonicalize_labels else list(cue_map.values())
    if not canonicalize_labels:
        _validate_edit_labels(sorted_cues)

    identity_to_id: dict[tuple[str, float, float, float], int] = {}
    cue_labels: set[str] = set()
    cues: list[dict[str, float | str | int]] = []
    for idx, cue in enumerate(sorted_cues):
        label = _label_from_index(idx) if canonicalize_labels else str(cue.get("label", "") or "").strip().upper()
        if not label:
            label = _label_from_index(idx)
        if label in cue_labels:
            raise ValueError("JumpCUE labels must be unique single letters A-Z.")
        cue_labels.add(label)
        identity = _cue_identity_key(cue)
        identity_to_id[identity] = idx
        cues.append(
            {
                "id": idx,
                "label": label,
                "comment": str(cue.get("comment", label) or label),
                "start": float(cue.get("start", 0.0)),
                "end": float(cue.get("end", 0.0)),
                "point": float(cue.get("point", 0.0)),
            }
        )

    adjacency: dict[int, set[int]] = {int(cue["id"]): set() for cue in cues}
    for pair in raw_pairs:
        a_id = int(identity_to_id[_cue_identity_key(pair["forward"])])
        b_id = int(identity_to_id[_cue_identity_key(pair["backward"])])
        if a_id == b_id:
            continue
        adjacency[a_id].add(b_id)
        adjacency[b_id].add(a_id)
    return cues, _build_group_membership(cues, adjacency)


def _build_group_membership(
    cues: list[dict[str, float | str | int]],
    adjacency: dict[int, set[int]],
) -> dict[int, int]:
    sorted_ids = sorted((int(cue["id"]) for cue in cues), key=lambda cid: float(next(c["point"] for c in cues if int(c["id"]) == cid)))
    group_by_id: dict[int, int] = {}
    group_index = 0
    for cue_id in sorted_ids:
        if cue_id in group_by_id:
            continue
        stack = [cue_id]
        while stack:
            cur = stack.pop()
            if cur in group_by_id:
                continue
            group_by_id[cur] = group_index
            for nxt in sorted(adjacency.get(cur, ())):
                if nxt not in group_by_id:
                    stack.append(nxt)
        group_index += 1
    return group_by_id


def build_jump_cues_np(
    jump_pairs: List[dict] | None,
    *,
    canonicalize_labels: bool = False,
    merge_coincident: bool = False,
) -> Dict[str, np.ndarray]:
    cues, group_by_id = _build_cues_and_groups_from_pairs(
        jump_pairs,
        canonicalize_labels=canonicalize_labels,
        merge_coincident=merge_coincident,
    )
    if not cues:
        return _empty_jump_cues_np()

    cues = sorted(cues, key=lambda cue: int(cue["id"]))
    label_width = max(1, max((len(str(cue["label"])) for cue in cues), default=1))
    comment_width = max(1, max((len(str(cue["comment"])) for cue in cues), default=1))

    graph_rows = sorted(group_by_id.items(), key=lambda item: (int(item[1]), int(item[0])))
    graph_ids = sorted(set(int(group_idx) for group_idx in group_by_id.values()))
    graph_colors = {graph_id: get_jumpcue_pair_color(graph_id) for graph_id in graph_ids}

    return {
        "cue_id": np.asarray([int(cue["id"]) for cue in cues], dtype=np.int32),
        "cue_label": np.asarray([str(cue["label"]) for cue in cues], dtype=f"U{label_width}"),
        "cue_comment": np.asarray([str(cue["comment"]) for cue in cues], dtype=f"U{comment_width}"),
        "cue_start": np.asarray([float(cue["start"]) for cue in cues], dtype=np.float32),
        "cue_end": np.asarray([float(cue["end"]) for cue in cues], dtype=np.float32),
        "cue_point": np.asarray([float(cue["point"]) for cue in cues], dtype=np.float32),
        "graph_cue_id": np.asarray([int(cue_id) for cue_id, _graph_idx in graph_rows], dtype=np.int32),
        "graph_idx": np.asarray([int(graph_idx) for _cue_id, graph_idx in graph_rows], dtype=np.int32),
        "graph_id": np.asarray(graph_ids, dtype=np.int32),
        "graph_color_r": np.asarray([graph_colors[idx][0] for idx in graph_ids], dtype=np.uint8),
        "graph_color_g": np.asarray([graph_colors[idx][1] for idx in graph_ids], dtype=np.uint8),
        "graph_color_b": np.asarray([graph_colors[idx][2] for idx in graph_ids], dtype=np.uint8),
    }


def _extract_pairs_from_new_graph_block(block: Dict[str, Any]) -> List[dict]:
    cue_length = min(
        len(block[key])
        for key in (
            "cue_id",
            "cue_label",
            "cue_comment",
            "cue_start",
            "cue_end",
            "cue_point",
        )
    )
    map_length = min(len(block[key]) for key in ("graph_cue_id", "graph_idx"))
    cue_lookup: dict[int, dict[str, float | str]] = {}
    for idx in range(cue_length):
        cue_lookup[int(block["cue_id"][idx])] = {
            "label": str(block["cue_label"][idx]),
            "comment": str(block["cue_comment"][idx]),
            "start": float(block["cue_start"][idx]),
            "end": float(block["cue_end"][idx]),
            "point": float(block["cue_point"][idx]),
        }

    members_by_graph: dict[int, list[int]] = {}
    for idx in range(map_length):
        cue_id = int(block["graph_cue_id"][idx])
        graph_idx = int(block["graph_idx"][idx])
        members_by_graph.setdefault(graph_idx, []).append(cue_id)

    pairs: list[dict] = []
    for graph_idx in sorted(members_by_graph):
        member_ids = sorted(
            set(members_by_graph[graph_idx]),
            key=lambda cue_id: float(cue_lookup.get(cue_id, {}).get("point", 0.0)),
        )
        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                a = cue_lookup.get(member_ids[i])
                b = cue_lookup.get(member_ids[j])
                if a is None or b is None:
                    continue
                pairs.append(
                    {
                        "forward": dict(a),
                        "backward": dict(b),
                        "lag_beats": 0.0,
                        "lag_sec": abs(float(b["point"]) - float(a["point"])),
                        "score": 1.0,
                        "confidence": 1.0,
                    }
                )
    return pairs


def extract_jump_cue_pairs(features: Dict[str, Any]) -> List[dict]:
    """
    New format: cue rows + graph membership rows.
    Legacy fallback: 0.1.1 flattened forward/backward arrays only.
    """
    block = _extract_jump_cue_block(features)
    if not block:
        return []

    if all(key in block for key in _GRAPH_BLOCK_REQUIRED):
        return _extract_pairs_from_new_graph_block(block)

    required = [
        "forward_label",
        "forward_start",
        "forward_end",
        "forward_point",
        "backward_label",
        "backward_start",
        "backward_end",
        "backward_point",
        "lag_beats",
        "lag_sec",
        "score",
        "confidence",
    ]
    if any(key not in block for key in required):
        return []

    length = min(len(block[key]) for key in required)
    pairs: list[dict] = []
    for idx in range(length):
        pairs.append(
            {
                "forward": {
                    "label": str(block["forward_label"][idx]),
                    "comment": str(block["forward_label"][idx]),
                    "start": float(block["forward_start"][idx]),
                    "end": float(block["forward_end"][idx]),
                    "point": float(block["forward_point"][idx]),
                },
                "backward": {
                    "label": str(block["backward_label"][idx]),
                    "comment": str(block["backward_label"][idx]),
                    "start": float(block["backward_start"][idx]),
                    "end": float(block["backward_end"][idx]),
                    "point": float(block["backward_point"][idx]),
                },
                "lag_beats": float(block["lag_beats"][idx]),
                "lag_sec": float(block["lag_sec"][idx]),
                "score": float(block["score"][idx]),
                "confidence": float(block["confidence"][idx]),
            }
        )
    return pairs


def extract_jump_cue_graph(features: Dict[str, Any]) -> tuple[List[dict[str, float | str | int]], List[dict[str, Any]]]:
    block = _extract_jump_cue_block(features)
    if not block:
        return [], []

    if not all(key in block for key in _GRAPH_BLOCK_REQUIRED):
        return build_jump_cue_graph(extract_jump_cue_pairs(features))

    cue_length = min(
        len(block[key])
        for key in (
            "cue_id",
            "cue_label",
            "cue_comment",
            "cue_start",
            "cue_end",
            "cue_point",
        )
    )
    map_length = min(len(block[key]) for key in ("graph_cue_id", "graph_idx"))
    graph_length = min(len(block[key]) for key in ("graph_id", "graph_color_r", "graph_color_g", "graph_color_b"))

    color_by_graph: dict[int, tuple[int, int, int]] = {}
    for idx in range(graph_length):
        graph_id = int(block["graph_id"][idx])
        color_by_graph[graph_id] = (
            int(block["graph_color_r"][idx]),
            int(block["graph_color_g"][idx]),
            int(block["graph_color_b"][idx]),
        )

    graph_by_cue_id: dict[int, int] = {}
    for idx in range(map_length):
        graph_by_cue_id[int(block["graph_cue_id"][idx])] = int(block["graph_idx"][idx])

    cues: list[dict[str, float | str | int]] = []
    cue_lookup: dict[int, dict[str, float | str | int]] = {}
    for idx in range(cue_length):
        cue_id = int(block["cue_id"][idx])
        graph_idx = int(graph_by_cue_id.get(cue_id, 0))
        color = color_by_graph.get(graph_idx, get_jumpcue_pair_color(graph_idx))
        cue = {
            "id": cue_id,
            "label": str(block["cue_label"][idx]),
            "comment": str(block["cue_comment"][idx]),
            "start": float(block["cue_start"][idx]),
            "end": float(block["cue_end"][idx]),
            "point": float(block["cue_point"][idx]),
            "component_index": graph_idx,
            "graph_idx": graph_idx,
            "color_index": graph_idx,
            "color": color,
        }
        cues.append(cue)
        cue_lookup[cue_id] = cue

    cues.sort(key=lambda cue: float(cue.get("point", 0.0)))
    members_by_graph: dict[int, list[dict[str, float | str | int]]] = {}
    for cue in cues:
        graph_idx = int(cue.get("graph_idx", 0) or 0)
        members_by_graph.setdefault(graph_idx, []).append(cue)

    links: list[dict[str, Any]] = []
    for graph_idx in sorted(members_by_graph):
        members = sorted(members_by_graph[graph_idx], key=lambda cue: float(cue.get("point", 0.0)))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a = members[i]
                b = members[j]
                lag_sec = abs(float(b["point"]) - float(a["point"]))
                directed = (
                    ("F", a, b),
                    ("B", b, a),
                )
                for kind, src, dst in directed:
                    links.append(
                        {
                            "kind": kind,
                            "source_id": int(src["id"]),
                            "target_id": int(dst["id"]),
                            "source_label": str(src["label"]),
                            "target_label": str(dst["label"]),
                            "source": dict(src),
                            "target": dict(dst),
                            "source_point": float(src["point"]),
                            "target_point": float(dst["point"]),
                            "component_index": graph_idx,
                            "graph_idx": graph_idx,
                            "color_index": graph_idx,
                            "color": src["color"],
                            "lag_beats": 0.0,
                            "lag_sec": lag_sec,
                            "score": 1.0,
                            "confidence": 1.0,
                        }
                    )

    links.sort(key=_link_display_sort_key)
    return cues, links


def sort_jump_cue_pairs_for_display(pairs: List[dict] | None) -> List[dict]:
    if not isinstance(pairs, list):
        return []
    return sorted((dict(pair) for pair in pairs), key=_pair_display_sort_key)


def build_jump_cue_graph(pairs: List[dict] | None) -> tuple[List[dict[str, float | str | int]], List[dict[str, Any]]]:
    block = build_jump_cues_np(pairs, canonicalize_labels=True, merge_coincident=True)
    return extract_jump_cue_graph({"jump_cues_np": block})


__all__ = [
    "build_jump_cues_np",
    "extract_jump_cue_pairs",
    "extract_jump_cue_graph",
    "sort_jump_cue_pairs_for_display",
    "build_jump_cue_graph",
]
