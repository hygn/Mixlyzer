"""JumpCUEs' stored form: a graph block per track (``jump_cues_np``, the user's; ``jump_cues_auto_np``, the automatic
analysis's): the cues (slot letter, name, passage, point), each cue's group, each group's colour preset. Read and
written as analyzer.music.marks.JumpCueGroup (extract_jump_cue_groups, build_jump_cue_block); the analysis's JumpCUE
pairs (the detector's) are made into a block by build_jump_cues_np."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from analyzer.music.marks import JumpCue, JumpCueGroup, cues_by_point
from core.marks.colors import JUMPCUE_PAIR_COLORS
from core.marks.jump_cues import SLOT_LETTERS, check_slot_letters
from system.i18n import tr


_VALID_JUMPCUE_LABELS = set(SLOT_LETTERS)

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
    "graph_color_idx",
]


def _cue_sort_key(cue: dict[str, float | str]) -> tuple[float, float, float, str]:
    point = float(cue.get("point", cue.get("start", 0.0)) or 0.0)
    start = float(cue.get("start", 0.0) or 0.0)
    end = float(cue.get("end", start) or start)
    label = str(cue.get("label", "") or "")
    return (point, start, end, label)


def _normalized_cue(cue: dict | None) -> dict[str, float | str]:
    cue = cue or {}
    start = float(cue.get("start", 0.0) or 0.0)
    label = str(cue.get("label", "") or "")
    out = {
        "label": label,
        "comment": str(cue.get("comment", label) or label),
        "start": start,
        "end": float(cue.get("end", start) or start),
        "point": float(cue.get("point", start) or start),
    }
    color_idx = _cue_color_idx(cue)
    if color_idx is not None:
        out["color_idx"] = color_idx
    return out


def _cue_color_idx(cue: dict) -> int | None:
    """The group colour preset a cue carries (None: not assigned)."""
    try:
        value = cue.get("color_idx")
        return None if value is None or int(value) < 0 else int(value)
    except (TypeError, ValueError):
        return None


def _label_from_index(idx: int) -> str:
    if 0 <= idx < len(SLOT_LETTERS):
        return SLOT_LETTERS[idx]
    raise ValueError(tr("JumpCUE labels are limited to unique A-Z."))


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
            raise ValueError(tr("JumpCUE labels must be unique single letters A-Z."))
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise ValueError(tr("JumpCUE labels must be unique single letters A-Z."))


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
                if prev is not None and "color_idx" not in cue and "color_idx" in prev:
                    cue = dict(cue, color_idx=prev["color_idx"])
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


def _extract_jump_cue_block(features: Dict[str, Any], key: str = "jump_cues_np") -> dict[str, Any]:
    if not isinstance(features, dict):
        return {}
    block = features.get(key)
    if isinstance(block, dict) and block:
        return block
    prefix = f"{key}."
    return {
        name[len(prefix) :]: value
        for name, value in features.items()
        if isinstance(name, str) and name.startswith(prefix)
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
        "graph_color_idx": np.asarray([], dtype=np.int32),
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
                    **({"color_idx": cue["color_idx"]} if "color_idx" in cue else {}),
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
            raise ValueError(tr("JumpCUE labels must be unique single letters A-Z."))
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
                **({"color_idx": cue["color_idx"]} if "color_idx" in cue else {}),
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
    # A group's colour is the preset its cues carry (the earliest cue's), not its
    # number; groups without one take preset number % N.
    color_idx_by_graph: dict[int, int] = {}
    for cue in sorted(cues, key=lambda c: float(c["point"])):
        graph_id = int(group_by_id.get(int(cue["id"]), 0))
        if graph_id not in color_idx_by_graph and "color_idx" in cue:
            color_idx_by_graph[graph_id] = int(cue["color_idx"])
    for graph_id in graph_ids:
        color_idx_by_graph.setdefault(graph_id, graph_id % len(JUMPCUE_PAIR_COLORS))

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
        "graph_color_idx": np.asarray([color_idx_by_graph[idx] for idx in graph_ids], dtype=np.int32),
    }


def _graph_color_indices(block: Dict[str, Any]) -> dict[int, int]:
    """Colour preset index of each group."""
    graph_ids = np.asarray(block["graph_id"], dtype=int).tolist()
    colors = np.asarray(block["graph_color_idx"], dtype=int).tolist()
    return {int(g): int(c) for g, c in zip(graph_ids, colors)}


def extract_jump_cue_groups(features: Dict[str, Any], key: str = "jump_cues_np") -> List[JumpCueGroup]:
    """The JumpCUE groups of the graph block ``key`` ('jump_cues_np' = the user's, 'jump_cues_auto_np' = the automatic
    analysis's), by their stored group index, each one's cues by point and its colour preset."""
    return _block_groups(_extract_jump_cue_block(features, key))


def jump_cue_groups_of_pairs(jump_pairs: List[dict] | None) -> List[JumpCueGroup]:
    """The detector's JumpCUE pairs (analyzer.pipeline.jumpcues) as the groups they are stored as: lettered A-Z by
    point, coincident cues merged."""
    return _block_groups(build_jump_cues_np(jump_pairs, canonicalize_labels=True, merge_coincident=True))


def _block_groups(block: Dict[str, Any]) -> List[JumpCueGroup]:
    if not block or not all(name in block for name in _GRAPH_BLOCK_REQUIRED):
        return []
    cue_length = min(len(block[name]) for name in ("cue_id", "cue_label", "cue_comment", "cue_start", "cue_end",
                                                   "cue_point"))
    map_length = min(len(block[name]) for name in ("graph_cue_id", "graph_idx"))
    graph_of_cue = {int(block["graph_cue_id"][i]): int(block["graph_idx"][i]) for i in range(map_length)}
    color_of_graph = _graph_color_indices(block)
    members: dict[int, list[JumpCue]] = {}
    for i in range(cue_length):
        graph_idx = graph_of_cue.get(int(block["cue_id"][i]), 0)
        # a letter in its one spelling (build_jump_cue_block's: "a" / " B" are slots A / B); a name that was the
        # letter follows it
        stored_label, comment = str(block["cue_label"][i]), str(block["cue_comment"][i])
        label = stored_label.strip().upper()
        members.setdefault(graph_idx, []).append(JumpCue(
            label=label, comment=label if comment == stored_label else comment,
            start=float(block["cue_start"][i]), end=float(block["cue_end"][i]), point=float(block["cue_point"][i])))
    return [JumpCueGroup(tuple(sorted(cues, key=lambda cue: cue.point)), int(color_of_graph.get(g, g)))
            for g, cues in sorted(members.items())]


def jump_cue_point_groups(features: Dict[str, Any], key: str = "jump_cues_np") -> List[List[float]]:
    """Cue points (seconds, ascending) of each JumpCUE group of the graph block ``key`` (as
    extract_jump_cue_groups); groups of fewer than two cues are left out."""
    return [[cue.point for cue in group.cues] for group in extract_jump_cue_groups(features, key)
            if len(group.cues) >= 2]


def build_jump_cue_block(groups: List[JumpCueGroup] | None) -> Dict[str, np.ndarray]:
    """The stored graph block of ``groups`` (empty groups left out): the cues numbered by point, the groups by their
    first cue's point, each with its colour preset; a cue without a name stored with its letter as its comment.
    ValueError when the letters are not single letters A-Z, no two alike (each cue is a slot)."""
    groups = [group for group in (groups or []) if group.cues]
    check_slot_letters(groups)
    if not groups:
        return _empty_jump_cues_np()
    groups = sorted(groups, key=lambda group: min(cue.point for cue in group.cues))
    cues = cues_by_point(groups)
    labels = [cue.label for cue, _gi in cues]
    comments = [str(cue.comment or "") or label for (cue, _gi), label in zip(cues, labels)]
    return {
        "cue_id": np.arange(len(cues), dtype=np.int32),
        "cue_label": np.asarray(labels, dtype=f"U{max(1, max(len(x) for x in labels))}"),
        "cue_comment": np.asarray(comments, dtype=f"U{max(1, max(len(x) for x in comments))}"),
        "cue_start": np.asarray([cue.start for cue, _gi in cues], dtype=np.float32),
        "cue_end": np.asarray([cue.end for cue, _gi in cues], dtype=np.float32),
        "cue_point": np.asarray([cue.point for cue, _gi in cues], dtype=np.float32),
        "graph_cue_id": np.arange(len(cues), dtype=np.int32),
        "graph_idx": np.asarray([gi for _cue, gi in cues], dtype=np.int32),
        "graph_id": np.arange(len(groups), dtype=np.int32),
        "graph_color_idx": np.asarray([int(group.color_idx) for group in groups], dtype=np.int32),
    }


__all__ = [
    "build_jump_cues_np",
    "build_jump_cue_block",
    "extract_jump_cue_groups",
    "jump_cue_point_groups",
]
