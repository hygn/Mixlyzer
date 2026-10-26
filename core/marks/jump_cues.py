"""The JumpCUEs' slots (rekordbox's hot cue letters): each cue is a slot of its own, a single letter A-Z, no two
alike. The one place of that rule: the model's edits, the editor's save and the stored block check it here, and a
stored block whose letters are not slots is lettered here (its stored form: core.analysis_library.jump_cues)."""
from __future__ import annotations

from dataclasses import replace
from typing import List

from analyzer.music.marks import JumpCueGroup
from system.i18n import tr

SLOT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SLOTS = frozenset(SLOT_LETTERS)


def has_slot_letters(groups: List[JumpCueGroup]) -> bool:
    """Every cue in a slot of its own: a single letter A-Z as it is spelled (the stored blocks read: upper case, no
    spaces), no two alike."""
    labels = [cue.label for group in groups for cue in group.cues]
    return all(label in _SLOTS for label in labels) and len(labels) == len(set(labels))


def check_slot_letters(groups: List[JumpCueGroup]) -> None:
    """ValueError unless every cue is in a slot of its own (has_slot_letters)."""
    if not has_slot_letters(groups):
        raise ValueError(tr("JumpCUE labels must be unique single letters A-Z."))


def lettered_by_point(groups: List[JumpCueGroup]) -> List[JumpCueGroup]:
    """The groups with their cues lettered A-Z by point (a name that was its letter follows it). There are 26 slots:
    cues past the 26th by point are left out (said on the console), and a group left without cues goes with them."""
    order = sorted((cue.point, gi, ci) for gi, group in enumerate(groups) for ci, cue in enumerate(group.cues))
    if len(order) > len(SLOT_LETTERS):
        print(f"[JumpCUE] {len(order) - len(SLOT_LETTERS)} of {len(order)} cues past slot Z left out "
              f"({len(SLOT_LETTERS)} slots)")
    letter = {(gi, ci): SLOT_LETTERS[k] for k, (_point, gi, ci) in enumerate(order[:len(SLOT_LETTERS)])}
    lettered = []
    for gi, group in enumerate(groups):
        cues = tuple(replace(cue, label=letter[(gi, ci)],
                             comment=letter[(gi, ci)] if cue.comment == cue.label else cue.comment)
                     for ci, cue in enumerate(group.cues) if (gi, ci) in letter)
        if cues:
            lettered.append(replace(group, cues=cues))
    return lettered
