"""What the training reports and refuses, in words the app gives: a text is marked here with its English source and
values (``msg``), never translated (the analyzer has no UI language); the app words it (core.workers.optimizer,
``render`` with its catalog). Progress may be several texts, one a line (a tuple)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Text:
    msgid: str                               # the English source, the catalogs' msgid
    params: tuple[tuple[str, Any], ...] = ()  # its placeholders' values (a value may be a Text)

    def __str__(self) -> str:
        return render(self)


@dataclass(frozen=True)
class Concat:
    """Texts on one line, one after the other."""
    parts: tuple


def msg(msgid: str, **params) -> Text:
    return Text(msgid, tuple(params.items()))


def render(words: Any, translate: Callable[[str], str] = lambda text: text) -> str:
    """``words`` (a Text, a tuple of them, one a line, or plain text) in words, each msgid through ``translate``."""
    if isinstance(words, Text):
        values = {k: render(v, translate) if isinstance(v, (Text, tuple, Concat)) else v for k, v in words.params}
        return translate(words.msgid).format(**values)
    if isinstance(words, tuple):
        return "\n".join(render(part, translate) for part in words)
    if isinstance(words, Concat):
        return "".join(render(part, translate) for part in words.parts)
    return str(words)


class TrainingError(RuntimeError):
    """A training refused (too few songs, no annotations): its words the app's."""

    def __init__(self, text: Text):
        self.text = text
        super().__init__(render(text))
