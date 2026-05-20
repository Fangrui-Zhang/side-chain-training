"""Terminal His-tag detection and masking."""

from __future__ import annotations

import re
from dataclasses import dataclass


HIS_TAG_RE = re.compile(r"H{6,}")


@dataclass(frozen=True)
class HisTagMask:
    """Terminal His-tag spans in 0-based, half-open coordinates."""

    n_terminal: tuple[int, int] | None
    c_terminal: tuple[int, int] | None

    def contains(self, index0: int) -> bool:
        """Return True when a 0-based position is in a detected tag."""

        for span in (self.n_terminal, self.c_terminal):
            if span is not None and span[0] <= index0 < span[1]:
                return True
        return False

    def to_string(self) -> str:
        parts = []
        if self.n_terminal:
            parts.append(f"N:{self.n_terminal[0] + 1}-{self.n_terminal[1]}")
        if self.c_terminal:
            parts.append(f"C:{self.c_terminal[0] + 1}-{self.c_terminal[1]}")
        return ";".join(parts)


def detect_terminal_histags(sequence: str, min_run: int = 6) -> HisTagMask:
    """Detect terminal runs of at least `min_run` histidines."""

    seq = sequence.upper()
    n_match = re.match(rf"^H{{{min_run},}}", seq)
    c_match = re.search(rf"H{{{min_run},}}$", seq)
    n_span = n_match.span() if n_match else None
    c_span = c_match.span() if c_match else None
    return HisTagMask(n_span, c_span)
