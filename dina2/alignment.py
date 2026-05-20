"""Strict sequence alignment utilities for residue-safe label transfer."""

from __future__ import annotations

from dataclasses import dataclass

from .histags import detect_terminal_histags


@dataclass(frozen=True)
class PairwiseAlignment:
    aligned_a: str
    aligned_b: str
    score: int

    @property
    def mapping_a_to_b(self) -> dict[int, int]:
        """Map 0-based sequence A positions to 0-based sequence B positions."""

        mapping: dict[int, int] = {}
        ia = -1
        ib = -1
        for aa, bb in zip(self.aligned_a, self.aligned_b, strict=True):
            if aa != "-":
                ia += 1
            if bb != "-":
                ib += 1
            if aa != "-" and bb != "-":
                mapping[ia] = ib
        return mapping


@dataclass(frozen=True)
class AlignmentQC:
    status: str
    reason: str
    label_coverage: float
    identity: float
    matched_label_positions: int
    reliable_label_positions: int
    mismatches: int
    label_length: int
    target_length: int
    histag_spans: str


def needleman_wunsch(
    seq_a: str,
    seq_b: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
) -> PairwiseAlignment:
    """Global alignment for modest protein sequences."""

    a = seq_a.upper()
    b = seq_b.upper()
    n = len(a)
    m = len(b)
    scores = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = i * gap
        trace[i][0] = "U"
    for j in range(1, m + 1):
        scores[0][j] = j * gap
        trace[0][j] = "L"

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = scores[i - 1][j - 1] + (match if ai == b[j - 1] else mismatch)
            up = scores[i - 1][j] + gap
            left = scores[i][j - 1] + gap
            best = max(diag, up, left)
            scores[i][j] = best
            trace[i][j] = "D" if best == diag else ("U" if best == up else "L")

    aligned_a: list[str] = []
    aligned_b: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = trace[i][j]
        if move == "D":
            aligned_a.append(a[i - 1])
            aligned_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif move == "U":
            aligned_a.append(a[i - 1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(b[j - 1])
            j -= 1

    return PairwiseAlignment("".join(reversed(aligned_a)), "".join(reversed(aligned_b)), scores[n][m])


def evaluate_label_to_target_alignment(
    label_sequence: str,
    target_sequence: str,
    min_label_coverage: float = 0.90,
    min_identity: float = 0.90,
    terminal_histag_min_run: int = 6,
) -> tuple[PairwiseAlignment, AlignmentQC]:
    """Align labels to a target sequence and compute fail-closed QC."""

    aln = needleman_wunsch(label_sequence, target_sequence)
    target_tags = detect_terminal_histags(target_sequence, min_run=terminal_histag_min_run)
    mapping = aln.mapping_a_to_b

    matched = 0
    reliable = 0
    mismatches = 0
    for label_i, target_i in mapping.items():
        matched += 1
        if target_tags.contains(target_i):
            continue
        if label_sequence[label_i].upper() == target_sequence[target_i].upper():
            reliable += 1
        else:
            mismatches += 1

    label_len = len(label_sequence)
    coverage = reliable / label_len if label_len else 0.0
    identity = reliable / (reliable + mismatches) if (reliable + mismatches) else 0.0
    status = "pass"
    reason = "ok"
    if label_len == 0:
        status, reason = "fail", "empty_label_sequence"
    elif coverage < min_label_coverage:
        status, reason = "fail", "low_label_coverage"
    elif identity < min_identity:
        status, reason = "fail", "low_identity"
    elif mismatches:
        status, reason = "fail", "mismatched_evaluated_residues"

    qc = AlignmentQC(
        status=status,
        reason=reason,
        label_coverage=coverage,
        identity=identity,
        matched_label_positions=matched,
        reliable_label_positions=reliable,
        mismatches=mismatches,
        label_length=label_len,
        target_length=len(target_sequence),
        histag_spans=target_tags.to_string(),
    )
    return aln, qc
