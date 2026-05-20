from dina2.alignment import evaluate_label_to_target_alignment
from dina2.histags import detect_terminal_histags


def test_detect_terminal_histags():
    mask = detect_terminal_histags("HHHHHHMABCDEHHHHHHH")
    assert mask.n_terminal == (0, 6)
    assert mask.c_terminal == (12, 19)
    assert mask.contains(0)
    assert mask.contains(18)
    assert not mask.contains(7)


def test_histag_does_not_break_label_coverage():
    _, qc = evaluate_label_to_target_alignment("ACDEFG", "ACDEFGHHHHHH", min_label_coverage=0.90, min_identity=0.90)
    assert qc.status == "pass"
    assert qc.label_coverage == 1.0
    assert qc.identity == 1.0
    assert qc.histag_spans == "C:7-12"


def test_alignment_fails_low_identity():
    _, qc = evaluate_label_to_target_alignment("ACDEFG", "AAAAAA", min_label_coverage=0.90, min_identity=0.90)
    assert qc.status == "fail"
    assert qc.reason in {"low_label_coverage", "low_identity", "mismatched_evaluated_residues"}
