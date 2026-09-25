#!/usr/bin/env python3
"""Exact source-private G3-T M3T model-admission dependency gate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check() -> None:
    extension = (ROOT / "native/g3t_model_admission_extension.esk").read_text()
    test = (ROOT / "tests/g3t/model_admission_test.esk").read_text()
    runner = (ROOT / "scripts/test-g3t-model-admission.sh").read_text()
    contract = (ROOT / "docs/g3/G3_T_MODEL_ADMISSION_LEAF.md").read_text()
    assert "(define g3t-registry (vector '()))" in extension
    assert "(define (g3t-model-entry-live model operation active-call)" in extension
    assert "(m3t-find model)" in extension
    for clause in ("(eq? (vector-ref entry 0) model)",
                   "(eq? (vector-ref entry 1) 'model)",
                   "(eq? (vector-ref entry 2) 'live)",
                   "(not (null? (vector-ref entry 3)))",
                   "(eq? (vector-ref entry 10) active-call)",
                   "(eq? (vector-ref entry 11) m3t-profile)",
                   "(vector-ref m3t-paths index)",
                   "(vector-ref m3t-shapes index)",
                   "(module-mode-internal model) 'eval"):
        assert clause in extension, clause
    assert "(provide" not in extension
    assert "(extern" not in extension
    for clause in ("(g3t-checked-m3t-model-create config initial)",
                   "(m3-call 'g3t-model-admission-test",
                   "copied shell", "foreign shell", "wrong M3T kind",
                   "missing native M3T owner", "missing canonical handle",
                   "C4 profile cannot relabel M3T",
                   "train mode", "active M3T frame", "G3-T registry still empty"):
        assert clause in test, clause
    for clause in ("build_mode normal", "build_mode sanitize",
                   "repeat.stdout", "detect_leaks=1", "ESHKOL_ARENA_POISON=1"):
        assert clause in runner, clause
    assert "cannot produce G3-T C2 output" in contract
    dependency = (ROOT / "native/m3_package_source_closure.txt").read_text().splitlines()
    additions = ["native/m3_call_adapters.esk",
                 "native/g3t_model_admission_extension.esk",
                 "tests/g3t/model_admission_test.esk",
                 "scripts/test-g3t-model-admission.sh",
                 "scripts/check-g3t-model-admission.py",
                 "docs/g3/G3_T_MODEL_ADMISSION_LEAF.md"]
    expected = dependency + [path for path in additions if path not in dependency]
    actual = (ROOT / "native/g3t_model_admission_source_closure.txt").read_text().splitlines()
    assert actual == expected, "model-admission source closure differs from dependency union"
    assert len(actual) == len(set(actual)), "source closure has duplicates"
    for path in actual:
        assert (ROOT / path).is_file(), path


if __name__ == "__main__":
    check()
    print("G3-T private M3T/C2 model admission source contract: PASS")
