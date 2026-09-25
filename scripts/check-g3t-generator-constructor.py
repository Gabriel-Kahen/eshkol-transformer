#!/usr/bin/env python3
"""Structural gate for the private G3-T seeded Eshkol constructor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ordered(text: str, parts: tuple[str, ...], label: str) -> None:
    cursor = -1
    for part in parts:
        cursor = text.find(part, cursor + 1)
        assert cursor >= 0, f"{label}: {part}"


def check() -> None:
    source = (ROOT / "native/g3t_generator_constructor_extension.esk").read_text()
    test = (ROOT / "tests/g3t/generator_constructor_test.esk").read_text()
    runner = (ROOT / "scripts/test-g3t-generator-constructor.sh").read_text()
    note = (ROOT / "docs/g3/G3_T_GENERATOR_CONSTRUCTOR_LEAF.md").read_text()
    assert "g3c4" not in source.lower()
    assert "(provide" not in source
    for stem in ("generator-seed", "generator-rng", "generator-close", "last-error-domain",
                 "last-error-category", "last-error-code"):
        assert f"g3t-native-{stem}" in source, stem
    for phrase in ("(define (g3t-generator-create model tokenizer config)",
                   "(define (g3t-generator-close! generator)",
                   "(m3-call operation", "(g3t-model-entry-live model operation #f)",
                   "(t1-private-entry tokenizer)", "(t1-core-tokenizer? core)",
                   "(t1-tokenizer-core-vocab-size core) 256",
                   "G3-T config must have exactly eight pairs",
                   "expected an exact G3-T RNG owner",
                   "(vector shell 'generator 'pending", "(vector-set! g3t-registry 0 next)",
                   "(car (vector-ref g3t-registry 0))", "(g3t-native-generator-seed",
                   "(vector-set! canonical 2 'live)", "(vector-set! canonical 2 'dead)",
                   "(g3t-native-generator-close", "(vector-set! entry 2 'dead)"):
        assert phrase in source, phrase
    constructor = source.split("(define (g3t-generator-create model tokenizer config)", 1)[1]
    ordered(constructor, ("(g3t-generator-find source operation)",
                          "(vector-set! g3t-registry 0 next)",
                          "(g3t-native-generator-rng"), "RNG admission before copy")
    assert source.count("sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704") == 1
    assert "(let* ((domain (g3t-native-last-error-domain))" in source
    assert "(list 'source-code (if valid code 10))" in source
    symbols = (ROOT / "native/g3t_generator_local_symbols.txt").read_text().splitlines()
    assert symbols == ["g3t-model-entry-live", "g3t-fail-raw", "g3t-native-fail-raw",
                       "g3t-seed-config", "g3t-generator-find"]
    assert "(define (g3t-model-entry-live " in (
        ROOT / "native/g3t_model_admission_extension.esk").read_text()
    for symbol in symbols[1:]:
        assert f"(define ({symbol} " in source, symbol
    ordered(test, ("(load \"m3_package_root.esk\")",
                   "(load \"m3_call_adapters.esk\")",
                   "(load \"g3t_model_admission_extension.esk\")",
                   "(load \"g3t_generator_constructor_extension.esk\")"), "load order")
    for phrase in ("authentic strict tokenizer unsupported", "wrong config profile",
                   "invalid greedy temperature", "RNG owner unavailable",
                   "train model", "M3T profile cannot be relabeled",
                   "same-model active frame", "native context allocation failure",
                   "A2 cache allocation failure", "context failure tombstone",
                   "source-rooted live generator", "busy generator close",
                   "idempotent close", "categorical C2 policy admitted",
                   "post-close fresh create"):
        assert phrase in test, phrase
    for phrase in ("build_mode normal", "build_mode sanitize", "repeat.stdout",
                   "detect_leaks=1", "production.o", "test hook escaped"):
        assert phrase in runner, phrase
    assert "Frame transcript" in note
    dependency = (ROOT / "native/g3t_native_context_source_closure.txt").read_text().splitlines()
    additions = ["native/g3t_generator_constructor_extension.esk",
                 "native/g3t_generator_local_symbols.txt",
                 "tests/g3t/generator_constructor_test.esk",
                 "scripts/check-g3t-generator-constructor.py",
                 "scripts/test-g3t-generator-constructor.sh",
                 "tests/q0/test_python_isolation.py",
                 "docs/g3/G3_T_GENERATOR_CONSTRUCTOR_LEAF.md"]
    expected = dependency + [path for path in additions if path not in dependency]
    actual = (ROOT / "native/g3t_generator_constructor_source_closure.txt").read_text().splitlines()
    assert actual == expected, "constructor closure differs from exact dependency union"
    assert len(actual) == len(set(actual)), "duplicate source closure path"
    for path in actual:
        if path.startswith(".deps/eshkol-src/"):
            continue
        assert (ROOT / path).is_file(), path


if __name__ == "__main__":
    check()
    print("G3-T private seeded generator constructor source contract: PASS")
