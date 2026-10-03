#!/usr/bin/env python3
"""Exact source boundary for source-private G3-R typed G3-T RNG import."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def ordered(source: str, *parts: str) -> None:
    cursor = -1
    for part in parts:
        cursor = source.find(part, cursor + 1)
        assert cursor >= 0, part


def balanced(source: str) -> bool:
    depth = 0
    quoted = escaped = comment = False
    for char in source:
        if char == "\n":
            comment = False
        if comment:
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == ";":
            comment = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            assert depth >= 0
    return depth == 0 and not quoted


base = (root / "native/g3m_seeded_p1_g1_source_closure.txt").read_text().splitlines()
owner = (root / "native/g3t_rng_owner_source_closure.txt").read_text().splitlines()
add = [
    "native/g3r_rng_import_extension.esk",
    "tests/g3r/rng_import_native_test.c",
    "tests/g3r/rng_import_allocation_shim.cpp",
    ".deps/eshkol-src/lib/core/arena_memory.h",
    "tests/g3r/rng_import_on.esk",
    "tests/g3r/rng_import_off.esk",
    "tests/g3r/rng_import_oracle.py",
    "scripts/check-g3r-rng-import.py",
    "scripts/test-g3r-rng-import.sh",
    "tests/q0/test_python_isolation.py",
    "docs/g3/G3_R_TYPED_RNG_IMPORT_CONTRACT.md",
    "docs/g3/G3_R_TYPED_RNG_IMPORT_LEAF.md",
    "docs/ROADMAP.md",
]
expected = list(dict.fromkeys(base + owner + add))
actual = (root / "native/g3r_rng_import_source_closure.txt").read_text().splitlines()
assert actual == expected
assert len(actual) == len(set(actual))
assert all((root / path).is_file() for path in actual if not path.startswith(".deps/"))

native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
header = (root / "src/eshkol_transformer/g3t_transport.h").read_text()
extension = (root / "native/g3r_rng_import_extension.esk").read_text()
on = (root / "tests/g3r/rng_import_on.esk").read_text()
off = (root / "tests/g3r/rng_import_off.esk").read_text()
runner = (root / "scripts/test-g3r-rng-import.sh").read_text()
ctor = native.split("void *et_g3t_private_rng_words_create_v1(", 1)[1].split(
    "int64_t et_g3t_private_rng_release_v1(", 1
)[0]
assert "#error \"G3-R word import requires kind-8 RNG clone ownership" in native
assert "#ifdef ET_G3T_RECORD_RNG_IMPORT_PRIVATE" in native
assert "#ifdef ET_G3T_RECORD_RNG_IMPORT_PRIVATE" in header
ordered(ctor, "g3t_clear();", "version != 1 || seed < 0", "calloc(1, sizeof(*clone))",
        "clone->words[0] = version", "clone->words[1] = seed",
        "clone->words[2] = counter_low", "clone->words[3] = counter_high",
        "clone->h.kind = G3T_RNG_CLONE", "clone->h.state = G3T_LIVE",
        "g3t_registry = &clone->h")
assert "memset(clone->words, 0, sizeof(clone->words))" in native
assert "memcpy(c->rng, source->words, sizeof(c->rng))" in native
assert "(provide" not in extension and "(load" not in extension
for public_root in ("native/g3g_public_extension.esk",
                    "native/g3g_package_root.esk",
                    "native/cli3_extension.esk"):
    assert "g3r_rng_import_extension.esk" not in (root / public_root).read_text()
assert balanced(extension) and balanced(on) and balanced(off)
ordered(extension, "(m3-call operation", "(m3t-exact-i64? version)",
        "(let* ((shell", "(next (cons entry", "(vector-set! g3t-registry 0 next)",
        "(g3r-native-rng-words-create", "(vector-set! canonical-ledger 0 native)",
        "(vector-set! canonical 3 native)", "(vector-set! canonical 10 #f)",
        "(vector-set! canonical 2 'live)")
assert "(g3t-native-rng-release created)" in extension
assert "(g3t-prefill-dead! entry)" in extension
assert "(exit 134)" in extension
for phrase in (
    "all four signed words bit-exact", "native zeroing and tombstone",
    "cut tombstones pending entry without clone", "source shell release independent",
    "foreign M3 registry pointer rejected", "exhaustion before categorical draw",
    "forged shell rejected", "stale shell rejected", "busy source rejected",
    "raw native pointer not Eshkol owner", "G3R_ORACLE",
    "Eshkol vector/cons allocation cuts reached",
    "genuine published-output RNG predecessor words",
    "imported four words equal authentic output-clone predecessor",
):
    assert phrase in on, phrase
ordered(on, "(g3m-generate-p1-g1! predecessor-generator predecessor-input)",
        "(g3t-generation-output-rng predecessor-output)",
        "(g3r-rng-owner-from-checked-words 1 1729 1 0)")
assert "(display expected-low)" not in on
assert "(display expected-high)" not in on
assert "(g3r-test-rng-word native-generator index)" in on
assert "(g3r-test-rng-clone-word native-clone index)" in on
assert "g3r_rng_import_extension.esk" not in off
for phrase in ("build_mode off", "build_mode normal", "build_mode sanitize",
               "repeat.stdout", "detect_leaks=1", "rng_import_native_test.c",
               "rng_import_oracle.py", "et_g3t_private_rng_words_create_v1",
               "closure.sha256", "--wrap=arena_allocate_vector_with_header",
               "--wrap=arena_allocate_cons_with_header"):
    assert phrase in runner, phrase
print("G3-R source-private typed RNG import contract: PASS")
