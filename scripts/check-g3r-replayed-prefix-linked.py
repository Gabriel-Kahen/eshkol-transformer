#!/usr/bin/env python3
"""Closed, source-only manifest and command check for the linked P1 gate."""
from hashlib import sha256
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = "native/g3m_seeded_p1_g1_source_closure.txt"
CLOSURE = "native/g3r_replayed_prefix_linked_source_closure.txt"
MANIFEST = "native/g3r_replayed_prefix_linked_manifest.tsv"
ADDED = [
    "native/g3r_rng_import_extension.esk",
    "native/g3r_replayed_prefix_extension.esk",
    "tests/g3r/replayed_prefix_transition_test.c",
    "scripts/test-g3r-replayed-prefix-native.sh",
    "docs/g3/G3_R_REPLAYED_PREFIX_GENERATED_TRANSITION_LEAF.md",
    "tests/g3r/replayed_prefix_linked_probe.c",
    "tests/g3r/replayed_prefix_linked_test.esk",
    "scripts/test-g3r-replayed-prefix-linked.sh",
    "scripts/check-g3r-replayed-prefix-linked.py",
    "docs/g3/G3_R_REPLAYED_PREFIX_LINKED_GATE.md",
    "toolchain/eshkol.lock",
]
HEADERS = [
    ("eshkol_commit", "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9"),
    ("llvm_version", "21.1.8"),
    ("clang_version", "21.1.8"),
    ("supported_cc", "clang-21"),
    ("supported_cxx", "clang++-21"),
    ("compiler_provenance_file", ".deps/eshkol-build/eshkol-transformer-provenance.tsv"),
    ("compiler_provenance_keys", "eshkol_commit,cc_path,cxx_path,cc_version,cxx_version,llvm_version,eshkol_binary_sha256"),
]
# Freeze the runner and executable build-mode body independently of the
# repinnable manifest row. Command or receipt-path changes need review.
RUNNER_SHA256 = "e4a0c4492220efa2514edfb1665f111e7a72a2beff472cf79928c68048fc5640"
BUILD_MODE_SHA256 = "ff381e33d2e8c445d28cf9a21cabb8b38a5ed3080a66b4b8d7e569a6b279f189"
RUNNER_NEEDLES = [
    "G3R_LINKED_RUNTIME_RELEASE", "G3R_LINKED_EXPECTED_COMMIT",
    "git -C \"$PROJECT_ROOT\" status --porcelain", "verify_toolchain",
    "resolve_provenance_compilers", "eshkol-transformer-provenance.tsv",
    "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9",
    "-DET_G3R_REPLAYED_PREFIX_PRIVATE", "-DET_G3R_COMMITTED_HISTORY_PRIVATE",
    "-DET_G3T_RECORD_RNG_IMPORT_PRIVATE", "-DET_G3T_MANUAL_P2_PREFILL_PRIVATE",
    "replayed_prefix_linked_probe.c", "-Wl,--wrap=et_kernel_runtime_dispatch",
    "replayed_prefix_linked_test.esk", "--strict-types", "--no-stdlib",
    "ESHKOL_ARENA_POISON=1", "build_mode normal", "build_mode sanitize",
    "raw-commands-and-exits.log", "printf 'exit: %d", "cmp \"$temporary/normal/stdout\"",
    "feature-off replayed-prefix symbol escaped", "feature-on replayed-prefix symbol missing",
]
FIXTURE_NEEDLES = [
    '(load "g3t_text_result_extension.esk")',
    '(load "g3r_rng_import_extension.esk")',
    '(load "g3r_replayed_prefix_extension.esk")',
    '(g3m-prefill-p1! generator input)',
    '(g3m-generate-p1-g1! reference reference-input)',
    '(g3r-generate-replayed-p1! replay)',
    '(g3m-prefill-p2! p2 p2-input)',
    '(g3r-generate-replayed-p1! p2)',
    '(g3r-generate-replayed-p1! exhausted)',
    '(g3r-generate-replayed-p1! retry)',
    '(g3r-generate-replayed-p1! retry)',
    '(same-output? retry-output retry-reference-output',
    '(= 1 (linked-history eos-replay-native 16))',
]
HISTORY_CLAIMS = [
    ('(= 2 (linked-history first-native 12))',
     '(= 1 (linked-history first-native 12))'),
    ('(= 1 (linked-history first-native 13))',
     '(= 0 (linked-history first-native 13))'),
    ('(= 1 (linked-history first-native 14))',
     '(= 0 (linked-history first-native 14))'),
    ('(= 0 (linked-history first-native 15))',
     '(= 1 (linked-history first-native 15))'),
    ('(check "prefix history length" (= 1 (linked-history replay-native 12)))',
     '(check "prefix history length" (= 0 (linked-history replay-native 12)))'),
    ('(check "prefix prompt length" (= 1 (linked-history replay-native 13)))',
     '(check "prefix prompt length" (= 0 (linked-history replay-native 13)))'),
    ('(check "prefix generated count" (= 0 (linked-history replay-native 14)))',
     '(check "prefix generated count" (= 1 (linked-history replay-native 14)))'),
    ('(check "prefix manual count" (= 0 (linked-history replay-native 15)))',
     '(check "prefix manual count" (= 1 (linked-history replay-native 15)))'),
    ('(check "prefix EOS terminal" (= 0 (linked-history replay-native 16)))',
     '(check "prefix EOS terminal" (= 1 (linked-history replay-native 16)))'),
    ('(= 0 (linked-history budget-native 15))',
     '(= 1 (linked-history budget-native 15))'),
    ('(= 0 (linked-history retry-native 15))',
     '(= 1 (linked-history retry-native 15))'),
]
PREFIX_CLAIMS = [
    '(check "prefix cache length" (= 1 (cache-length replay-native)))',
    '(check "prefix RNG version" (= 1 (rng-word replay-native 0)))',
    '(check "prefix RNG seed" (= 1729 (rng-word replay-native 1)))',
    '(check "prefix RNG counter low" (= 5 (rng-word replay-native 2)))',
    '(check "prefix RNG counter high" (= 0 (rng-word replay-native 3)))',
    '(check "budget-zero prompt-only manual count"',
    '(check "fault rollback prompt-only manual count"',
]
BRIDGE_NEEDLES = [
    "__wrap_et_kernel_runtime_dispatch",
    "__real_et_kernel_runtime_dispatch(runtime, call, error)",
    "memcmp(inputs[0].data, prefix_logits, sizeof(prefix_logits))",
    "memcmp(inputs[4].data, prefix_rng, sizeof(prefix_rng))",
    "et_g3t_test_manual_cache_kv_word_v1",
    "ET_KERNEL_ERROR_INTERNAL",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_runner(source: str) -> None:
    for needle in RUNNER_NEEDLES:
        require(needle in source, f"runner omitted {needle}")
    require(source.index("G3R_LINKED_RUNTIME_RELEASE") < source.index("verify_toolchain"),
            "release must precede toolchain execution")
    require(source.index("python3 \"$PROJECT_ROOT/scripts/check-g3r-replayed-prefix-linked.py\"")
            < source.index("verify_toolchain"), "source check must precede build")
    wrapper_start = '  cat >"$directory/cxx-wrap" <<WRAPPER'
    require(source.count(wrapper_start) == 1, "linker wrapper definition changed")
    wrapper = re.findall(r'(?m)^  cat >"\$directory/cxx-wrap" <<WRAPPER\n(.*?)^WRAPPER$',
                         source, re.DOTALL | re.MULTILINE)
    expected_wrapper = ('#!/usr/bin/env bash\n'
                        'exec $(printf \'%q\' "$cxx") '
                        '-Wl,--wrap=et_kernel_runtime_dispatch '
                        '$sanitizer_link "\\$@"\n')
    require(wrapper == [expected_wrapper], "linker wrapper command changed")
    build = re.findall(r'(?m)^build_mode\(\) \{\n(.*?)^\}\nbuild_mode normal$',
                       source, re.DOTALL | re.MULTILINE)
    require(len(build) == 1, "build_mode command structure changed")
    feature_flag = r'(?<!\S)-DET_G3R_REPLAYED_PREFIX_PRIVATE(?=\s|\\)'
    require(len(re.findall(feature_flag, build[0])) == 1,
            "exact replayed-prefix feature flag changed")
    expected_pass = ('  [[ ! -s "$directory/stderr" ]] || '
                     'die "$mode runtime stderr is nonempty"\n'
                     "  run_logged grep -E '^G3-R linked replayed P1 PASS: "
                     "checks=[1-9][0-9]*$' \\\n"
                     '    "$directory/stdout" >/dev/null || '
                     'die "$mode PASS line missing"\n')
    require(build[0].endswith(expected_pass), "logged PASS check command changed")
    require(build[0].count("grep -E") == 1, "PASS check command duplicated")
    require(sha256(build[0].encode()).hexdigest() == BUILD_MODE_SHA256,
            "build_mode command/receipt path changed")
    require(sha256(source.encode()).hexdigest() == RUNNER_SHA256,
            "runner command/receipt path changed")


def validate_fixture(source: str) -> None:
    for needle in FIXTURE_NEEDLES:
        require(needle in source, f"fixture omitted {needle}")
    for claim, _ in HISTORY_CLAIMS:
        require(source.count(claim) == 1, f"fixture history claim changed: {claim}")
    for claim in PREFIX_CLAIMS:
        require(source.count(claim) == 1, f"fixture prefix claim changed: {claim}")
    require(source.index('(load "g3t_final_publication_extension.esk")')
            < source.index('(load "g3t_text_result_extension.esk")'),
            "text result must load after final publication")


def validate_manifest(source: str, paths: list[str], root: Path,
                      overrides: dict[str, bytes] | None = None) -> None:
    lines = source.splitlines()
    expected_headers = [f"{key}\t{value}" for key, value in HEADERS]
    require(lines[:len(expected_headers)] == expected_headers, "toolchain provenance header changed")
    lock = dict(line.split("\t", 1) for line in (root / "toolchain/eshkol.lock").read_text().splitlines()
                if line and not line.startswith("#"))
    for key, value in HEADERS[:5]:
        require(lock.get(key) == value, f"toolchain lock drift: {key}")
    local = [path for path in paths if not path.startswith(".deps/")]
    sha_lines = lines[len(expected_headers):]
    require(len(sha_lines) == len(local), "manifest path omission or addition")
    for line, path in zip(sha_lines, local):
        content = (overrides or {}).get(path)
        digest = sha256(content if content is not None else (root / path).read_bytes()).hexdigest()
        require(line == f"source_sha256\t{digest}\t{path}", f"source hash/path drift: {path}")


def require_rejected_repin(runner: str, manifest: str, paths: list[str],
                           old: str, new: str) -> None:
    require(runner.count(old) == 1, f"mutation target changed: {old}")
    mutant = runner.replace(old, new, 1)
    path = "scripts/test-g3r-replayed-prefix-linked.sh"
    old_row = f"source_sha256\t{sha256(runner.encode()).hexdigest()}\t{path}"
    new_row = f"source_sha256\t{sha256(mutant.encode()).hexdigest()}\t{path}"
    require(manifest.count(old_row) == 1, "runner manifest row changed")
    repinned = manifest.replace(old_row, new_row, 1)
    validate_manifest(repinned, paths, ROOT, {path: mutant.encode()})
    try:
        validate_runner(mutant)
    except ValueError:
        return
    raise ValueError(f"repinned semantic mutation escaped checker: {old} -> {new}")


def require_rejected_fixture_repin(fixture: str, manifest: str,
                                   paths: list[str], old: str, new: str) -> None:
    require(fixture.count(old) == 1, f"fixture mutation target changed: {old}")
    mutant = fixture.replace(old, new, 1)
    path = "tests/g3r/replayed_prefix_linked_test.esk"
    old_row = f"source_sha256\t{sha256(fixture.encode()).hexdigest()}\t{path}"
    new_row = f"source_sha256\t{sha256(mutant.encode()).hexdigest()}\t{path}"
    require(manifest.count(old_row) == 1, "fixture manifest row changed")
    validate_manifest(manifest.replace(old_row, new_row, 1), paths, ROOT,
                      {path: mutant.encode()})
    try:
        validate_fixture(mutant)
    except ValueError:
        return
    raise ValueError(f"repinned history mutation escaped checker: {old} -> {new}")


def main() -> None:
    base = (ROOT / BASE).read_text().splitlines()
    closure = (ROOT / CLOSURE).read_text().splitlines()
    require(closure == base + ADDED, "ordered native/Eshkol closure changed")
    require(len(closure) == len(set(closure)), "duplicate source closure entry")
    for path in closure:
        if not path.startswith(".deps/"):
            require((ROOT / path).is_file(), f"missing source: {path}")
    runner = (ROOT / "scripts/test-g3r-replayed-prefix-linked.sh").read_text()
    fixture = (ROOT / "tests/g3r/replayed_prefix_linked_test.esk").read_text()
    bridge = (ROOT / "tests/g3r/replayed_prefix_linked_probe.c").read_text()
    manifest = (ROOT / MANIFEST).read_text()
    validate_runner(runner)
    validate_fixture(fixture)
    for needle in BRIDGE_NEEDLES:
        require(needle in bridge, f"bridge omitted {needle}")
    validate_manifest(manifest, closure, ROOT)
    fixture_path = "tests/g3r/replayed_prefix_linked_test.esk"
    load = '(load "g3t_text_result_extension.esk")\n'
    require(fixture.count(load) == 1, "text-result fixture load changed")
    missing_load = fixture.replace(load, "", 1)
    old_row = f"source_sha256\t{sha256(fixture.encode()).hexdigest()}\t{fixture_path}"
    new_row = f"source_sha256\t{sha256(missing_load.encode()).hexdigest()}\t{fixture_path}"
    require(manifest.count(old_row) == 1, "fixture manifest row changed")
    validate_manifest(manifest.replace(old_row, new_row, 1), closure, ROOT,
                      {fixture_path: missing_load.encode()})
    try:
        validate_fixture(missing_load)
    except ValueError:
        pass
    else:
        raise ValueError("repinned missing text-result load escaped checker")
    for old, new in HISTORY_CLAIMS:
        require_rejected_fixture_repin(fixture, manifest, closure, old, new)
    # Repin the changed runner SHA first, so these negatives prove command
    # semantics are checked independently of manifest drift.
    for old, new in (("-Wl,--wrap=et_kernel_runtime_dispatch",
                      "-Wl,--wrap=et_kernel_runtime_dispatch_broken"),
                     ("run_logged grep -E", "grep -E"),
                     ("-DET_G3R_REPLAYED_PREFIX_PRIVATE",
                      "-DET_G3R_REPLAYED_PREFIX_PRIVATE_BROKEN"),
                     ('  [[ ! -s "$directory/stderr" ]]',
                      '  return 0\n  [[ ! -s "$directory/stderr" ]]'),
                     ("-DET_G3R_REPLAYED_PREFIX_PRIVATE", ""),
                     ("replayed_prefix_linked_probe.c", "")):
        require_rejected_repin(runner, manifest, closure, old, new)
    lines = manifest.splitlines()
    try:
        validate_manifest("\n".join(lines[:-1]) + "\n", closure, ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("omitted manifest entry escaped checker")
    print("G3-R linked replayed P1 source/manifest gate: PASS")


if __name__ == "__main__":
    main()
