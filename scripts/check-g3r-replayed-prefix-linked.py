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
    expected_pass = ('  [[ ! -s "$directory/stderr" ]] || '
                     'die "$mode runtime stderr is nonempty"\n'
                     "  run_logged grep -E '^G3-R linked replayed P1 PASS: "
                     "checks=[1-9][0-9]*$' \\\n"
                     '    "$directory/stdout" >/dev/null || '
                     'die "$mode PASS line missing"\n')
    require(build[0].endswith(expected_pass), "logged PASS check command changed")
    require(build[0].count("grep -E") == 1, "PASS check command duplicated")


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
    for needle in FIXTURE_NEEDLES:
        require(needle in fixture, f"fixture omitted {needle}")
    for needle in BRIDGE_NEEDLES:
        require(needle in bridge, f"bridge omitted {needle}")
    validate_manifest(manifest, closure, ROOT)
    # Repin the changed runner SHA first, so these negatives prove command
    # semantics are checked independently of manifest drift.
    for old, new in (("-Wl,--wrap=et_kernel_runtime_dispatch",
                      "-Wl,--wrap=et_kernel_runtime_dispatch_broken"),
                     ("run_logged grep -E", "grep -E"),
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
