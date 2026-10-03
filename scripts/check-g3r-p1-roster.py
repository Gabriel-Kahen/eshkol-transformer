#!/usr/bin/env python3
"""Check the private candidate roster stays outside ordinary P1 products."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "native/p1_identity.c").read_text()
header = (root / "native/p1_identity_internal.h").read_text()
test = (root / "tests/p1/test_g3r_candidate_roster.c").read_text()
runner = (root / "scripts/test-g3r-p1-roster.sh").read_text()

def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(f"G3-R P1 ROSTER STRUCTURE FAIL: {message}")

flag = "ET_G3R_CANDIDATE_RETIRE_PRIVATE"
require(flag in source and flag in header and flag in test, "feature guard missing")
for name in (
    "et_p1_private_candidate_construction_begin_v1",
    "et_p1_private_candidate_graph_preflight_v1",
    "et_p1_private_candidate_graph_revoke_v1",
):
    require(source.count(name) == 1 and header.count(name) == 1,
            f"missing or duplicate private entry: {name}")
require("record->candidate_construction != ledger" in source and
        "record->candidate_index != i" in source and
        source.index("!p1_record_registered(record)") <
        source.index("record->candidate_construction != ledger"),
        "roster records are read before registry-first admission")
require("ledger->entries != ledger->candidate_roster_base" in source and
        "ledger->count != ledger->sealed_count" in source,
        "roster extent or identity check missing")
require("ET_G3R_CANDIDATE_RETIRE_PRIVATE" not in
        (root / "native/p1_identity_public_symbols.txt").read_text(),
        "public manifest mentions candidate feature")
for name in (
    "et_p1_private_candidate_construction_begin_v1",
    "et_p1_private_candidate_graph_preflight_v1",
    "et_p1_private_candidate_graph_revoke_v1",
):
    require(name not in
            (root / "native/p1_identity_trusted_symbols.txt").read_text(),
            "ordinary trusted manifest includes candidate symbol")

def runner_valid(text: str) -> bool:
    required = {
        "(( $# <= 1 ))": 1,
        "if (( $# == 1 )); then": 1,
        '[[ "${temporary}" == /* && ! -e "${temporary}" && ! -L "${temporary}" ]]': 1,
        'mkdir -- "${temporary}"': 1,
        'trap \'rm -rf -- "${temporary}"\' EXIT': 1,
        'for role in public trusted; do': 1,
        'for run in a b; do': 1,
        'baseline=90f3c42025556596a206b125138de49e09e3f7d3': 1,
        '1fc04e8299610d5fe508d44c888f1f4b1c9a4021': 1,
        'cmp "${temporary}/baseline-${role}.o" "${temporary}/current-${role}.o"': 1,
        'p1_identity_${role}_symbols.txt': 1,
        'cmp "${temporary}/expected-candidate-symbols" "${temporary}/candidate-symbols"': 1,
        "grep -F 'requires the trusted P1 build'": 1,
        'cmp "${temporary}/run-a.stdout" "${temporary}/run-b.stdout"': 1,
        'cmp "${temporary}/run-a.stdout" "${temporary}/san.stdout"': 1,
        '-fsanitize=address,undefined': 1,
        'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1': 1,
        '-Wl,--wrap=calloc': 2,
        'forbidden-public.exit': 2,
        'forbidden-public.stderr': 2,
        'candidate-symbols': 4,
        'current-${role}.compile.stderr': 1,
        'baseline-${role}.compile.stderr': 1,
        'roster-${run}.compile.stderr': 1,
        'roster-sanitized.compile.stderr': 1,
        'run-${run}.stderr': 2,
        'run-${run}.exit': 1,
        'san.stderr': 2,
        'san.exit': 1,
        'status=$?': 2,
        'set +e': 2,
        'printf \'%d\\n\' "${status}"': 2,
        '[[ "${status}" -eq 0 ]] || die': 2,
        'compiler-provenance.tsv': 1,
        'compiler-selected.tsv': 1,
        'repeat-cmp.stdout': 1,
        'san-cmp.stdout': 1,
    }
    if any(text.count(fragment) != count for fragment, count in required.items()):
        return False
    fresh = text.index('[[ "${temporary}" == /*')
    mkdir = text.index('mkdir -- "${temporary}"')
    default_branch = text.index("\nelse\n", mkdir)
    cleanup = text.index('trap \'rm -rf -- "${temporary}"\' EXIT')
    branch_end = text.index("\nfi\n", cleanup)
    verify = text.index('verify_toolchain >"${temporary}/toolchain.stdout"')
    normal = text[text.index('for run in a b; do'):
                  text.index('\ndone', text.index('for run in a b; do'))]
    sanitize = text[text.index('"${cc}" "${flags[@]}" -fsanitize=address,undefined'):]

    def ordered(block: str, fragments: tuple[str, ...]) -> bool:
        try:
            positions = [block.index(fragment) for fragment in fragments]
        except ValueError:
            return False
        return positions == sorted(positions) and len(set(positions)) == len(positions)

    normal_checked = ordered(normal, (
        'set +e', 'timeout --foreground', 'status=$?', 'set -e',
        'printf \'%d\\n\' "${status}" >"${temporary}/run-${run}.exit"',
        '[[ "${status}" -eq 0 ]] || die "native roster ${run} exited ${status}"',
        '[[ ! -s "${temporary}/run-${run}.stderr" ]]',
    ))
    sanitize_checked = ordered(sanitize, (
        'set +e', 'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1',
        'timeout --foreground', 'status=$?', 'set -e',
        'printf \'%d\\n\' "${status}" >"${temporary}/san.exit"',
        '[[ "${status}" -eq 0 ]] || die "sanitized native roster exited ${status}"',
        'cmp "${temporary}/run-a.stdout" "${temporary}/san.stdout"',
    ))
    return (fresh < mkdir < default_branch < cleanup < branch_end < verify
            and text.count("rm -rf --") == 1 and normal_checked
            and sanitize_checked)


require(runner_valid(runner), "runner freshness, retention, or gate mode narrowed")
for mutation in (
    '! -e "${temporary}"',
    'mkdir -- "${temporary}"',
    'trap \'rm -rf -- "${temporary}"\' EXIT',
    'for role in public trusted; do',
    'for run in a b; do',
    'cmp "${temporary}/baseline-${role}.o" "${temporary}/current-${role}.o"',
    'cmp "${temporary}/run-a.stdout" "${temporary}/san.stdout"',
    '-fsanitize=address,undefined',
    'forbidden-public.exit',
    'roster-sanitized.compile.stderr',
    'run-${run}.exit',
    'san.exit',
    'status=$?',
    '[[ "${status}" -eq 0 ]] || die "native roster ${run} exited ${status}"',
    '[[ "${status}" -eq 0 ]] || die "sanitized native roster exited ${status}"',
):
    require(not runner_valid(runner.replace(mutation, "", 1)),
            f"runner mutation escaped admission: {mutation}")
print("G3-R P1 ROSTER STRUCTURE PASS: private guard, registry-first admission, retained runner evidence")
