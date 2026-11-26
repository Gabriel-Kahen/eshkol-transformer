#!/usr/bin/env python3
"""Stage one opt-in C2 span-hash AOT witness from the accepted model gate."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise SystemExit(f"composition recipe anchor changed: {before[:72]}")
    return source.replace(before, after, 1)


def check_parens(source: str) -> None:
    depth = 0
    quoted = escaped = comment = False
    for char in source:
        if comment:
            if char == "\n":
                comment = False
        elif quoted:
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
            if depth < 0:
                raise SystemExit("Eshkol source closes a form early")
    if depth != 0 or quoted:
        raise SystemExit("Eshkol source has an unclosed form or string")


def stage(directory: Path) -> Path:
    runtime = (ROOT / "tests/c2/c2_model_encode_runtime.esk").read_text()
    suffix = (ROOT / "tests/c2/c2_sha_span_composition_suffix.esk").read_text()
    runtime = replace_once(
        runtime, '(load "c2_model_encode_extension.esk")',
        '(load "c2_model_encode_extension.esk")\n'
        '(load "c2_sha_span_composition_extension.esk")',
    )
    runtime = replace_once(
        runtime, '(check "live model state releases" (state-dict-release! state))',
        suffix + '\n(check "live model state releases" (state-dict-release! state))',
    )
    check_parens(runtime)
    staged_runtime = directory / "span-runtime.esk"
    staged_runtime.write_text(runtime)

    gate = (ROOT / "scripts/test-c2-model-encode.sh").read_text()
    gate = replace_once(
        gate,
        'source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"',
        f'source "{ROOT}/scripts/common.sh"',
    )
    gate = replace_once(
        gate,
        '  "${cc}" "${cflags[@]}" -c "${PROJECT_ROOT}/native/${source}.c" \\\n'
        '    -o "${runtime}/${source}.o"',
        '  extra=()\n'
        '  if [[ "$source" == c2_checkpoint_format ]]; then\n'
        '    extra=(-DET_C2_SHA_SPANS_EXPERIMENT)\n'
        '  fi\n'
        '  "${cc}" "${cflags[@]}" "${extra[@]}" -c '
        '"${PROJECT_ROOT}/native/${source}.c" \\\n'
        '    -o "${runtime}/${source}.o"',
    )
    gate = replace_once(
        gate, '        -I "${PROJECT_ROOT}/native" -L "${runtime}" \\\n',
        '        -I "${PROJECT_ROOT}/tests/c2" \\\n'
        '        -I "${PROJECT_ROOT}/native" -L "${runtime}" \\\n',
    )
    gate = replace_once(
        gate, '"${PROJECT_ROOT}/tests/c2/c2_model_encode_runtime.esk"',
        '"${C2_SHA_SPAN_STAGED_RUNTIME}"',
    )
    cut = 'compile_probe clang-a "${cxx}"\n'
    if gate.count(cut) != 1:
        raise SystemExit("composition recipe gate tail changed")
    gate = gate.split(cut, 1)[0] + (
        'compile_runtime span "${cxx}"\n'
        'printf "C2 SHA span composition: focused AOT PASS\\n"\n'
    )
    staged_gate = directory / "gate.sh"
    staged_gate.write_text(gate)
    subprocess.run(["bash", "-n", str(staged_gate)], check=True)
    return staged_runtime, staged_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="stage and lint only")
    args = parser.parse_args()
    c1 = (ROOT / "internal/c1/lib/transformer/checkpoint_internal.esk").read_text()
    adapter = (ROOT / "native/c2_model_encode_extension.esk").read_text()
    suffix = (ROOT / "tests/c2/c2_sha_span_composition_suffix.esk").read_text()
    extension = (ROOT / "tests/c2/c2_sha_span_composition_extension.esk").read_text()
    for source in (c1, extension, suffix):
        check_parens(source)
    assert c1.count("(c1-encode-digest-parts") == 3
    assert "(c1-encode-digest sha (c1-concat parts) operation)" in c1
    assert "(not (or (procedure? sha) (c1-private-span-sha? sha)))" in c1
    assert "c2-model-checkpoint-sha256 operation" in adapter
    assert "et_c2_private_model_sha256_spans_experiment_v1" in extension
    expected = re.findall(r"\(define c1-(?:container|tensor)-domain\s*"
                          r"\(bytevector ([0-9\s]+)\)\)", c1)
    observed = re.findall(r"\(bytevector ([0-9\s]+)\)", suffix)
    assert len(expected) == len(observed) == 2
    assert [tuple(map(int, value.split())) for value in expected] == [
        tuple(map(int, value.split())) for value in observed]
    marker = "(define (c1-encode-digest sha bytes operation)"
    current = c1.split(marker, 1)[1].split(";;; A C2-private tagged callback", 1)[0]
    assert hashlib.sha256(current.encode()).hexdigest() == (
        "cb32ad5985ccc908d4029eba4cca334f498dc2b500c4632f5707234ea1c2e3fd"
    ), "ordinary one-argument SHA callback changed"
    scratch_parent = None if args.check else ROOT / ".deps"
    if scratch_parent is not None:
        scratch_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c2-sha-span-composition.",
                                     dir=scratch_parent) as temporary:
        runtime, gate = stage(Path(temporary))
        if args.check:
            print("C2 SHA span composition source/recipe: PASS (no compile)")
            return
        environment = dict(os.environ, C2_SHA_SPAN_STAGED_RUNTIME=str(runtime))
        subprocess.run(["bash", str(gate)], env=environment, check=True)


if __name__ == "__main__":
    main()
