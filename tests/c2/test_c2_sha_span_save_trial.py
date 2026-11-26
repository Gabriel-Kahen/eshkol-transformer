#!/usr/bin/env python3
"""Stage the same C2 joint SAVE gate with a private SHA selector."""

import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.c2.test_c2_sha_span_composition import check_parens, replace_once
from tests.c2.test_checkpoint_format import C1_DOMAIN, C1_TENSOR_DOMAIN

SELECTORS = {
    "whole": "c2-model-encode-c2-owned-state-internal",
    "spans": "c2-model-encode-c2-owned-state-spans-experiment-internal",
}


def stage(directory: Path, selector: str) -> tuple[Path, Path]:
    if selector not in SELECTORS:
        raise SystemExit("selector must be whole or spans")
    directory.mkdir(parents=True, exist_ok=False)
    runtime = (ROOT / "tests/c2/c2_checkpoint_operational_runtime.esk").read_text()
    runtime = replace_once(
        runtime, '(load "c2_wave2_root.esk")',
        '(load "c2_wave2_root.esk")\n'
        '(load "c2_sha_span_composition_extension.esk")\n'
        '(set! c2-save-model-encode-internal\n'
        f'  {SELECTORS[selector]})',
    )
    check_parens(runtime)
    staged_runtime = directory / "runtime.esk"
    staged_runtime.write_text(runtime)

    gate = (ROOT / "scripts/test-c2-checkpoint-operational.sh").read_text()
    gate = replace_once(
        gate,
        'source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"',
        f'source "{ROOT}/scripts/common.sh"',
    )
    gate = replace_once(
        gate,
        '  "${cc}" "${runtime_cflags[@]}" -c '
        '"${PROJECT_ROOT}/native/${source}.c" \\\n',
        '  extra=()\n'
        '  if [[ "$source" == c2_checkpoint_format ]]; then\n'
        '    extra=(-DET_C2_SHA_SPANS_EXPERIMENT)\n'
        '  fi\n'
        '  "${cc}" "${runtime_cflags[@]}" "${extra[@]}" -c '
        '"${PROJECT_ROOT}/native/${source}.c" \\\n',
    )
    gate = replace_once(
        gate, '        -I "${PROJECT_ROOT}/native" -L "${runtime}" \\\n',
        '        -I "${PROJECT_ROOT}/tests/c2" \\\n'
        '        -I "${PROJECT_ROOT}/native" -L "${runtime}" \\\n',
    )
    gate = replace_once(
        gate,
        '      "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/lib" "${PROJECT_ROOT}/native"',
        '      "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/lib" \\\n'
        '      "${PROJECT_ROOT}/tests/c2" "${PROJECT_ROOT}/native"',
    )
    original = '"${PROJECT_ROOT}/tests/c2/c2_checkpoint_operational_runtime.esk"'
    if gate.count(original) != 2:
        raise SystemExit("operational AOT/manifest source anchors changed")
    gate = gate.replace(original, '"${C2_SPAN_TRIAL_RUNTIME}"')
    staged_gate = directory / "gate.sh"
    staged_gate.write_text(gate)
    subprocess.run(["bash", "-n", str(staged_gate)], check=True)
    return staged_runtime, staged_gate


def check_pair() -> None:
    save = (ROOT / "native/c2_checkpoint_save_extension.esk").read_text()
    extension = (ROOT / "tests/c2/c2_sha_span_composition_extension.esk").read_text()
    native = (ROOT / "native/c2_checkpoint_format.c").read_text()
    if (save.count("(define c2-save-model-encode-internal\n"
                   "  c2-model-encode-c2-owned-state-internal)") != 1
            or save.count("(c2-save-model-encode-internal\n") != 1
            or "et_c2_private_model_sha256_spans_experiment_v1" not in native
            or SELECTORS["spans"] not in extension):
        raise SystemExit("private SAVE/SHA selector source contract changed")
    with tempfile.TemporaryDirectory(prefix="c2-span-save-check.") as temp:
        whole, whole_gate = stage(Path(temp) / "whole", "whole")
        spans, spans_gate = stage(Path(temp) / "spans", "spans")
        if whole_gate.read_bytes() != spans_gate.read_bytes():
            raise SystemExit("arm gate scripts differ")
        normalized = [
            path.read_text().replace(name, "SELECTED_MODEL_ENCODER")
            for path, name in ((whole, SELECTORS["whole"]),
                               (spans, SELECTORS["spans"]))
        ]
        if normalized[0] != normalized[1]:
            raise SystemExit("arm runtime sources differ beyond selector")
    print("C2 span SAVE trial source/recipe: PASS (no compile)")


def estimate_preimages(path: Path) -> None:
    """Count the C1 concat outputs avoided by six joint model builds."""
    artifact = path.read_bytes()
    u32 = lambda data, at: struct.unpack_from("<I", data, at)[0]
    u64 = lambda data, at: struct.unpack_from("<Q", data, at)[0]
    model_at, model_bytes = u64(artifact, 64), u64(artifact, 96)
    model = artifact[model_at:model_at + model_bytes]
    if len(artifact) != 16777216 or len(model) != model_bytes:
        raise SystemExit("expected exact 16 MiB joint fixture")
    count = u32(model, 80)
    cursor = u64(model, 48) + u32(model, 88)
    tensor_preimage_bytes = payload_bytes = 0
    for _ in range(count):
        record_bytes = u64(model, cursor)
        tensor_bytes = u64(model, cursor + 16)
        tensor_preimage_bytes += len(C1_TENSOR_DOMAIN) + record_bytes + tensor_bytes
        payload_bytes += tensor_bytes
        cursor += record_bytes
    if (count != 62 or cursor != u64(model, 48) + u64(model, 56)
            or payload_bytes != u64(model, 72)):
        raise SystemExit("joint C1 tensor layout differs from trial contract")
    container_preimage_bytes = len(C1_DOMAIN) + model_bytes - 32
    print(f"model_bytes={model_bytes}")
    print(f"tensor_count={count}")
    print(f"tensor_preimage_bytes_per_model={tensor_preimage_bytes}")
    print(f"container_preimage_bytes_per_model={container_preimage_bytes}")
    print(f"preimage_allocations_skipped_six_models={6 * (count + 1)}")
    print(f"preimage_bytes_skipped_six_models={6 * (tensor_preimage_bytes + container_preimage_bytes)}")


def normalized_stdout(path: Path) -> bytes:
    lines = path.read_text(encoding="ascii").splitlines()
    prefixes = ("C2 OPERATIONAL MEASURE:", "C2 OPERATIONAL PHASE:",
                "C2 OPERATIONAL RUNTIME PASS:")
    if len(lines) != 3 or any(not line.startswith(prefix)
                              for line, prefix in zip(lines, prefixes)):
        raise SystemExit("joint stdout marker schema changed")
    normalized = []
    for line in lines:
        fields = []
        for field in line.split():
            key, sep, value = field.partition("=")
            if sep and ("-kib" in key or key == "elapsed-ms"
                        or key.endswith("arena-delta")):
                if not value.isdecimal():
                    raise SystemExit("joint stdout has a malformed measurement")
                field = key + "=<measured>"
            fields.append(field)
        normalized.append(" ".join(fields))
    return ("\n".join(normalized) + "\n").encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--fixture", type=Path)
    action.add_argument("--compare-stdout", nargs=2, type=Path,
                        metavar=("WHOLE", "SPANS"))
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--selector", choices=tuple(SELECTORS))
    args = parser.parse_args()
    if args.compare_stdout:
        if args.stage or args.selector:
            parser.error("--compare-stdout takes no stage or selector")
        whole, spans = map(normalized_stdout, args.compare_stdout)
        if whole != spans:
            raise SystemExit("joint stdout differs beyond measured fields")
        sys.stdout.buffer.write(whole)
    elif args.fixture:
        if args.stage or args.selector:
            parser.error("--fixture takes no stage or selector")
        estimate_preimages(args.fixture)
    elif args.check:
        if args.stage or args.selector:
            parser.error("--check takes no stage or selector")
        check_pair()
    else:
        if args.stage is None or args.selector is None:
            parser.error("--stage and --selector are required")
        stage(args.stage, args.selector)


if __name__ == "__main__":
    main()
