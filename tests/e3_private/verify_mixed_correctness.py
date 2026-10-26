"""Compare both genuine mixed-correctness observations with the independent M3 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.e3_private.parse_numerical_observation import ROLE_NAMES
from tests.e3_private.parse_mixed_correctness_observation import _unique, parse_mixed
from tests.e3_reference.mixed_correctness import validate_materialized
from tests.e3_reference.transcript import (
    F32_TOLERANCE, bits_f32, canonical, decode, validate_mathematical_case,
    validate_observed_case,
)


NAMES = ("matched-correct", "mixed")


def _canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(value, dict) or canonical(value) != raw:
        raise ValueError("mixed-correctness record is not canonical")
    return value


def _paired_witness(observations: list[dict[str, object]]) -> list[str]:
    first, second = (item["batches"][0] for item in observations)
    source_ce = (("40af07af", "40aeb52c"),
                 ("40af07af", "40b0b72f"))
    for item, expected in zip((first, second), source_ce):
        for actual_word, source_word in zip(item["ce_f32"], expected):
            actual = bits_f32(actual_word, "mixed observed CE")[1]
            source = bits_f32(source_word, "mixed Q0 CE")[1]
            if abs(actual - source) > F32_TOLERANCE[0] + F32_TOLERANCE[1] * abs(source):
                raise ValueError("mixed-correctness CE exceeds source-oracle tolerance")
    first_second = bits_f32(first["ce_f32"][1], "matched CE2")[1]
    mixed_second = bits_f32(second["ce_f32"][1], "mixed CE2")[1]
    if (first["inputs"] != second["inputs"]
            or first["mask"] != second["mask"]
            or first["targets"] != [14, 31]
            or second["targets"] != [14, 0]
            or first["logits_f32"] != second["logits_f32"]
            or first["ce_f32"][0] != second["ce_f32"][0]
            or not mixed_second - first_second > 0.05
            or first["correct_f32"] != "40000000"
            or second["correct_f32"] != "3f800000"
            or first["reduction_f32"][1] != "40000000"
            or second["reduction_f32"][1] != "40000000"
            or first["state_f32"][1:] != ["40000000", "40000000"]
            or second["state_f32"][1:] != ["40000000", "3f800000"]
            or observations[0]["final"]["counters"] != [2, 1]
            or observations[1]["final"]["counters"] != [2, 1]
            or observations[0]["final"]["metrics_f32"][1] != "40000000"
            or observations[1]["final"]["metrics_f32"][1] != "40000000"
            or observations[0]["final"]["metrics_f32"][3] != "3f800000"
            or observations[1]["final"]["metrics_f32"][3] != "3f000000"):
        raise ValueError("mixed-correctness paired-logit and CE witness differs")
    return [first["ce_f32"][1], second["ce_f32"][1]]


def verify(reference: dict[str, object], manifest: dict[str, object],
           observations: list[dict[str, object]], roles_paths: list[Path],
           results: list[dict[str, object]], stdout_paths: list[Path],
           manifest_sha256: str) -> dict[str, object]:
    from tests.e3_native_parity.verify_observation import _f32, _f64, _roles
    from tests.e3_reference.reference import validate_bundle

    validate_bundle(reference)
    if [case["name"] for case in reference["cases"]] != list(NAMES):
        raise ValueError("mixed-correctness reference case inventory differs")
    if (len(observations) != 2 or len(roles_paths) != 2
            or len(results) != 2 or len(stdout_paths) != 2):
        raise ValueError("mixed-correctness observations require exactly two cases")
    maximum_role_error = 0.0
    for index, name in enumerate(NAMES):
        case = reference["cases"][index]
        corpus = manifest["cases"][index]
        observed, result = observations[index], results[index]
        parsed_observed, parsed_roles, parsed_result = parse_mixed(
            stdout_paths[index].read_bytes(), manifest, name, manifest_sha256,
        )
        if (observed != parsed_observed or result != parsed_result
                or roles_paths[index].read_bytes() != parsed_roles):
            raise ValueError("mixed-correctness records differ from the raw compiled stream")
        if case["corpus"] != corpus:
            raise ValueError("mixed-correctness reference corpus identity differs")
        if observed.get("kind") != "observed-f32":
            raise ValueError("mixed-correctness record is not an observed frame")
        expected = tuple(corpus["batches"])
        validate_observed_case(observed, expected)
        validate_mathematical_case(
            observed, case["mathematical"], expected,
            case["mathematical_loss_f64"],
        )
        if (set(result) != {"kind", "case", "outputs_u32", "cursor_before",
                            "cursor_after", "provenance"}
                or result["kind"] != "e3-mixed-correctness-result-v1"
                or result["case"] != name
                or result["cursor_before"] != corpus["start_cursor"]
                or result["cursor_after"] != corpus["start_cursor"]
                or not isinstance(result["provenance"], dict)
                or set(result["provenance"]) != {"corpora_sha256", "stdout_sha256"}
                or result["provenance"]["corpora_sha256"] != manifest_sha256
                or result["provenance"]["stdout_sha256"]
                != hashlib.sha256(stdout_paths[index].read_bytes()).hexdigest()):
            raise ValueError("mixed-correctness result identity or cursor differs")
        final = observed["final"]
        if result["outputs_u32"] != (
            [int(word, 16) for word in final["metrics_f32"]] + final["counters"]
        ):
            raise ValueError("mixed-correctness result differs from observed final")
        roles = _roles(roles_paths[index])
        if len(roles) != len(ROLE_NAMES):
            raise ValueError("mixed-correctness role count differs")
        for ordinal, (actual, expected_role) in enumerate(
            zip(roles, case["mathematical"][0]["roles"])
        ):
            batch_ordinal, role_name, shape, words = actual
            if (batch_ordinal != 0 or role_name != ROLE_NAMES[ordinal]
                    or role_name != expected_role["name"]
                    or list(shape) != expected_role["shape"]):
                raise ValueError("mixed-correctness role identity or shape differs")
            reference_values = [_f64(word) for word in expected_role["values_f64"]]
            actual_values = [_f32(word) for word in words]
            delta = max(abs(left - right)
                        for left, right in zip(actual_values, reference_values))
            scale = max(abs(value) for value in reference_values)
            if delta > 1.0e-10 + 3.0e-4 * scale:
                raise ValueError(f"mixed-correctness role {role_name} exceeds M3 tolerance")
            maximum_role_error = max(maximum_role_error, delta)
            if role_name == "Z" and list(words) != observed["batches"][0]["logits_f32"]:
                raise ValueError("mixed-correctness logits and Z role differ")
    return {"kind": "e3-mixed-correctness-comparison-v1",
            "cases": list(NAMES), "maximum_role_error": maximum_role_error,
            "second_ce_f32": _paired_witness(observations)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    for name in NAMES:
        parser.add_argument(f"--stdout-{name}", required=True, type=Path)
        parser.add_argument(f"--observed-{name}", required=True, type=Path)
        parser.add_argument(f"--roles-{name}", required=True, type=Path)
        parser.add_argument(f"--result-{name}", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    manifest_raw = (args.corpus / "corpora.json").read_bytes()
    manifest = _canonical_json(args.corpus / "corpora.json")
    validate_materialized(args.corpus, manifest)
    reference_raw = args.reference.read_bytes()
    reference = decode(reference_raw)
    report = verify(
        reference, manifest,
        [_canonical_json(getattr(args, f"observed_{name.replace('-', '_')}")) for name in NAMES],
        [getattr(args, f"roles_{name.replace('-', '_')}") for name in NAMES],
        [_canonical_json(getattr(args, f"result_{name.replace('-', '_')}")) for name in NAMES],
        [getattr(args, f"stdout_{name.replace('-', '_')}") for name in NAMES],
        hashlib.sha256(manifest_raw).hexdigest(),
    )
    report["corpora_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    report["reference_sha256"] = hashlib.sha256(reference_raw).hexdigest()
    with args.report.open("xb") as stream:
        stream.write(canonical(report))


if __name__ == "__main__":
    main()
