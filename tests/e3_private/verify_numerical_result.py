"""Join five actual private AOT transcripts to the frozen D1/D2/M3 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.e3_reference.corpus import CASES
from tests.e3_reference.join import validate_five_case_join
from tests.e3_reference.transcript import canonical, decode
from tests.e3_private.verify_layout_results import verify_results


ROLE_COUNT = 21
ROW_FIELDS = (
    "inputs", "targets", "mask", "logits_f32", "ce_f32",
    "reduction_f32", "correct_f32",
)


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate result field: {key}")
        result[key] = value
    return result


def _result(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_pairs)
    if not isinstance(value, dict) or canonical(value) != raw:
        raise ValueError(f"{path.name}: noncanonical boxed result")
    return value


def _roles(path: Path, rows: int) -> tuple[tuple[str, ...], ...]:
    raw = path.read_bytes()
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name}: roles are not ASCII") from error
    if raw != ("\n".join(lines) + "\n").encode("ascii") or \
            len(lines) != rows * ROLE_COUNT:
        raise ValueError(f"{path.name}: incomplete or noncanonical role rows")
    normalized = []
    for ordinal, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) != 4 or fields[0] != str(ordinal // ROLE_COUNT):
            raise ValueError(f"{path.name}: role ordinal or fields differ")
        normalized.append(tuple(fields[1:]))
    return tuple(normalized)


def verify_transcript_relations(observed: dict[str, dict[str, object]],
                                roles: dict[str, tuple[tuple[str, ...], ...]]) -> None:
    names = {case[0] for case in CASES}
    if set(observed) != names or set(roles) != names:
        raise ValueError("compiled transcript case set differs")
    single = observed["packed-single"]["batches"]
    sharded = observed["packed-sharded"]["batches"]
    suffix = observed["packed-suffix"]["batches"]
    if single != sharded or roles["packed-single"] != roles["packed-sharded"]:
        raise ValueError("compiled packed repartition rows or roles differ")
    if len(suffix) != len(single) - 1:
        raise ValueError("compiled packed suffix row count differs")
    for actual, prior in zip(suffix, single[1:]):
        if any(actual[field] != prior[field] for field in ROW_FIELDS):
            raise ValueError("compiled packed suffix row words differ")
    if roles["packed-suffix"] != roles["packed-single"][ROLE_COUNT:]:
        raise ValueError("compiled packed suffix role words differ")


def verify_files(corpus_root: Path, reference_path: Path, results_root: Path,
                 variant: str) -> None:
    corpus = decode((corpus_root / "corpus-reference.json").read_bytes())
    bundle = decode(reference_path.read_bytes())
    joined = validate_five_case_join(corpus_root, corpus, bundle)
    results = {}
    observed = {}
    roles = {}
    for case in joined:
        name = case["name"]
        results[name] = _result(results_root / f"result-{variant}-{name}.json")
        observed[name] = decode(
            (results_root / f"transcript-{variant}-{name}.json").read_bytes()
        )
        if observed[name].get("kind") != "observed-f32":
            raise ValueError(f"{name}: compiled transcript kind differs")
        roles[name] = _roles(
            results_root / f"roles-{variant}-{name}.tsv", len(case["rows"])
        )
    verify_results(joined, results)
    verify_transcript_relations(observed, roles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--variant", choices=("a", "b", "poison", "sanitized"),
                        required=True)
    args = parser.parse_args()
    verify_files(args.corpus_root, args.reference, args.results, args.variant)


if __name__ == "__main__":
    main()
