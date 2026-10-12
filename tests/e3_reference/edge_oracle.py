"""Pinned development-only M3/CE oracle for the N5 one-valid-final corpus."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import platform

from tests.e3_reference.edge_corpus import validate
from tests.e3_reference.transcript import canonical, validate_observed_case


def build(root: Path, manifest: dict[str, object]) -> dict[str, object]:
    validate(root, manifest)
    # Imported only in the Q0 development environment; no training/runtime path
    # depends on Python or PyTorch.
    from tests.e3_reference.reference import _synthetic_case
    from tests.m3 import reference as m3

    if platform.python_version() != "3.14.6" or m3.torch.__version__ != "2.13.0+cpu":
        raise RuntimeError("N5 oracle requires pinned Q0 Python/PyTorch")
    m3.setup()
    parameters, _ = m3.parameters(1729)
    case = manifest["cases"][3]
    synthetic, mathematical, _ = _synthetic_case(case["batches"], parameters)
    validate_observed_case(synthetic, tuple(case["batches"]))
    if [batch["mask"] for batch in case["batches"]] != [[True, True], [True, False]]:
        raise AssertionError("one-valid-final D2 masks changed")
    last = synthetic["batches"][-1]
    result = {
        "kind": "e3-edge-oracle-v1",
        "case": case["name"],
        "cursor": case["start_cursor"],
        "inputs": last["inputs"], "targets": last["targets"],
        "mask": last["mask"],
        "logits_f32": last["logits_f32"],
        "ce_f32": last["ce_f32"],
        "reduction_f32": last["reduction_f32"],
        "state_f32": last["state_f32"],
        "final_f32": synthetic["final"]["metrics_f32"],
        "counters": synthetic["final"]["counters"],
        "model_seed": 1729,
        "corpus_sha256": hashlib.sha256((root / "edge-corpus.json").read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": m3.torch.__version__,
    }
    if len(result["logits_f32"]) != 512 or result["counters"] != [3, 2]:
        raise AssertionError("one-valid-final numerical shape/count changed")
    if len(mathematical) != 2:
        raise AssertionError("mathematical batch count changed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import json
    raw = (args.corpus / "edge-corpus.json").read_bytes()
    manifest = json.loads(raw)
    if canonical(manifest) != raw:
        raise ValueError("noncanonical edge corpus manifest")
    oracle = build(args.corpus, manifest)
    with args.output.open("xb") as stream:
        stream.write(canonical(oracle))
