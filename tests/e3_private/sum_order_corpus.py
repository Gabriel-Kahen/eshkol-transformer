"""Materialize the frozen genuine sum-order D1 resource at a fresh path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.d2.reference import ReferenceConfig, ReferenceDataset
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, _reject_symlink_ancestors
from tests.e3_reference.transcript import canonical
from tests.t1.reference import encode_artifact, encode_bytes


FIXTURE = Path(__file__).resolve().parents[1] / "e3_reference/sum_order_witness_v1.json"


def materialize(output: Path) -> dict[str, object]:
    fixture = json.loads(FIXTURE.read_bytes())
    search, corpus = fixture["search"], fixture["corpus"]
    index = search["first_witness_index"]
    domain = search["domain"].encode("ascii")
    if (fixture["kind"] != "genuine-m3-ce-sum-order" or fixture["version"] != 1
            or index != 2 or search["maximum_candidates"] != 64
            or domain != b"e3-section9-sum-order-search-v1"):
        raise ValueError("frozen search identity differs")
    raw = bytes.fromhex(corpus["source_hex"])
    if (len(raw) != 6 or raw != hashlib.sha256(domain + index.to_bytes(4, "little")).digest()[:6]
            or hashlib.sha256(raw).hexdigest() != corpus["source_sha256"]
            or encode_bytes(IDENTITY, raw) != tuple(raw)
            or hashlib.sha256(encode_artifact(IDENTITY)).hexdigest()
            != corpus["tokenizer_sha256"]):
        raise ValueError("frozen raw V256 source identity differs")
    _reject_symlink_ancestors(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("sum-order D1 output must be fresh")
    output.mkdir(parents=True)
    manifest = write_d1_resource(
        output, (tuple(raw),), fingerprint=FINGERPRINT, vocab=256,
    )
    loaded, digest = load_d1_resource(
        output, expected_fingerprint=FINGERPRINT, expected_vocab=256,
        maximum_total_tokens=6,
    )
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in output.iterdir()}
    if (loaded != (tuple(raw),) or manifest[-32:] != digest
            or digest.hex() != corpus["d1_manifest_trailer_sha256"]
            or files != corpus["d1_files_sha256"]):
        raise ValueError("physical D1 resource differs from frozen hashes")
    dataset = ReferenceDataset(
        loaded, ReferenceConfig(1, 2, None, 1, True),
        manifest_digest=digest, tokenizer_fingerprint=FINGERPRINT, vocab_size=256,
    )
    rows = []
    while (batch := dataset.next_batch()) is not None:
        rows.append({"inputs": list(batch.inputs[0]),
                     "targets": list(batch.targets[0]),
                     "mask": [int(bit) for bit in batch.loss_mask[0]]})
    if (len(rows) != 3 or hashlib.sha256(canonical(rows)).hexdigest()
            != corpus["d2_rows_sha256"]
            or rows != [{key: batch[key] for key in ("inputs", "targets", "mask")}
                        for batch in fixture["batches"]]):
        raise ValueError("packed D2 rows differ from frozen witness")
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    materialize(args.output)
    print("E3-SUM-ORDER-CORPUS-PASS")


if __name__ == "__main__":
    main()
