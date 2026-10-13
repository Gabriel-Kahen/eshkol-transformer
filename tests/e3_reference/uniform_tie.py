"""Development-only D1/D2 and C1 zero-state fixture for the E3 tie witness."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import platform

from tests.c1.test_checkpoint_format import build_container, parse_golden
from tests.d1.reference_format import parse_corpus
from tests.d2.reference import ReferenceConfig, ReferenceDataset, decode_reference_cursor
from tests.d2.test_resources import load_d1_resource, write_d1_resource
from tests.e3_reference.corpus import FINGERPRINT, IDENTITY, PROFILE, VOCAB_SIZE
from tests.e3_reference.transcript import canonical, encode, f32_bits
from tests.t1.reference import encode_artifact, encode_bytes


ROOT = Path(__file__).resolve().parents[2]
RAW = bytes(6)
NAME = "uniform-zero"
PROVIDER = b"i2-dense-cpu-f32-v1"
KIND = "e3-uniform-tie-corpus-v1"
CONFIG = ReferenceConfig(1, 2, None, 1, True)
# The M3 parameter contract is deliberately repeated here so a future change
# to the mathematical reference cannot silently change the C1 fixture.
SHAPES = (
    ("blocks/0/attention/key/weight", (4, 4)),
    ("blocks/0/attention/output/weight", (4, 4)),
    ("blocks/0/attention/query/weight", (4, 4)),
    ("blocks/0/attention/value/weight", (4, 4)),
    ("blocks/0/ffn/down/weight", (4, 8)),
    ("blocks/0/ffn/up/weight", (8, 4)),
    ("blocks/0/norm1/bias", (4,)),
    ("blocks/0/norm1/weight", (4,)),
    ("blocks/0/norm2/bias", (4,)),
    ("blocks/0/norm2/weight", (4,)),
    ("head/weight", (256, 4)),
    ("norm_final/bias", (4,)),
    ("norm_final/weight", (4,)),
    ("position_embedding/weight", (2, 4)),
    ("token_embedding/weight", (256, 4)),
)
ALIAS = (10, 14)
BAD_SHAPES = SHAPES[:11] + (("norm_final/bias", (5,)),) + SHAPES[12:]


def checkpoints() -> dict[str, bytes]:
    valid = zero_checkpoint()
    corrupt = bytearray(valid)
    corrupt[-1] ^= 1
    return {
        "zero-state.etcp": valid,
        "bad-checksum.etcp": bytes(corrupt),
        "bad-provider.etcp": zero_checkpoint(provider=b"foreign-cpu-f32-v1"),
        "bad-shape.etcp": zero_checkpoint(shapes=BAD_SHAPES),
        "bad-tie.etcp": zero_checkpoint(alias=(6, 8)),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zero_checkpoint(*, provider: bytes = PROVIDER,
                    shapes: tuple = SHAPES, alias: tuple[int, int] = ALIAS) -> bytes:
    entries = tuple(
        (tuple(segment.encode("ascii") for segment in path.split("/")),
         1, 3, shape, bytes(4 * math.prod(shape)))
        for path, shape in shapes
    )
    raw = build_container(entries, (alias,), provider)
    parsed = parse_golden(raw, provider)
    if len(parsed.records) != 15 or len(parsed.aliases) != 1:
        raise AssertionError("zero C1 topology changed")
    if tuple(parsed.aliases[0].members) != alias:
        raise AssertionError("zero C1 tied alias changed")
    if any(any(raw[row.payload_start:row.payload_start + row.payload_bytes])
           for row in parsed.records):
        raise AssertionError("zero C1 payload is not all zero")
    return raw


def _files(directory: Path) -> list[dict[str, object]]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("uniform corpus directory is not a real directory")
    if any(not path.is_file() or path.is_symlink() for path in directory.iterdir()):
        raise ValueError("uniform corpus contains a nonregular file")
    return [{"path": path.name, "bytes": path.stat().st_size,
             "sha256": _hash(path)} for path in sorted(directory.iterdir())]


def _batch(row) -> dict[str, object]:
    return {"inputs": list(row.inputs[0]), "targets": list(row.targets[0]),
            "mask": list(row.loss_mask[0])}


def materialize(output: Path) -> dict[str, object]:
    if output.exists() or output.is_symlink() or output.parent.resolve() != output.parent.absolute():
        raise ValueError("uniform fixture output must be fresh and canonical")
    output.mkdir(parents=True)
    artifact = encode_artifact(IDENTITY)
    if encode_bytes(IDENTITY, RAW) != tuple(RAW):
        raise AssertionError("T1 changed raw zero token IDs")
    (output / "tokenizer.etok").write_bytes(artifact)
    for name, raw in checkpoints().items():
        (output / name).write_bytes(raw)
    corpus = output / NAME
    corpus.mkdir()
    manifest = write_d1_resource(corpus, (tuple(RAW),),
                                 fingerprint=FINGERPRINT, vocab=VOCAB_SIZE)
    loaded, digest = load_d1_resource(corpus, expected_fingerprint=FINGERPRINT,
                                      expected_vocab=VOCAB_SIZE, maximum_total_tokens=6)
    if tuple(parse_corpus(corpus)["tokens"]) != tuple(RAW):
        raise AssertionError("D1 zero bytes changed")
    dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=digest,
                               tokenizer_fingerprint=FINGERPRINT,
                               vocab_size=VOCAB_SIZE)
    start = dataset.snapshot().hex()
    batches = []
    while (row := dataset.next_batch()) is not None:
        batches.append(_batch(row))
    case = {"name": NAME, "source_bytes": RAW.hex(), "batches": batches,
            "start_cursor": start, "end_cursor": dataset.snapshot().hex(),
            "total_rows": len(dataset.rows),
            "corpus": {"manifest_trailer_sha256": manifest[-32:].hex(),
                       "files": _files(corpus)}}
    payload = {"kind": KIND, "profile": PROFILE,
               "tokenizer": {"artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                             "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE},
               "checkpoint": {"path": "zero-state.etcp", "bytes": len(zero_checkpoint()),
                              "sha256": _hash(output / "zero-state.etcp"),
                              "provider": PROVIDER.decode("ascii"),
                              "entry_count": 15, "alias_indices": list(ALIAS),
                              "negatives": [
                                  {"path": name, "bytes": len(raw),
                                   "sha256": hashlib.sha256(raw).hexdigest()}
                                  for name, raw in checkpoints().items()
                                  if name != "zero-state.etcp"]},
               "cases": [case]}
    validate_materialized(output, payload, check_manifest=False)
    (output / "uniform-manifest.json").write_bytes(canonical(payload))
    return payload


def validate_materialized(root: Path, payload: object, *, check_manifest: bool = True) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "kind", "profile", "tokenizer", "checkpoint", "cases"
    } or payload["kind"] != KIND or payload["profile"] != PROFILE:
        raise ValueError("uniform fixture schema mismatch")
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError("uniform fixture path is not canonical")
    if any(path.is_symlink() or not (path.is_file() or path.is_dir())
           for path in root.rglob("*")):
        raise ValueError("uniform fixture contains a symlink or nonregular entry")
    artifact = encode_artifact(IDENTITY)
    if payload["tokenizer"] != {
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "fingerprint": FINGERPRINT, "vocab_size": VOCAB_SIZE,
    } or (root / "tokenizer.etok").read_bytes() != artifact:
        raise ValueError("uniform T1 identity mismatch")
    expected_c1 = zero_checkpoint()
    negative = [
        {"path": name, "bytes": len(raw),
         "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in checkpoints().items() if name != "zero-state.etcp"
    ]
    if payload["checkpoint"] != {
        "path": "zero-state.etcp", "bytes": len(expected_c1),
        "sha256": hashlib.sha256(expected_c1).hexdigest(),
        "provider": PROVIDER.decode("ascii"), "entry_count": 15,
        "alias_indices": list(ALIAS), "negatives": negative,
    } or any((root / name).is_symlink() or (root / name).read_bytes() != raw
             for name, raw in checkpoints().items()):
        raise ValueError("uniform C1 checksum/provider/shape/tie differs")
    if not isinstance(payload["cases"], list) or len(payload["cases"]) != 1:
        raise ValueError("uniform case count differs")
    case = payload["cases"][0]
    corpus = root / NAME
    if not isinstance(case, dict) or set(case) != {
        "name", "source_bytes", "batches", "start_cursor", "end_cursor",
        "total_rows", "corpus",
    } or case["name"] != NAME or case["source_bytes"] != RAW.hex():
        raise ValueError("uniform case identity differs")
    if not isinstance(case["corpus"], dict) or set(case["corpus"]) != {
        "manifest_trailer_sha256", "files"
    } or corpus.is_symlink() or case["corpus"]["files"] != _files(corpus):
        raise ValueError("uniform D1 file inventory differs")
    loaded, digest = load_d1_resource(corpus, expected_fingerprint=FINGERPRINT,
                                      expected_vocab=VOCAB_SIZE, maximum_total_tokens=6)
    if tuple(parse_corpus(corpus)["tokens"]) != tuple(RAW) or \
            case["corpus"]["manifest_trailer_sha256"] != digest.hex():
        raise ValueError("uniform D1 bytes or checksum differs")
    dataset = ReferenceDataset(loaded, CONFIG, manifest_digest=digest,
                               tokenizer_fingerprint=FINGERPRINT,
                               vocab_size=VOCAB_SIZE)
    start = dataset.snapshot().hex()
    rows = []
    while (row := dataset.next_batch()) is not None:
        rows.append(_batch(row))
    if case["batches"] != rows or case["total_rows"] != 3 or \
            case["start_cursor"] != start or case["end_cursor"] != dataset.snapshot().hex() or \
            [row["mask"] for row in rows] != [[True, True], [True, True], [True, False]] or \
            any(row["inputs"] != [0, 0] or row["targets"] != [0, 0] for row in rows):
        raise ValueError("uniform D2 rows/count/cursor differ")
    for ordinal, key in ((0, "start_cursor"), (3, "end_cursor")):
        cursor = decode_reference_cursor(bytes.fromhex(case[key]))
        if cursor.next_ordinal != ordinal or cursor.total_rows != 3 or \
                cursor.manifest_digest != digest or cursor.config != CONFIG or \
                cursor.tokenizer_fingerprint != FINGERPRINT:
            raise ValueError("uniform cursor identity differs")
    expected = {"tokenizer.etok", *checkpoints(), NAME,
                *(f"{NAME}/{item['path']}" for item in case["corpus"]["files"])}
    if check_manifest:
        expected.add("uniform-manifest.json")
        if (root / "uniform-manifest.json").read_bytes() != canonical(payload):
            raise ValueError("uniform manifest bytes differ")
    if {str(path.relative_to(root)) for path in root.rglob("*")} != expected:
        raise ValueError("uniform resource inventory differs")


def build_reference(payload: dict[str, object]) -> dict[str, object]:
    # PyTorch is confined to development reference generation.
    from tests.e3_reference.reference import _synthetic_case, validate_bundle
    from tests.m3 import reference as m3

    m3.setup()
    if tuple(m3.PARAMETERS) != SHAPES[:-1]:
        raise ValueError("accepted M3 parameter shape contract changed")
    parameters = {path: m3.torch.zeros(shape, dtype=m3.torch.float64)
                  for path, shape in m3.PARAMETERS}
    case = payload["cases"][0]
    synthetic, mathematical, loss = _synthetic_case(case["batches"], parameters)
    if synthetic["final"]["metrics_f32"][3] != "3f800000":
        raise ValueError("uniform tie did not select lowest target ID")
    if any(batch["ce_f32"] != ["40b17218", "40b17218"]
           for batch in synthetic["batches"]):
        raise ValueError("uniform CE word differs from log(256)")
    if any(word != "00000000" for batch in synthetic["batches"]
           for word in batch["logits_f32"]):
        raise ValueError("zero M3 parameters yielded nonzero logits")
    _, successor = m3.parameters(1729)
    total_bytes = sum(math.prod(shape) * 4 for _, shape in SHAPES[:-1])
    result = {"kind": "reference-bundle", "profile": m3.PROFILE,
              "cases": [{"name": NAME, "corpus": case, "synthetic": synthetic,
                         "mathematical": mathematical,
                         "mathematical_loss_f64": loss}],
              "model": {"initializer_seed": 1729,
                        "initializer_successor": list(successor),
                        "parameter_f32_sha256": hashlib.sha256(bytes(total_bytes)).hexdigest(),
                        "reference_manifest_sha256": _hash(ROOT / "tests/m3/reference_manifest.json")},
              "provenance": {"corpus_source_sha256": _hash(Path(__file__)),
                             "generator_source_sha256": _hash(Path(__file__)),
                             "m3_reference_source_sha256": _hash(ROOT / "tests/m3/reference.py"),
                             "initializer_source_sha256": _hash(ROOT / "tests/n3k/test_initializer_reference.py"),
                             "rational_reference_source_sha256": _hash(ROOT / "tests/e3_metrics/reference_probe.py"),
                             "q0_lock_sha256": _hash(ROOT / "tests/q0/requirements-oracle.lock"),
                             "python": platform.python_version(), "torch": m3.torch.__version__}}
    validate_bundle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()
    if args.reference.exists() or args.reference.is_symlink():
        raise FileExistsError("uniform reference output must be fresh")
    payload = materialize(args.output)
    args.reference.write_bytes(encode(build_reference(payload)))


if __name__ == "__main__":
    main()
