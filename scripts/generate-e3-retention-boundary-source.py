#!/usr/bin/env python3
"""Generate fixed test-only D2 and E1 boundary probes for import discovery."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASE_D2 = "ae2c4661ac518dab5314dbf0ae7d9326425bd4add190d7f04485bc75fdcfe414"
BASE_E1 = "9b545f57e131dcdfaa121f318caa78a80db7e831bb91eab836182e2331fdc259"
D2_DECL = "(extern i64 d2-boundary-mark i64 :real et_e3_retention_boundary_v1)\n"
E1_DECL = "(extern i64 e1-boundary-mark i64 :real et_e3_retention_boundary_v1)\n"
D2_CLOSE = """                  (let ((promoted (region-close region caught)))
                    (vector d2-region-exception-marker promoted)))))"""
D2_CLOSE_PROBE = """                  (begin
                    (d2-boundary-mark 3)
                    (let ((promoted (region-close region caught)))
                      (d2-boundary-mark 4)
                      (let ((envelope
                             (vector d2-region-exception-marker promoted)))
                        (d2-boundary-mark 5)
                        envelope))))))"""
D2_RERAISE = "           (raise (vector-ref result 1)))"
D2_RERAISE_PROBE = "           (begin (d2-boundary-mark 6) (raise (vector-ref result 1))))"
E1_RAISE = """    (define (core-raise category operation message details cause)
      (raise (build-error 'transformer-error-raise
                          category operation message details cause)))"""
E1_RAISE_PROBE = """    (define (core-raise category operation message details cause)
      (e1-boundary-mark 1)
      (let ((built (build-error 'transformer-error-raise
                                category operation message details cause)))
        (e1-boundary-mark 2)
        (raise built)))"""


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def replace_one(source, old, new):
    if source.count(old) != 1:
        raise ValueError("expected exactly one fixed source form")
    return source.replace(old, new)


def generate(output_dir):
    output_dir = Path(output_dir)
    source = output_dir / "source"
    d2_path = source / "e3_d2_dataset.esk"
    e1_path = ROOT / "lib/transformer/error_core.esk"
    d2_bytes = d2_path.read_bytes()
    e1_bytes = e1_path.read_bytes()
    if digest(d2_bytes) != BASE_D2 or digest(e1_bytes) != BASE_E1:
        raise ValueError("fixed generated D2 or canonical E1 source changed")
    d2 = d2_bytes.decode()
    e1 = e1_bytes.decode()
    d2 = replace_one(d2, D2_CLOSE, D2_CLOSE_PROBE)
    d2 = replace_one(d2, D2_RERAISE, D2_RERAISE_PROBE)
    d2 = D2_DECL + d2
    e1 = replace_one(e1, E1_RAISE, E1_RAISE_PROBE)
    e1 = E1_DECL + e1
    e1_output = source / "transformer/error_core.esk"
    e1_output.parent.mkdir(mode=0o700, exist_ok=True)
    if e1_output.is_symlink() or d2_path.is_symlink():
        raise ValueError("generated source symlink rejected")
    d2_path.write_text(d2)
    e1_output.write_text(e1)
    receipt = {
        "format": "e3-retention-boundary-test-only-v1",
        "base_d2_sha256": BASE_D2,
        "base_e1_sha256": BASE_E1,
        "d2_probe_sha256": digest(d2.encode()),
        "e1_probe_sha256": digest(e1.encode()),
        "d2_probe_bytes": len(d2.encode()),
        "e1_probe_bytes": len(e1.encode()),
        "stages": ["arm-before-call", "e1-before-build", "e1-after-build",
                   "d2-before-close", "d2-after-close", "d2-after-envelope",
                   "d2-before-reraise", "outer-catch", "outer-region-closed",
                   "after-checks"],
    }
    (output_dir / "boundary-source-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
