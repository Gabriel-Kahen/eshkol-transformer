#!/usr/bin/env python3
"""Generate only the pinned, noninstalled E3 zero P1 projection probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "internal/p1/lib/transformer/module.esk"
SOURCE_SHA256 = "8629d7ee347fb81d0a24cbb5d296828208af9bf75965244d70f7a05ecb132e40"
EMITTED_SHA256 = "06f0d533f2ea1c1188ba939709ab814ba36e01391f91b434f7ca80a058c680c8"
MANIFEST = ROOT / "tests/e3_private/zero_control_p1_overlay_manifest.json"
ANCHOR = b'''  (lambda (identity)
    (construction-seal-prepared! identity))
  )))

(define (module-parameters module)'''

# The selector is fixed and test-only. Slot 70 reads the P1 projection; no
# observer that commits it runs while the single raw field is perturbed.
INSERTION = b'''  ;; Zero-control-only P1 projection rejection, private slot 73.
  (lambda (model modules handles carriers selector)
    (if (not (or (eq? selector 'path-10)
                 (eq? selector 'shape-10)
                 (eq? selector 'tie-10)
                 (eq? selector 'throw-path-10))) 1
        (let ((before (tr3-fixed-recheck! model modules handles carriers)))
          (if (not (= before 0)) before
              (let* ((head (tr3-fixed-shell-raw (vector-ref handles 10) 'handle))
                     (other (tr3-fixed-shell-raw (vector-ref handles 13) 'handle))
                     (token (tr3-fixed-module-raw modules 16))
                     (binding (if (and token (pair? (vector-ref token 2)))
                                  (car (vector-ref token 2)) #f)))
                (if (not (and (handle-value? head) (handle-value? other)
                              (not (eq? head other))
                              (vector? binding) (= (vector-length binding) 3)
                              (string? (vector-ref binding 1))
                              (string=? (vector-ref binding 1) "weight")
                              (eq? (vector-ref binding 2) head))) 4
                    (let* ((tie? (eq? selector 'tie-10))
                           (shape? (eq? selector 'shape-10))
                           (target (if tie? binding head))
                           (field (if tie? 2 (if shape? 3 2)))
                           (original (vector-ref target field))
                           (replacement
                             (cond (tie? other)
                                   (shape? (list 255 4))
                                   (else (list "head" "wrong")))))
                      (guard (caught
                               (#t
                                 (vector-set! target field original)
                                 (if (not (eq? (vector-ref target field) original))
                                     (raise 'e3-zero-topology-fatal))
                                 (if (not (= (tr3-fixed-recheck!
                                               model modules handles carriers) 0))
                                     (raise 'e3-zero-topology-fatal))
                                 (raise caught)))
                        (vector-set! target field replacement)
                        (if (eq? selector 'throw-path-10)
                            (raise 'e3-zero-topology-test-marker))
                        (let ((mismatch
                                (tr3-fixed-recheck!
                                  model modules handles carriers)))
                          (if (not (= mismatch 4))
                              (raise 'e3-zero-topology-fatal))
                          (vector-set! target field original)
                          (if (not (eq? (vector-ref target field) original))
                              (raise 'e3-zero-topology-fatal))
                          (if (not (= (tr3-fixed-recheck!
                                        model modules handles carriers) 0))
                              (raise 'e3-zero-topology-fatal))
                          mismatch)))))))))
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compose(source: bytes) -> bytes:
    if digest(source) != SOURCE_SHA256:
        raise ValueError("canonical P1 source differs from the reviewed bytes")
    if source.count(ANCHOR) != 1:
        raise ValueError("expected exactly one complete P1 slot-72 trailing anchor")
    prefix, suffix = source.split(ANCHOR)
    result = prefix + ANCHOR.replace(b"  )))\n", INSERTION + b"  )))\n", 1) + suffix
    if result.replace(INSERTION, b"", 1) != source or result.count(INSERTION) != 1:
        raise ValueError("P1 overlay changed bytes outside one insertion")
    if not EMITTED_SHA256 or digest(result) != EMITTED_SHA256:
        raise ValueError("generated P1 overlay lacks its reviewed SHA-256 pin")
    return result


def reject_aliases(path: Path) -> None:
    if not path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("overlay output must be an exact absolute path")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("overlay path contains a symlink alias")


def verify_manifest() -> None:
    reject_aliases(MANIFEST)
    expected = {
        "kind": "e3-zero-p1-topology-overlay-v1",
        "source_sha256": SOURCE_SHA256,
        "emitted_sha256": EMITTED_SHA256,
        "generator_sha256": digest(Path(__file__).read_bytes()),
    }
    if json.loads(MANIFEST.read_text()) != expected:
        raise ValueError("zero P1 overlay manifest differs from reviewed hashes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_manifest()
    if (not args.output.startswith("/") or
            any(part in ("", ".", "..") for part in args.output[1:].split("/"))):
        raise ValueError("overlay output rejects lexical path aliases")
    output = Path(args.output)
    reject_aliases(CANONICAL)
    reject_aliases(output)
    if output.name != "module.esk" or output.parent.name != "transformer" or output.parent.parent.name != "source":
        raise ValueError("overlay must occupy source/transformer/module.esk")
    source = CANONICAL.read_bytes()
    result = compose(source)
    if args.check:
        if not output.is_file() or output.read_bytes() != result:
            raise ValueError("retained zero P1 overlay differs from reviewed bytes")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        reject_aliases(output)
        with output.open("xb") as stream:
            stream.write(result)
    print(f"zero-p1-source={SOURCE_SHA256} emitted={EMITTED_SHA256}")


if __name__ == "__main__":
    main()
