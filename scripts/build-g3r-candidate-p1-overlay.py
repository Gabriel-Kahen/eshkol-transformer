#!/usr/bin/env python3
"""Generate the noninstalled G3-R P1 source from one pinned trusted source."""

from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "internal/p1/lib/transformer/module.esk"
INSIDE = ROOT / "native/g3r_candidate_p1_inside.esk"
EXTENSION = ROOT / "native/g3r_candidate_construction_extension.esk"
EXPECTED = "1146eee9e053b4a761c24f56b75ed43e756ec2d907d0de48b04e72abe805775d"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise SystemExit(f"overlay anchor count {source.count(old)}: {old[:70]!r}")
    return source.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build-g3r-candidate-p1-overlay.py OUTPUT_DIRECTORY")
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise SystemExit("trusted P1 source changed; review overlay anew")
    source = raw.decode("utf-8")
    source = replace_once(
        source,
        "(extern i64 p1-native-construction-begin ptr\n"
        "        :real et_p1_private_construction_begin_v1)\n",
        "(extern i64 p1-native-construction-begin ptr\n"
        "        :real et_p1_private_construction_begin_v1)\n"
        "(extern i64 p1-native-candidate-construction-begin ptr\n"
        "        :real et_p1_private_candidate_construction_begin_v1)\n"
        "(extern i64 p1-native-candidate-roster-preflight ptr ptr\n"
        "        :real et_p1_private_candidate_graph_preflight_v1)\n",
    )
    source = replace_once(
        source,
        "         module-construction-parameters-internal\n         p1-trusted-surface)",
        "         module-construction-parameters-internal\n"
        "         module-candidate-construction-begin-internal\n"
        "         module-candidate-native-roster-preflight-internal!\n"
        "         g3r-candidate-ledger-begin-internal!\n"
        "         g3r-candidate-model-create-internal!\n"
        "         i2-candidate-construction-begin-internal!\n"
        "         i2-candidate-construction-seal-internal!\n"
        "         g3r-candidate-test-fault-internal!\n"
        "         p1-trusted-surface)",
    )
    source = replace_once(
        source,
        "(define (construction-begin provider-name)\n",
        INSIDE.read_text() + "\n(define (construction-begin provider-name)\n",
    )
    source = replace_once(
        source,
        "\n\n(vector\n  (lambda (module)\n",
        "\n\n" + EXTENSION.read_text() + "\n(vector\n  (lambda (module)\n",
    )
    seal = """    (native-check operation (p1-native-construction-seal native-context identity))
    (vector-set! record 1 'sealed)
    (vector-set! record 4 #f)
    (vector-set! construction-registry-root 1 #f)
    root-shell))"""
    candidate_seal = """    (if (= (vector-length record) 6)
        (let ((sidecar (vector-ref record 5))
              (irreversible? #f))
          ;; Complete rooted enrollment before invoking native seal.
          (vector-set! sidecar 3 (candidate-construction-shells identity))
          (guard (caught (#t (if irreversible? (exit 134) (raise caught))))
            (let ((status
                    (p1-native-construction-seal native-context identity)))
              ;; A nonzero native result is pre-write by its accepted contract.
              (set! irreversible? (= status 0))
              (native-check operation status))
            (vector-set! record 1 'sealed)
            (vector-set! record 4 #f)
            (vector-set! construction-registry-root 1 #f)
            (vector-set! sidecar 1 'sealed)
            root-shell))
        (begin
          (native-check operation
            (p1-native-construction-seal native-context identity))
          (vector-set! record 1 'sealed)
          (vector-set! record 4 #f)
          (vector-set! construction-registry-root 1 #f)
          root-shell))))"""
    source = replace_once(source, seal, candidate_seal)
    surface = """  ;; Candidate-only appended slot 73 in this noninstalled overlay.
  (lambda (provider-name enrollment-cell)
    (candidate-construction-begin provider-name enrollment-cell))
  ;; Candidate-only slot 74 authenticates the native retained roster.
  (lambda (identity)
    (let ((record (construction-record
                    'module-candidate-native-roster-preflight-internal!
                    identity)))
      (if (or (not (= (vector-length record) 6))
              (not (eq? (vector-ref record 1) 'sealed))
              (not (eq? (vector-ref (vector-ref record 5) 4) identity)))
          (fail 'invalid-state
            'module-candidate-native-roster-preflight-internal!
            "candidate P1 sidecar is not sealed" '()))
      (native-check 'module-candidate-native-roster-preflight-internal!
        (p1-native-candidate-roster-preflight native-context identity))))
  ;; Fixed private operations close over the same hidden holder as P1 begin.
  (lambda (config initializer)
    (g3r-candidate-ledger-begin-internal! config initializer))
  (lambda (ledger config initializer)
    (g3r-candidate-model-create-internal! ledger config initializer))
  (lambda (sidecar)
    (i2-candidate-construction-begin-internal! sidecar))
  (lambda (construction sidecar)
    (i2-candidate-construction-seal-internal! construction sidecar))
  (lambda (stage)
    (set! g3r-candidate-test-fault stage))
"""
    source = replace_once(
        source,
        "  (lambda (identity)\n    (construction-seal-prepared! identity))\n  )))",
        "  (lambda (identity)\n    (construction-seal-prepared! identity))\n"
        + surface + "  )))",
    )
    source += """
;; Candidate-only wrapper, absent from the installed module source.
(define (module-candidate-construction-begin-internal provider enrollment-cell)
  ((vector-ref p1-trusted-surface 73) provider enrollment-cell))
(define (module-candidate-native-roster-preflight-internal! identity)
  ((vector-ref p1-trusted-surface 74) identity))
(define (g3r-candidate-ledger-begin-internal! config initializer)
  ((vector-ref p1-trusted-surface 75) config initializer))
(define (g3r-candidate-model-create-internal! ledger config initializer)
  ((vector-ref p1-trusted-surface 76) ledger config initializer))
(define (i2-candidate-construction-begin-internal! sidecar)
  ((vector-ref p1-trusted-surface 77) sidecar))
(define (i2-candidate-construction-seal-internal! construction sidecar)
  ((vector-ref p1-trusted-surface 78) construction sidecar))
(define (g3r-candidate-test-fault-internal! stage)
  ((vector-ref p1-trusted-surface 79) stage))
"""
    out = Path(sys.argv[1]).resolve() / "transformer/module.esk"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source)
    print(f"{hashlib.sha256(out.read_bytes()).hexdigest()}  {out}")


if __name__ == "__main__":
    main()
