#!/usr/bin/env python3
"""Pin and compose the noninstalled E3 in-call witness and its two mutants."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
E3_ROOT = ROOT / "native/e3_private_root.esk"
E3_EXTENSION = ROOT / "native/e3_private_extension.esk"
TEST_ROOT = ROOT / "tests/e3_mode_graph/private_root.esk"
P1_GENERATOR = ROOT / "scripts/generate-e3-p1-mode-test-variant.py"
PINS = {
    E3_ROOT: "c5a0cdbceff21aea36b68a4b438e76962250a56cbf5259e2b765c92ce50c63b0",
    E3_EXTENSION: "08f1c3fda42ddf98cd3a69dac1c0d8fa3f35510531cc95db4cc5e9f72b0fd15b",
}
OUTPUT_DIGESTS = {
    "normal": (
        "f909e79420850f228f88965e4cfa3a328c54b463fbac1f124cd61942c8ba05c6",
        "11ac65012eb02ca855e19a3123cc3d0108585dbdd05031e02aae8295f5b90fb9",
    ),
    "missed-child": (
        "c620d8c6235334c4ea0aa95ec81b2852db8925f4f068d08e5e7267e9d200e8f3",
        "11ac65012eb02ca855e19a3123cc3d0108585dbdd05031e02aae8295f5b90fb9",
    ),
    "enter-write-only": (
        "e7f8a0068b63e57e5f1ff70d2411f95cf6350491bcd34299680bf5f51c8b8ea0",
        "11ac65012eb02ca855e19a3123cc3d0108585dbdd05031e02aae8295f5b90fb9",
    ),
    "graph-create": (
        "f909e79420850f228f88965e4cfa3a328c54b463fbac1f124cd61942c8ba05c6",
        "8b0059c99b84eb43abd37f085379ee27d46fb474493dc4f6920673652061c57e",
    ),
}


def digest(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def exact_replace(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one source anchor: {old[:72]!r}")
    return source.replace(old, new, 1)


def pinned(path: Path) -> str:
    source = path.read_text()
    if digest(source) != PINS[path]:
        raise ValueError(f"canonical source digest differs: {path}")
    return source


def root_source() -> str:
    return exact_replace(pinned(E3_ROOT),
                         '(load "e3_private_extension.esk")',
                         '(load "e3_mode_graph_extension.esk")')


MODE_SEAM = '''
;;; Exact noninstalled in-call witness. Every per-call store is preallocated.
(extern i64 e3-test-graph-event i64 :real et_e3_test_graph_event_v1)
(define e3-mode-graph-cell
  (vector #f #f #f 0 (make-vector 4 -1) (make-vector 4 -1)
          (make-vector 4 -1) (make-vector 4 -1) (make-vector 4 -1)))

(define (e3-mode-graph-arm! model frame input)
  (let* ((entry (e3-frame-entry-live frame 'e3-mode-graph-arm!))
         (owner (m3t-entry model 'model 'e3-mode-graph-arm! #t))
         (input-entry (m3t-entry input 'input 'e3-mode-graph-arm! #t)))
    (if (or (not (= (vector-ref entry 1) 1))
            (not (eq? owner (vector-ref entry 3)))
            (vector-ref owner 10)
            (not (= (p1-e3-test-mode-mask (vector-ref entry 6)) 43690)))
        (e3-native-invariant-fail))
    ;; Create and enroll the real same-model M3 workspace before the E3 call.
    (let ((workspace (m3-workspace-for owner 'e3-mode-graph-arm!)))
      (vector-set! e3-mode-graph-cell 0 entry)
      (vector-set! e3-mode-graph-cell 1 workspace)
      (vector-set! e3-mode-graph-cell 2 input-entry)
      #t)))

(define (e3-mode-graph-clear-observations!)
  (vector-set! e3-mode-graph-cell 3 0)
  (let clear ((index 0))
    (if (= index 4) #t
        (begin
          (vector-set! (vector-ref e3-mode-graph-cell 4) index -1)
          (vector-set! (vector-ref e3-mode-graph-cell 5) index -1)
          (vector-set! (vector-ref e3-mode-graph-cell 6) index -1)
          (vector-set! (vector-ref e3-mode-graph-cell 7) index -1)
          (vector-set! (vector-ref e3-mode-graph-cell 8) index -1)
          (clear (+ index 1))))))

(define (e3-mode-graph-note! entry)
  (let ((ordinal (vector-ref e3-mode-graph-cell 3)))
    (if (or (not (eq? entry (vector-ref e3-mode-graph-cell 0)))
            (not (= (vector-ref entry 1) 2))
            (not (e3-entry-bit? entry 8))
            (< ordinal 0) (>= ordinal 4))
        (e3-native-invariant-fail))
    (vector-set! (vector-ref e3-mode-graph-cell 4) ordinal
                 (p1-e3-test-mode-mask (vector-ref entry 6)))
    (vector-set! (vector-ref e3-mode-graph-cell 5) ordinal
                 (e3-test-graph-event 0))
    (vector-set! (vector-ref e3-mode-graph-cell 6) ordinal
                 (e3-test-graph-event 1))
    (vector-set! (vector-ref e3-mode-graph-cell 7) ordinal
                 (e3-test-graph-event 2))
    (vector-set! (vector-ref e3-mode-graph-cell 8) ordinal
                 (e3-test-graph-event 3))
    (vector-set! e3-mode-graph-cell 3 (+ ordinal 1))
    #t))

{graph_helper}
'''

NORMAL_GRAPH_HELPER = '''(define (e3-mode-graph-preacquire! entry)
  (if (not (eq? entry (vector-ref e3-mode-graph-cell 0)))
      (e3-native-invariant-fail))
  #t)
'''

MUTANT_GRAPH_HELPER = '''(define (e3-mode-graph-preacquire! entry)
  (if (not (eq? entry (vector-ref e3-mode-graph-cell 0)))
      (e3-native-invariant-fail))
  (let ((workspace (vector-ref e3-mode-graph-cell 1))
        (input (vector-ref e3-mode-graph-cell 2))
        (started #f) (graph #f))
    (if (or (not workspace) (not input)
            (not (eq? (vector-ref workspace 4) (vector-ref entry 3)))
            (not (eq? (vector-ref workspace 10) 'idle)))
        (e3-native-invariant-fail))
    ;; This is inside the existing outer m3-call. Calling a public wrapper
    ;; here would reject reentrancy before reaching the real graph registry.
    (guard
     (caught
      (#t
       (guard
        (cleanup-failure (#t (e3-native-invariant-fail)))
        (if started (m3-reset! workspace 'e3-mode-graph-preacquire!))
        (if graph
            (m3t-check (m3-native-graph-release graph)
                       'e3-mode-graph-preacquire!)))
       (m3t-rethrow-raw caught 'e3-mode-graph-preacquire!)))
     (m3t-public-workspace-begin! (vector-ref workspace 0)
                                  (vector-ref input 0))
     (set! started #t)
     (m3-forward-schedule-internal (vector-ref workspace 0))
     (set! graph (m3-native-graph-capture (vector-ref workspace 3)))
     (if (null? graph) (m3t-native-fail 'e3-mode-graph-preacquire!))
     (m3-reset! workspace 'e3-mode-graph-preacquire!)
     (set! started #f)
     (m3t-check (m3-native-graph-release graph)
                'e3-mode-graph-preacquire!)
     (set! graph #f)
     #t)))
'''


def extension_source(mode: str) -> str:
    source = pinned(E3_EXTENSION)
    seam = MODE_SEAM.format(graph_helper=(MUTANT_GRAPH_HELPER
                                         if mode == "graph-create"
                                         else NORMAL_GRAPH_HELPER))
    source = exact_replace(source, "(define e3-frame-registry-root",
                           seam + "\n(define e3-frame-registry-root")
    source = exact_replace(source,
        '(e3-stage-set! entry 3)\n                     (if (vector-ref (vector-ref entry 3) 10)',
        '(e3-stage-set! entry 3)\n                     (e3-mode-graph-preacquire! entry)\n                     (if (vector-ref (vector-ref entry 3) 10)')
    source = exact_replace(source,
        '(e3-stage-set! entry 7)\n                                (e3-forward-schedule-internal frame)',
        '(e3-stage-set! entry 7)\n                                (e3-mode-graph-note! entry)\n                                (e3-forward-schedule-internal frame)')
    source = exact_replace(source,
        '(vector-set! entry 13 #t)\n                                         (if (e3-traverse! frame dataset)',
        '(vector-set! entry 13 #t)\n                                         (e3-mode-graph-note! entry)\n                                         (if (e3-traverse! frame dataset)')
    return source


def p1_source(mode: str) -> str:
    spec = importlib.util.spec_from_file_location("p1_mode_variant", P1_GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.compose(module.CANONICAL.read_text())
    if digest(source) != module.VARIANT_SHA256:
        raise ValueError("approved P1 mode variant differs")
    if mode in ("missed-child", "enter-write-only"):
        forms = module.spans(source)
        begin, end = module.one_form(source, forms, "(define (e3-mode-enter!", 2)
        old = '(vector-set! (vector-ref nodes 2) 1 e3-mode-eval)'
        changed = exact_replace(source[begin:end], old, '#t')
        source = source[:begin] + changed + source[end:]
        if mode == "missed-child":
            forms = module.spans(source)
            begin, end = module.one_form(source, forms,
                                          "(define (e3-mode-snapshot-check", 2)
            old = '(if entered? e3-mode-eval (vector-ref saved index))'
            new = '(if entered? (if (= index 2) e3-mode-train e3-mode-eval)\n                            (vector-ref saved index))'
            changed = exact_replace(source[begin:end], old, new)
            source = source[:begin] + changed + source[end:]
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "missed-child",
                                            "enter-write-only", "graph-create"),
                        default="normal")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected_root = root_source()
    if TEST_ROOT.read_text() != expected_root:
        raise ValueError("checked-in E3 test root differs from pinned transform")
    outputs = {
        args.output_dir / "transformer/module.esk": p1_source(args.mode),
        args.output_dir / "e3_mode_graph_extension.esk":
            extension_source(args.mode),
    }
    observed = tuple(digest(source) for source in outputs.values())
    if observed != OUTPUT_DIGESTS[args.mode]:
        raise ValueError(f"generated {args.mode} digests differ: {observed}")
    for path, source in outputs.items():
        if args.check:
            if path.read_text() != source:
                raise ValueError(f"generated source bytes differ: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        print(f"{digest(source)}  {path.name}")


if __name__ == "__main__":
    main()
