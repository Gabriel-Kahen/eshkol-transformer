#!/usr/bin/env python3
"""Compose the noninstalled P1 mode witness from one pinned canonical source."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "internal/p1/lib/transformer/module.esk"
CANONICAL_SHA256 = "8629d7ee347fb81d0a24cbb5d296828208af9bf75965244d70f7a05ecb132e40"
VARIANT_SHA256 = "4dfed83790ced23de91e91bede7bbc730b80273e9419de6b2c2678c5e8d58e1a"
FAULT_VARIANT_SHA256 = "a3b2bebba27d9e1ed594055957266cfacc1236b20da68c5cf477a82d05d69c37"


def spans(source: str) -> list[tuple[int, int, int]]:
    """Return balanced Scheme form byte spans and nesting depths."""
    result: list[tuple[int, int, int]] = []
    stack: list[int] = []
    quoted = False
    escaped = False
    comment = False
    for index, char in enumerate(source):
        if comment:
            if char == "\n":
                comment = False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == ";":
            comment = True
        elif char == '"':
            quoted = True
        elif char == "(":
            stack.append(index)
        elif char == ")":
            if not stack:
                raise ValueError("unmatched close in pinned P1 source")
            start = stack.pop()
            result.append((start, index + 1, len(stack)))
    if stack or quoted:
        raise ValueError("unterminated form or string in pinned P1 source")
    return result


def one_form(source: str, forms: list[tuple[int, int, int]],
             prefix: str, depth: int) -> tuple[int, int]:
    found = [(start, end) for start, end, form_depth in forms
             if form_depth == depth and source.startswith(prefix, start)]
    if len(found) != 1:
        raise ValueError(f"expected one parsed {prefix!r} at depth {depth}")
    return found[0]


def apply_insertions(source: str, additions: list[tuple[int, str]]) -> str:
    """Apply distinct original-source offsets; ambiguous shared offsets reject."""
    positions = [position for position, _ in additions]
    if len(set(positions)) != len(positions):
        raise ValueError("duplicate P1 insertion position")
    if any(position < 0 or position > len(source) for position in positions):
        raise ValueError("P1 insertion position is outside source")
    for position, replacement in sorted(additions, reverse=True):
        source = source[:position] + replacement + source[position:]
    return source


TEST_WRAPPERS = '''
;;; This cell exists only in the generated, noninstalled E3 mode-test variant.
;;; The canonical 73-slot trusted surface and raw P1 node ownership are intact.
(define p1-e3-test-seam (vector #f #f #f #f))
(define (p1-e3-test-fixed-mixed! token)
  ((vector-ref p1-e3-test-seam 0) token))
(define (p1-e3-test-mode-mask token)
  ((vector-ref p1-e3-test-seam 1) token))

'''


def _writes() -> str:
    return "\n".join(
        f"            (vector-set! (vector-ref nodes {index}) 1 "
        f"e3-mode-{'train' if index % 2 == 0 else 'eval'})"
        for index in range(17)
    )


def _mask_terms() -> str:
    return "\n".join(
        f"                    (if (eq? (vector-ref (vector-ref nodes {index}) 1) "
        f"e3-mode-eval) {1 << index} 0)"
        for index in range(17)
    )


TEST_CLOSURES = '''
;;; P1-owned test closures retain no shell or raw-node authority at the call
;;; site. Only the one genuine bound token and its lexical ledger record are
;;; pinned after full idle preflight; the fixed writes then cannot fail.
(vector-set! p1-e3-test-seam 0
  (lambda (token)
    (let ((record (e3-mode-record token)))
      (cond
        ((not record) 1)
        ((and (vector-ref p1-e3-test-seam 2)
              (not (eq? token (vector-ref p1-e3-test-seam 2)))) 1)
        ((not (e3-mode-i64=? (vector-ref record 1) 0)) 2)
        (else
          (let ((status (e3-mode-check record)))
            (if (not (= status 0)) status
                (begin
                  (vector-set! p1-e3-test-seam 2 token)
                  (vector-set! p1-e3-test-seam 3 record)
                  (let ((nodes (vector-ref record 5)))
{writes}
                    0)))))))))
(vector-set! p1-e3-test-seam 1
  (lambda (token)
    (let ((record (vector-ref p1-e3-test-seam 3)))
      (cond
        ((not (and record (eq? token (vector-ref p1-e3-test-seam 2))
                   (eq? token (vector-ref record 0)))) -1)
        ((not (or (e3-mode-i64=? (vector-ref record 1) 0)
                  (e3-mode-i64=? (vector-ref record 1) 2))) -2)
        ((not (= (e3-mode-check record) 0)) -4)
        (else
          (let ((nodes (vector-ref record 5)))
            (+
{mask_terms})))))))

'''

FAULT_WRAPPERS = '''
;;; Fault-tuple-only P1 selector; all calls require the genuine retained token.
(define (p1-e3-test-enter-arm! token)
  ((vector-ref p1-e3-test-seam 4) token))
(define (p1-e3-test-enter-read! token)
  ((vector-ref p1-e3-test-seam 5) token))
(define (p1-e3-test-enter-clear! token)
  ((vector-ref p1-e3-test-seam 6) token))
(define (p1-e3-test-enter-denial! token)
  (if (and (eq? token (vector-ref p1-e3-test-seam 8))
           (= (vector-ref p1-e3-test-seam 7) 1))
      (begin (vector-set! p1-e3-test-seam 7 2) #t)
      #f))

'''

FAULT_CLOSURES = '''
;;; 0 idle, 1 armed, 2 fired, 3 read. Clear rejects an unreached arm.
(vector-set! p1-e3-test-seam 7 0)
(vector-set! p1-e3-test-seam 4
  (lambda (token)
    (let ((record (e3-mode-record token)))
      (cond ((not (and record
                       (eq? token (vector-ref p1-e3-test-seam 2))
                       (eq? record (vector-ref p1-e3-test-seam 3))
                       (eq? token (vector-ref record 0)))) 1)
            ((not (= (vector-ref p1-e3-test-seam 7) 0)) 2)
            ((not (and (e3-mode-i64=? (vector-ref record 1) 0)
                       (= (e3-mode-check record) 0))) 2)
            (else (vector-set! p1-e3-test-seam 8 token)
                  (vector-set! p1-e3-test-seam 7 1)
                  0)))))
(vector-set! p1-e3-test-seam 5
  (lambda (token)
    (if (and (eq? token (vector-ref p1-e3-test-seam 8))
             (= (vector-ref p1-e3-test-seam 7) 2))
        (begin (vector-set! p1-e3-test-seam 7 3) 2)
        -1)))
(vector-set! p1-e3-test-seam 6
  (lambda (token)
    (if (and (eq? token (vector-ref p1-e3-test-seam 8))
             (= (vector-ref p1-e3-test-seam 7) 3))
        (begin (vector-set! p1-e3-test-seam 8 #f)
               (vector-set! p1-e3-test-seam 7 0) 0)
        1)))
'''


def compose(source: str, fault_enter_denial: bool = False) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != CANONICAL_SHA256:
        raise ValueError("canonical P1 source digest differs")
    forms = spans(source)
    provide_start, provide_end = one_form(source, forms, "(provide ", 0)
    trusted_start, trusted_end = one_form(
        source, forms, "(define p1-trusted-surface", 0)
    if not (provide_end < trusted_start):
        raise ValueError("P1 provide/trusted source ordering changed")
    provide = source[provide_start:provide_end]
    suffix = "         p1-trusted-surface)"
    if provide.count(suffix) != 1 or not provide.endswith(suffix):
        raise ValueError("P1 provide anchor changed")
    let_start, let_end = one_form(source, forms, "(let ()", 1)
    if not (trusted_start < let_start < let_end < trusted_end):
        raise ValueError("P1 lexical trusted body changed")
    final_vectors = [(start, end) for start, end, depth in forms
                     if depth == 2 and let_start < start < end < let_end
                     and source.startswith("(vector\n  (lambda (module)", start)]
    if len(final_vectors) != 1:
        raise ValueError("P1 final trusted vector anchor changed")
    vector_start, _ = final_vectors[0]
    closures = TEST_CLOSURES.format(writes=_writes(), mask_terms=_mask_terms())
    if fault_enter_denial:
        previous = """((and (vector-ref p1-e3-test-seam 2)
              (not (eq? token (vector-ref p1-e3-test-seam 2)))) 1)"""
        retired_ok = """((and (vector-ref p1-e3-test-seam 2)
              (not (eq? token (vector-ref p1-e3-test-seam 2)))
              (not (e3-mode-i64=?
                     (vector-ref (vector-ref p1-e3-test-seam 3) 1) 3))) 1)"""
        if closures.count(previous) != 1:
            raise ValueError("P1 test token switch anchor changed")
        closures = closures.replace(previous, retired_ok, 1)
    additions = [
        (provide_end - 1,
         " p1-e3-test-fixed-mixed! p1-e3-test-mode-mask" +
         (" p1-e3-test-enter-arm! p1-e3-test-enter-read!"
          " p1-e3-test-enter-clear!" if fault_enter_denial else "")),
        (trusted_start, (TEST_WRAPPERS.replace(
             "(vector #f #f #f #f)",
             "(vector #f #f #f #f #f #f #f 0 #f)")
             if fault_enter_denial else TEST_WRAPPERS) +
         (FAULT_WRAPPERS if fault_enter_denial else "")),
        (vector_start, closures +
         (FAULT_CLOSURES if fault_enter_denial else "")),
    ]
    if fault_enter_denial:
        enter_start, enter_end = one_form(source, forms, "(define (e3-mode-enter!", 2)
        enter = source[enter_start:enter_end]
        anchor = "(let ((nodes (vector-ref record 5)))"
        if enter.count(anchor) != 1:
            raise ValueError("P1 enter write anchor changed")
        node_start = enter_start + enter.index(anchor)
        node_form = [(start, end) for start, end, _ in forms if start == node_start]
        if len(node_form) != 1 or node_form[0][1] >= enter_end:
            raise ValueError("P1 enter write form changed")
        node_end = node_form[0][1]
        additions.extend([
            (node_start, "(if (p1-e3-test-enter-denial! token) 2 "),
            (node_end, ")"),
        ])
    return apply_insertions(source, additions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fault-enter-denial", action="store_true")
    args = parser.parse_args()
    output = compose(CANONICAL.read_text(), args.fault_enter_denial)
    digest = hashlib.sha256(output.encode()).hexdigest()
    if digest != (FAULT_VARIANT_SHA256 if args.fault_enter_denial else VARIANT_SHA256):
        raise SystemExit(f"generated P1 variant digest differs: {digest}")
    if args.check:
        if args.output.read_text() != output:
            raise SystemExit("generated P1 variant bytes differ")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(digest)


if __name__ == "__main__":
    main()
