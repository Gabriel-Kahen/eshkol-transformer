#!/usr/bin/env python3
"""Compose the noninstalled P1 mode witness from one pinned canonical source."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "internal/p1/lib/transformer/module.esk"
CANONICAL_SHA256 = "1146eee9e053b4a761c24f56b75ed43e756ec2d907d0de48b04e72abe805775d"
VARIANT_SHA256 = "f909e79420850f228f88965e4cfa3a328c54b463fbac1f124cd61942c8ba05c6"


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


def compose(source: str) -> str:
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
    additions = [
        (provide_end - 1,
         " p1-e3-test-fixed-mixed! p1-e3-test-mode-mask"),
        (trusted_start, TEST_WRAPPERS),
        (vector_start, TEST_CLOSURES.format(
            writes=_writes(), mask_terms=_mask_terms())),
    ]
    result = source
    for position, replacement in reversed(additions):
        result = result[:position] + replacement + result[position:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = compose(CANONICAL.read_text())
    digest = hashlib.sha256(output.encode()).hexdigest()
    if digest != VARIANT_SHA256:
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
