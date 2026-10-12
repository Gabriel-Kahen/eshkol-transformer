#!/usr/bin/env python3
"""Project the existing D1 public wrappers without writer or D2 authority."""

from pathlib import Path
import re
import sys


BEGIN = ";;; D1 facade component begin."
END = ";;; D1 facade component end."
PUBLIC = {
    "et_e1b_public_d1_token_corpus_validate_v1",
    *(f"et_e1b_public_d1_token_corpus_summary_{name}_v1" for name in (
        "shard_count", "total_tokens", "vocab_size", "shard_token_limit",
        "tokenizer_fingerprint", "total_shard_bytes"
    )),
}
NAMES = (
    "token-corpus-validate", "token-corpus-summary-shard-count",
    "token-corpus-summary-total-tokens", "token-corpus-summary-vocab-size",
    "token-corpus-summary-shard-token-limit",
    "token-corpus-summary-tokenizer-fingerprint",
    "token-corpus-summary-total-shard-bytes",
)


def project(source: str) -> str:
    if source.count(BEGIN) != 1 or source.count(END) != 1:
        raise ValueError("D1 facade markers must occur exactly once")
    body = source.split(BEGIN, 1)[1].split(END, 1)[0]
    cuts = (
        ("(define e1b-safe-d1-token-corpus-write-call\n",
         "(define e1b-safe-d1-token-corpus-validate-call\n"),
        ("(define (token-corpus-write! ",
         "(define (token-corpus-validate "),
    )
    for start, end in cuts:
        if body.count(start) != 1 or body.count(end) != 1:
            raise ValueError("D1 writer/validator anchors changed")
        left, right = body.index(start), body.index(end)
        if left >= right:
            raise ValueError("D1 writer/validator order changed")
        body = body[:left] + body[right:]
    symbols = re.findall(r":real ([^\s)]+)", body)
    definitions = re.findall(r"(?m)^\(define (?:\(([^\s()]+)|([^\s()]+))", body)
    actual_definitions = {name for pair in definitions for name in pair if name}
    expected_definitions = set(NAMES) | {f"e1b-safe-d1-{name}-call" for name in NAMES}
    if (len(symbols) != len(PUBLIC) or set(symbols) != PUBLIC or
            len(definitions) != len(expected_definitions) or
            actual_definitions != expected_definitions):
        raise ValueError("D1 read-only wrapper surface changed")
    if "token-corpus-write!" in body or "et_e1b_public_d2_" in body:
        raise ValueError("D1 projection retained writer or D2 authority")
    provides = "\n         ".join(NAMES)
    return (";;; CLI3 read-only projection of production D1 public wrappers.\n"
            "(require transformer.error_consumer)\n"
            f"(provide {provides})\n" + body.strip() + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: stage-cli3-d1-readonly.py SOURCE OUTPUT")
    target = Path(sys.argv[2])
    target.write_text(project(Path(sys.argv[1]).read_text()))
