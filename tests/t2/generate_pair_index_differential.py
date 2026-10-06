"""Generate bounded Eshkol differential checks from the independent Python T2 oracle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

from tests.t2.reference import Core, Merge, encode_artifact, encode_bytes, pack_i64, train, validate


SEED = 0x7A2B91


def bytevector(raw: bytes) -> str:
    return "(bytevector" + "".join(f" {value}" for value in raw) + ")"


def chunks(document: tuple[bytes, ...]) -> str:
    return "(list" + "".join(f" {bytevector(chunk)}" for chunk in document) + ")"


def documents(value: tuple[tuple[bytes, ...], ...]) -> str:
    return "(list" + "".join(f" {chunks(doc)}" for doc in value) + ")"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def emit_training(lines: list[str], name: str,
                  docs: tuple[tuple[bytes, ...], ...], maximum: int,
                  minimum: int) -> None:
    expected = digest(encode_artifact(train(docs, maximum, minimum)))
    lines.append(
        f'(check "{name}" (equal? (digest (t2-tokenizer-core-serialize '
        f'(t2-bpe-train-core {documents(docs)} {maximum} {minimum} '
        f"'raw '() '() '()))) \"{expected}\"))"
    )


def emit_encoding(lines: list[str], name: str, core: Core, raw: bytes) -> None:
    validate(core)
    merges = "(list" + "".join(
        f" (list {item.rank} {item.result_id} {item.left_id} {item.right_id})"
        for item in core.merges
    ) + ")"
    expected = digest(pack_i64(encode_bytes(core, raw)))
    lines.append(
        f'(check "{name}" (equal? (digest '
        f'(t2-tokenizer-core-encode-i64-le '
        f"(t2-tokenizer-core-create 'raw {merges} '() '() '()) "
        f'{bytevector(raw)})) "{expected}"))'
    )


def generate() -> str:
    rng = random.Random(SEED)
    lines = [
        '(load "internal/t1/lib/t1_tokenizer_core.esk")',
        '(load "internal/t2/lib/t2_bpe_core.esk")',
        '(define checks 0)',
        '(define (digest bytes) (t1-core-hex (t1-core-sha256 bytes)))',
        '(define (check name condition) '
        '(if condition (set! checks (+ checks 1)) '
        '(begin (display "FAIL: ") (display name) (newline) (exit 1))))',
    ]

    # Several source-document shapes force overlaps, rank-created adjacency,
    # ties, empty documents, chunk partitioning, and spelling-length cutoff.
    fixed = [
        ((b"aaaaa",),), ((b"aaaaaa",),),
        ((b"a",), (b"a",), (b"aa",)),
        ((b"abab", b"aba"), (b"baba",)),
        ((), (b"",), (b"aba", b"", b"ba")),
        ((b"a" * 256,),),
        ((b"a" * 513,),),
        ((b"a" * 128,),),
        ((b"a" * 129,),),
        ((bytes(range(256)),),),
    ]
    for index, docs in enumerate(fixed):
        emit_training(lines, f"fixed-train-{index}", docs, 12, 1)

    for index in range(48):
        base = tuple(
            bytes(rng.choice((97, 97, 97, 98, 99, 100, 0, 255))
                  for _ in range(rng.randrange(0, 26)))
            for _ in range(rng.randrange(0, 7))
        )
        docs = tuple((raw[: len(raw) // 2], raw[len(raw) // 2:])
                     if index % 3 == 0 else (raw,) for raw in base)
        maximum = rng.randrange(0, 17)
        minimum = rng.randrange(1, 4)
        emit_training(lines, f"random-train-{index}", docs, maximum, minimum)
        if index % 8 == 0:
            emit_training(lines, f"reordered-train-{index}",
                          tuple(reversed(docs)), maximum, minimum)

    for index in range(10):
        raw = bytes(rng.choice((97, 97, 98, 99, 100, 0, 255))
                    for _ in range(rng.randrange(170, 260)))
        docs = ((raw[:90], raw[90:]),) if index % 2 else ((raw,),)
        emit_training(lines, f"indexed-random-train-{index}", docs, 16, 1)

    # Both the sparse heap and dense original-index route must be exercised.
    chain = Core(merges=(Merge(0, 256, 97, 97),
                         Merge(1, 257, 256, 97),
                         Merge(2, 258, 256, 257)))
    emit_encoding(lines, "dense-overlap", chain, b"a" * 512 + b"b")
    emit_encoding(lines, "dense-rank-created", chain, b"a" * 513 + b"b")
    emit_encoding(lines, "route-boundary-128", chain, b"a" * 127 + b"b")
    emit_encoding(lines, "route-boundary-129", chain, b"a" * 128 + b"b")
    emit_encoding(lines, "rank-created-adjacency", chain, b"aaaaa")
    emit_encoding(lines, "sparse-overlap", chain,
                  b"ab" * 40 + b"aaaaaa" + b"ab" * 40)
    duplicate_spelling = Core(merges=(
        Merge(0, 256, 97, 98), Merge(1, 257, 98, 99),
        Merge(2, 258, 256, 99), Merge(3, 259, 97, 257),
    ))
    emit_encoding(lines, "valid-duplicate-spelling", duplicate_spelling,
                  b"abcabcbcaabcabc")

    for index in range(44):
        spellings = [bytes((value,)) for value in range(256)]
        pairs: set[tuple[int, int]] = set()
        merges: list[Merge] = []
        for rank in range(rng.randrange(0, 14)):
            while True:
                left = rng.choice((97, 98, 99, 100, *range(256, 256 + rank)))
                right = rng.choice((97, 98, 99, 100, *range(256, 256 + rank)))
                if ((left, right) not in pairs
                        and len(spellings[left]) + len(spellings[right]) <= 256):
                    break
            pairs.add((left, right))
            merges.append(Merge(rank, 256 + rank, left, right))
            spellings.append(spellings[left] + spellings[right])
        raw = bytes(rng.choice((97, 97, 98, 99, 100))
                    for _ in range(rng.randrange(0, 130)))
        emit_encoding(lines, f"random-encode-{index}",
                      Core(merges=tuple(merges)), raw)

    lines.extend(['(display "T2 PAIR DIFFERENTIAL PASS: ")',
                  '(display checks) (display " checks") (newline)'])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(generate())


if __name__ == "__main__":
    main()
