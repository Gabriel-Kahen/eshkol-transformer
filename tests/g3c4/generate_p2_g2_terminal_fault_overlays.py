#!/usr/bin/env python3
"""Generate disposable exact-source T1 and postzero fault witnesses."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
T1 = ROOT / "native/g3c4_t1_output_decode_extension.esk"
COMPOSER = ROOT / "native/g3c4_p2g2_terminal_composer_extension.esk"
TEMPLATE = ROOT / "tests/g3c4/p2_g2_terminal_fault_fixture.esk.in"
T1_SHA = "ed6b7fa5195179c8ad987cb410dd938cda4a08faf42e39e79094c83c917a34d9"
COMPOSER_SHA = "cc263096cb6814de887514fb5662f00590047587e5fa1e11ef066ee891cf10ef"
T1_MARKER = ("                      (bytevector-u8-set!\n"
             "                        raw 0 (bytevector-u8-ref staging 0))")
COMPOSER_MARKER = "                  (vector-set! barrier 0 #t)\n"
T1_INSERT = ("                      (vector-set! terminal-decode-cut 0\n"
             "                        (+ 1 (vector-ref terminal-decode-cut 0)))\n"
             "                      (if (= (vector-ref terminal-decode-cut 0)\n"
             "                             (vector-ref terminal-decode-cut 1))\n"
             "                          (raise terminal-decode-condition))\n")
COMPOSER_INSERT = "                  (raise terminal-postzero-condition)\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def insert_exact(source: str, marker: str, insertion: str, after: bool) -> str:
    if source.count(marker) != 1:
        raise ValueError("source marker must occur exactly once")
    location = marker + insertion if after else insertion + marker
    result = source.replace(marker, location, 1)
    if result.count(insertion) != 1 or result.replace(insertion, "", 1) != source:
        raise ValueError("overlay removal identity failed")
    return result


def fixture(template: str, mode: str, ordinal: int,
            t1: Path, composer: Path) -> str:
    fields = {"@MODE@": mode, "@ORDINAL@": str(ordinal),
              "@T1_SOURCE@": json.dumps(str(t1)),
              "@COMPOSER_SOURCE@": json.dumps(str(composer))}
    for old, new in fields.items():
        template = template.replace(old, new)
    if any(key in template for key in fields):
        raise ValueError("unsubstituted fixture field")
    return template


def main(destination: Path) -> None:
    source_t1 = T1.read_bytes()
    source_composer = COMPOSER.read_bytes()
    if digest(source_t1) != T1_SHA or digest(source_composer) != COMPOSER_SHA:
        raise ValueError("immutable original source identity changed")
    t1 = source_t1.decode("utf-8")
    composer = source_composer.decode("utf-8")
    # Negative controls prevent a shifted or ambiguous insertion from passing.
    for bad, marker, insertion, after in (
        (t1 + T1_MARKER, T1_MARKER, T1_INSERT, False),
        (t1.replace(T1_MARKER, "wrong marker", 1), T1_MARKER, T1_INSERT, False),
        (composer + COMPOSER_MARKER, COMPOSER_MARKER, COMPOSER_INSERT, True),
        (composer.replace(COMPOSER_MARKER, "wrong marker", 1),
         COMPOSER_MARKER, COMPOSER_INSERT, True),
    ):
        try:
            insert_exact(bad, marker, insertion, after)
        except ValueError:
            pass
        else:
            raise ValueError("ambiguous/wrong-marker negative control escaped")
    t1_overlay = insert_exact(t1, T1_MARKER, T1_INSERT, False)
    composer_overlay = insert_exact(
        composer, COMPOSER_MARKER, COMPOSER_INSERT, True)
    destination.mkdir(parents=True, exist_ok=False)
    t1_path = destination / "t1-decode-overlay.esk"
    composer_path = destination / "composer-postzero-overlay.esk"
    t1_path.write_text(t1_overlay)
    composer_path.write_text(composer_overlay)
    template = TEMPLATE.read_text()
    variants = (
        ("parity-original", "parity", 0, T1, COMPOSER),
        ("parity-overlay", "parity", 0, t1_path, COMPOSER),
        ("unmarked", "unmarked", 0, T1, COMPOSER),
        ("decode-first", "first", 1, t1_path, COMPOSER),
        ("decode-second", "second", 2, t1_path, COMPOSER),
        ("postzero", "postzero", 0, T1, composer_path),
        ("precommit", "precommit", 0, T1, COMPOSER),
    )
    created = [t1_path, composer_path]
    for name, mode, ordinal, t1_file, composer_file in variants:
        path = destination / f"{name}.esk"
        path.write_text(fixture(template, mode, ordinal, t1_file, composer_file))
        created.append(path)
    (destination / "SHA256SUMS").write_text("".join(
        f"{digest(path.read_bytes())}  {path.name}\n" for path in created))
    print("G3-C4 terminal exact-source fault overlays PASS: "
          "two markers, four rejection controls, seven disposable fixtures")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_p2_g2_terminal_fault_overlays.py OUTPUT_DIR")
    main(Path(sys.argv[1]).resolve())
