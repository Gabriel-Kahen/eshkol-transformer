from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "native" / "o2_wave2_extension.esk"
PREDECESSOR_SHA256 = "4ae490979328499bb96f9d6cb3e13ae61398300275b7bfd83361e72d597ad590"
PUBLIC_HEADER_TREE_SHA256 = (
    "1198c573cdc535b2b4f49d4ef1d31d717ceb6690c6878f5da9d0b44960bc9166"
)
PUBLIC_FACADE_SHA256 = "b2f96c6119cf51388221dd541c7eda3fefb564c14934e46dede310d0ab9c98b0"

CELL = b"(define o2-tr3-create-overlap-check-cell-internal (vector 'uninstalled))\n"
HOOK_START = b";;; Source-private leaf for the eventual single TR3 aggregate."
HOOK_END = b"(define (o2-member-identity? value values)"
CALL = (
    b"    (if (= (o2-tr3-create-overlap-check-internal handles) 1)\n"
    b"        (o2-fail 'invalid-state operation\n"
    b"                 \"optimizer parameters overlap an enrolled trainer\"))\n"
)
PRIVATE_NAMES = (
    "o2-tr3-create-overlap-check-cell-internal",
    "o2-install-tr3-create-overlap-check-internal!",
    "o2-tr3-create-overlap-check-internal",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def header_tree_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "include").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def reconstruct_predecessor(source: bytes) -> bytes:
    if source.count(CELL) != 1 or source.count(CALL) != 1:
        raise AssertionError("O2 hook cell or callsite drifted")
    if source.count(HOOK_START) != 1 or source.count(HOOK_END) != 1:
        raise AssertionError("O2 hook implementation boundary drifted")
    start = source.index(HOOK_START)
    end = source.index(HOOK_END)
    if start >= end:
        raise AssertionError("O2 hook implementation order drifted")
    source = source[:start] + source[end:]
    return source.replace(CELL, b"", 1).replace(CALL, b"", 1)


class ModelLeaseCreateHookPackageTests(unittest.TestCase):
    def test_only_the_frozen_leaf_delta_changed_the_predecessor(self) -> None:
        predecessor = reconstruct_predecessor(EXTENSION.read_bytes())
        self.assertEqual(hashlib.sha256(predecessor).hexdigest(), PREDECESSOR_SHA256)

    def test_public_facade_unchanged_and_header_tree_pinned(self) -> None:
        self.assertEqual(
            sha256(ROOT / "lib" / "transformer" / "optim.esk"),
            PUBLIC_FACADE_SHA256,
        )
        self.assertEqual(header_tree_sha256(), PUBLIC_HEADER_TREE_SHA256)

    def test_private_hook_names_enter_no_public_or_closure_manifest(self) -> None:
        paths = [
            ROOT / "native" / f"o2_wave2_{kind}.txt"
            for kind in (
                "defined_symbols",
                "native_source_closure",
                "private_renames",
                "public_exports",
                "public_strings",
                "source_closure",
                "undefined_symbols",
            )
        ]
        paths.extend(
            (
                ROOT / "native" / "o2_wave2_root.esk",
                ROOT / "lib" / "transformer" / "optim.esk",
            )
        )
        for path in paths:
            text = path.read_text()
            for name in PRIVATE_NAMES:
                self.assertNotIn(name, text, path)
            self.assertNotIn("model_lease_create_hook_test_bridge", text, path)

    def test_existing_public_manifest_cardinalities_are_unchanged(self) -> None:
        expected = {
            "o2_wave2_defined_symbols.txt": 53,
            "o2_wave2_public_exports.txt": 47,
            "o2_wave2_public_strings.txt": 53,
            "o2_wave2_private_renames.txt": 6,
            "o2_wave2_undefined_symbols.txt": 153,
            "o2_wave2_source_closure.txt": 15,
            "o2_wave2_native_source_closure.txt": 29,
        }
        for name, count in expected.items():
            lines = (ROOT / "native" / name).read_text().splitlines()
            entries = [
                line
                for line in lines
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertEqual(len(entries), count, name)


if __name__ == "__main__":
    unittest.main()
