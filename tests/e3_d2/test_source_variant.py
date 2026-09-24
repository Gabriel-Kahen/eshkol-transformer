#!/usr/bin/env python3
"""Independent byte/provenance and hostile-input proof for the fixed E3-D2 generator."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = "scripts/generate-e3-d2-source.py"
MANIFEST = "native/e3_d2_source_variant.json"
SOURCE = "internal/d2/lib/d2_dataset.esk"
OUTPUT = "source/e3_d2_dataset.esk"
PROVENANCE = "source/e3_d2_source_provenance.json"
SOURCE_SHA = "53c5f1eb1c6a306503036251960607d1800a2d944a2322af6b5240fc5b20d5a9"
RESULT_SHA = "d102ee50a33140271653f732f9dfea7c1c842c5c0922e99b4509ebffdc46ccaa"
REPLACEMENT = (b"(define (d2-tokenizer-identity tokenizer)\n"
               b"  (let ((result (e3-d2-byte-identity tokenizer))) result))\n\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SourceVariantTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e3-d2-source-test-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.repository = self.workspace / "repository"
        for relative in (GENERATOR, MANIFEST, SOURCE):
            destination = self.repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        self.output = self.workspace / "artifact"
        self.source = self.repository / SOURCE
        self.manifest = json.loads((self.repository / MANIFEST).read_bytes())
        self.expected = self.manifest["expected_form"].encode()

    def run_generator(self, *extra, output=None, script=None):
        return subprocess.run(
            [sys.executable, str(script or self.repository / GENERATOR),
             "--output-dir", str(output or self.output), *extra],
            cwd=self.workspace, capture_output=True, timeout=10)

    def reject(self, *extra, output=None, script=None):
        result = self.run_generator(*extra, output=output, script=script)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, b"")
        self.assertFalse((self.output / OUTPUT).exists())
        return result

    def test_two_generations_exact_independent_byte_delta_and_provenance(self):
        original = self.source.read_bytes()
        self.assertEqual(sha(original), SOURCE_SHA)
        # Independently delimit the actual full form using the following definition.
        start = original.index(b"(define (d2-tokenizer-identity tokenizer)\n")
        end = original.index(b"(define (d2-token-dataset-open config tokenizer)\n", start)
        self.assertEqual(original[start:end], self.expected)
        expected = original[:start] + REPLACEMENT + original[end:]
        evidence = []
        for name in ("first", "second"):
            output = self.workspace / name
            result = self.run_generator(output=output)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = (output / OUTPUT).read_bytes()
            self.assertEqual(actual, expected)
            self.assertEqual(sha(actual), RESULT_SHA)
            self.assertEqual(actual.count(REPLACEMENT), 1)
            self.assertNotIn(b"t2-private-tokenizer-", actual)
            self.assertEqual(actual[:start], original[:start])
            self.assertEqual(actual[start + len(REPLACEMENT):], original[end:])
            self.assertNotIn(b"\r", actual)
            provenance = (output / PROVENANCE).read_bytes()
            self.assertEqual(result.stdout, provenance)
            self.assertNotIn(str(self.workspace).encode(), provenance)
            report = json.loads(provenance)
            for field, data in (("source", original), ("generated", actual),
                                ("expected_form", original[start:end]),
                                ("replacement_form", REPLACEMENT),
                                ("prefix", original[:start]), ("suffix", original[end:]),
                                ("generator", (self.repository / GENERATOR).read_bytes()),
                                ("manifest", (self.repository / MANIFEST).read_bytes())):
                self.assertEqual(report[field + "_sha256"], sha(data))
            evidence.append((actual, provenance))
        self.assertEqual(evidence[0], evidence[1])
        self.assertEqual(self.source.read_bytes(), original)

    def test_repeat_same_destination_is_byte_identical(self):
        first = self.run_generator()
        second = self.run_generator()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(sorted(p.name for p in (self.output / "source").iterdir()),
                         ["e3_d2_dataset.esk", "e3_d2_source_provenance.json"])

    def test_predecessor_drift_and_unexpected_includes(self):
        original = self.source.read_bytes()
        for changed in (original + b"; drift\n", original.replace(b"\n", b"\r\n"),
                        b'(include "foreign.esk")\n' + original,
                        original + b'(load "foreign.esk")\n',
                        original.replace(b"d2-token-dataset-open config", b"foreign-open config")):
            with self.subTest(changed_sha=sha(changed)):
                self.source.write_bytes(changed)
                self.reject()

    def test_missing_duplicate_or_altered_definition(self):
        original = self.source.read_bytes()
        for changed in (original.replace(self.expected, b""), original + self.expected,
                        original.replace(self.expected, REPLACEMENT),
                        original + b"(define (d2-tokenizer-identity other) other)\n"):
            with self.subTest(changed_sha=sha(changed)):
                self.source.write_bytes(changed)
                self.reject()

    def test_manifest_cannot_override_paths_forms_or_pins(self):
        path = self.repository / MANIFEST
        for field, value in (("source", "../foreign.esk"), ("output", "../escape.esk"),
                             ("source_sha256", "0" * 64),
                             ("expected_form", ""), ("replacement_form", "(evil)\n"),
                             ("include_dirs", ["/tmp"]), ("version", 2)):
            with self.subTest(field=field):
                path.write_text(json.dumps(dict(self.manifest, **{field: value})))
                self.reject()

    def test_unknown_cli_sources_roots_includes_substitutions(self):
        for argument in ("--root", "--source", "--manifest", "--include", "-I",
                         "--replacement", "--output", "--output-di"):
            with self.subTest(argument=argument):
                self.reject(argument, str(self.workspace))
        self.reject("unexpected.esk")

    def test_requires_explicit_output_directory(self):
        result = subprocess.run([sys.executable, str(self.repository / GENERATOR)],
                                capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"--output-dir", result.stderr)

    def test_source_manifest_and_generator_leaf_symlinks_reject(self):
        for relative in (SOURCE, MANIFEST, GENERATOR):
            with self.subTest(relative=relative):
                path = self.repository / relative
                moved = path.with_name(path.name + ".real")
                path.rename(moved)
                path.symlink_to(moved.name)
                self.reject()
                path.unlink()
                moved.rename(path)

    def test_input_parent_symlink_rejects(self):
        original = self.source.parent
        moved = original.with_name("lib.real")
        original.rename(moved)
        original.symlink_to(moved.name, target_is_directory=True)
        self.reject()

    def test_repository_parent_symlink_rejects(self):
        link = self.workspace / "repository-link"
        link.symlink_to(self.repository, target_is_directory=True)
        self.reject(script=link / GENERATOR)

    def test_output_directory_and_parent_symlinks_reject(self):
        external = self.workspace / "external"
        external.mkdir()
        self.output.symlink_to(external, target_is_directory=True)
        self.reject()
        self.assertEqual(list(external.iterdir()), [])
        self.reject(output=self.output / "nested")
        self.assertEqual(list(external.iterdir()), [])

    def test_output_source_directory_symlink_rejects(self):
        self.output.mkdir()
        external = self.workspace / "external"
        external.mkdir()
        (self.output / "source").symlink_to(external, target_is_directory=True)
        self.reject()
        self.assertEqual(list(external.iterdir()), [])

    def test_output_leaf_symlinks_reject_without_touching_sentinel(self):
        source_dir = self.output / "source"
        source_dir.mkdir(parents=True)
        sentinel = self.workspace / "sentinel"
        sentinel.write_bytes(b"preserve me")
        for name in ("e3_d2_dataset.esk", "e3_d2_source_provenance.json"):
            with self.subTest(name=name):
                leaf = source_dir / name
                leaf.symlink_to(sentinel)
                result = self.run_generator()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(sentinel.read_bytes(), b"preserve me")
                self.assertEqual(list(source_dir.iterdir()), [leaf])
                leaf.unlink()

    def test_parent_traversal_rejects(self):
        self.reject(output=self.workspace / "artifact" / ".." / "escape")
        self.assertFalse((self.workspace / "escape").exists())

    def test_nonregular_source_rejects_without_blocking(self):
        self.source.unlink()
        os.mkfifo(self.source)
        self.reject()

    def test_output_nonregular_leaf_rejects(self):
        destination = self.output / OUTPUT
        destination.mkdir(parents=True)
        result = self.run_generator()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(destination.iterdir()), [])
        self.assertFalse((self.output / PROVENANCE).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
