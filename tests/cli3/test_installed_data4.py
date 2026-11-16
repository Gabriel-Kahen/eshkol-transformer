"""Installed DATA4 command and non-atomic sidecar contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class InstalledData4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary = Path(os.environ["CLI3_DATA4_BINARY"])
        cls.tokenizer = Path(os.environ["CLI3_DATA4_TOKENIZER"])
        cls.fault_shim = Path(os.environ["CLI3_DATA4_FAULT_SHIM"])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="cli3-data4-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "corpus"
        self.output.mkdir()
        for name, data in (("a", b"AB"), ("dup", b"AB"), ("b", b"CD"),
                           ("short", b"E")):
            (self.root / name).write_bytes(data)

    def command(self, *args: str, status: int = 0,
                fault: str | None = None) -> subprocess.CompletedProcess[str]:
        env = {"PATH": "/definitely-not-a-real-path"}
        if fault is not None:
            env.update(LD_PRELOAD=str(self.fault_shim), CLI3_TEST_FSYNC_MODE=fault)
        result = subprocess.run(
            [str(self.binary), *map(str, args)], capture_output=True, text=True,
            timeout=30, check=False, env=env,
        )
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        return result

    def prepare(self, directory: Path | None = None, *, fault: str | None = None,
                status: int = 0,
                sources: tuple[str, ...] = ("a", "dup", "b", "short")) -> subprocess.CompletedProcess[str]:
        directory = directory or self.output
        arguments = ["corpus", "prepare", "--tokenizer", str(self.tokenizer)]
        for name in sources:
            arguments.extend(("--source", f"{name}={self.root / name}"))
        arguments.extend(("--minimum-bytes", "2", "--maximum-bytes", "2",
                          "--output-directory", str(directory),
                          "--shard-token-limit", "3"))
        return self.command(*arguments, status=13 if fault else status, fault=fault)

    def validate(self, directory: Path | None = None, *, status: int = 0) -> subprocess.CompletedProcess[str]:
        return self.command("corpus", "report-validate", "--input-directory",
                            str(directory or self.output), status=status)

    def test_installed_help_routes(self) -> None:
        group = self.command("corpus", "--help")
        self.assertIn("prepare  Filter and deduplicate", group.stdout)
        self.assertIn("report-validate  Validate a DATA4 report", group.stdout)
        self.assertEqual(group.stderr, "")
        for command in ("prepare", "report-validate"):
            help_result = self.command("corpus", command, "--help")
            self.assertIn(f"eshkol-transformer corpus {command}", help_result.stdout)
            self.assertEqual(help_result.stderr, "")

    def test_prepared_report_and_d1_summary(self) -> None:
        built = self.prepare()
        summary = json.loads(built.stdout)
        self.assertEqual(summary["total_tokens"], 4)
        self.assertEqual(built.stderr, "")
        report = (self.output / "data4-report.tsv").read_bytes()
        prefix, checksum = report.rsplit(b"checksum\t", 1)
        self.assertEqual(checksum, hashlib.sha256(prefix).hexdigest().encode() + b"\n")
        self.assertIn(b"document\t1\tdup\t2\t", report)
        self.assertIn(b"\tduplicate\ta\n", report)
        self.assertIn(b"document\t3\tshort\t1\t", report)
        self.assertIn(b"\tbelow-min\t\n", report)
        checked = self.validate()
        self.assertEqual(checked.stdout, built.stdout)
        self.assertEqual(checked.stderr, "")

    def test_report_tamper_and_malformed_row(self) -> None:
        self.prepare()
        path = self.output / "data4-report.tsv"
        original = path.read_bytes()
        altered = bytearray(original)
        altered[0] ^= 1
        path.write_bytes(altered)
        self.assertIn("category=\"corrupt-data\"", self.validate(status=12).stderr)
        prefix = original.rsplit(b"checksum\t", 1)[0]
        malformed = prefix.replace(b"\tduplicate\ta\n", b"\tkeep\t\n", 1)
        self.assertNotEqual(malformed, prefix)
        path.write_bytes(malformed + b"checksum\t" +
                         hashlib.sha256(malformed).hexdigest().encode() + b"\n")
        self.assertIn("category=\"corrupt-data\"", self.validate(status=12).stderr)

    def test_mismatched_summary_swap_is_rejected(self) -> None:
        self.prepare()
        report = (self.output / "data4-report.tsv").read_bytes()
        other = self.root / "other"
        other.mkdir()
        (self.root / "different").write_bytes(b"XYZ")
        self.command("corpus", "prepare", "--tokenizer", str(self.tokenizer),
                     "--source", f"different={self.root / 'different'}",
                     "--minimum-bytes", "0", "--maximum-bytes", "3",
                     "--output-directory", str(other), "--shard-token-limit", "3")
        (other / "data4-report.tsv").write_bytes(report)
        self.assertIn("category=\"corrupt-data\"",
                      self.validate(other, status=12).stderr)

    def test_same_summary_swap_is_unbound(self) -> None:
        self.prepare()
        report = (self.output / "data4-report.tsv").read_bytes()
        other = self.root / "same-summary"
        other.mkdir()
        (self.root / "different").write_bytes(b"WXYZ")
        self.command("corpus", "build", "--tokenizer", str(self.tokenizer),
                     "--document", str(self.root / "different"),
                     "--output-directory", str(other), "--shard-token-limit", "3")
        (other / "data4-report.tsv").write_bytes(report)
        # This is accepted only because v1 has no manifest-digest binding.
        self.assertEqual(json.loads(self.validate(other).stdout)["total_tokens"], 4)

    def test_missing_oversize_and_noncanonical_report(self) -> None:
        self.prepare()
        path = self.output / "data4-report.tsv"
        original = path.read_bytes()
        path.unlink()
        self.assertIn("category=\"io\"", self.validate(status=13).stderr)
        path.write_bytes(b"X" * 32769)
        self.assertIn("category=\"corrupt-data\"", self.validate(status=12).stderr)
        prefix = original.rsplit(b"checksum\t", 1)[0]
        changed = prefix.replace(b"input-count\t4\n", b"input-count\t04\n", 1)
        path.write_bytes(changed + b"checksum\t" +
                         hashlib.sha256(changed).hexdigest().encode() + b"\n")
        self.assertIn("category=\"corrupt-data\"", self.validate(status=12).stderr)

    def test_precommit_report_failure_preserves_published_d1(self) -> None:
        failed = self.prepare(fault="regular-once")
        self.assertEqual(failed.stdout, "")
        self.assertIn("corpus published; report not published by this invocation",
                      failed.stderr)
        self.assertIn('"corpus-published?":true', failed.stderr)
        self.assertIn('"report-published?":false', failed.stderr)
        self.assertTrue((self.output / "manifest.etm").is_file())
        self.assertFalse((self.output / "data4-report.tsv").exists())

    def test_dangling_sidecar_symlink_is_never_replaced(self) -> None:
        sidecar = self.output / "data4-report.tsv"
        sidecar.symlink_to(self.root / "missing-target")
        failed = self.prepare(status=13)
        self.assertEqual(failed.stdout, "")
        self.assertTrue(sidecar.is_symlink())
        self.assertFalse(sidecar.exists())
        if (self.output / "manifest.etm").exists():
            self.assertIn("corpus published; report not published by this invocation",
                          failed.stderr)
            self.assertIn('"report-published?":false', failed.stderr)
        else:
            self.assertIn("report sidecar already exists", failed.stderr)

    def test_postcommit_report_failure_marks_unknown_durability(self) -> None:
        failed = self.prepare(fault="directory-once")
        self.assertEqual(failed.stdout, "")
        self.assertIn("corpus published; report visible with uncertain durability",
                      failed.stderr)
        self.assertIn('"corpus-published?":true', failed.stderr)
        self.assertIn('"report-published?":true', failed.stderr)
        self.assertIn('"report-durability":"unknown"', failed.stderr)
        self.assertTrue((self.output / "manifest.etm").is_file())
        self.assertTrue((self.output / "data4-report.tsv").is_file())

    def test_invalid_sources_and_existing_sidecar(self) -> None:
        for source in ("bad/name=" + str(self.root / "a"), "missing-equals"):
            result = self.command(
                "corpus", "prepare", "--tokenizer", str(self.tokenizer),
                "--source", source, "--minimum-bytes", "0",
                "--maximum-bytes", "2", "--output-directory", self.output,
                "--shard-token-limit", "3", status=2,
            )
            self.assertEqual(result.stdout, "")
        self.assertFalse((self.output / "manifest.etm").exists())
        duplicated = self.command(
            "corpus", "prepare", "--tokenizer", str(self.tokenizer),
            "--source", f"x={self.root / 'a'}", "--source", f"x={self.root / 'b'}",
            "--minimum-bytes", "0", "--maximum-bytes", "2",
            "--output-directory", self.output, "--shard-token-limit", "3",
            status=10,
        )
        self.assertEqual(duplicated.stdout, "")
        self.assertFalse((self.output / "manifest.etm").exists())
        (self.output / "data4-report.tsv").write_bytes(b"occupied")
        result = self.prepare(status=13)
        self.assertEqual(result.returncode, 13)
        self.assertFalse((self.output / "manifest.etm").exists())


if __name__ == "__main__":
    unittest.main()
