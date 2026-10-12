#!/usr/bin/env python3
"""Host-only pin and negative controls for the one-use J IR object capture."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/capture-e3-sealed-fixture-ir-object.sh"
CLAIM = ROOT / "scripts/claim-e3-sealed-fixture-ir-object.sh"
SEAL = ROOT / "scripts/seal-e3-ir-object-capture.sh"
SPEC = importlib.util.spec_from_file_location(
    "verify_j", ROOT / "scripts/verify-e3-sealed-fixture-j.py")
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class JIRCapture(unittest.TestCase):
    def test_pin_and_closure_changes_reject(self) -> None:
        source = f"{VERIFY.SOURCE_HEAD}\n{VERIFY.SOURCE_TREE}\n".encode()
        runtime = f"{VERIFY.RUNTIME_HEAD}\n".encode()
        closure = b"one\ntwo\n"
        with patch.object(VERIFY, "SOURCE_CLOSURE_SHA256", sha(closure)):
            VERIFY.check_pins(source, runtime, closure)
            for changed in ((source + b"x", runtime, closure),
                            (source, runtime + b"x", closure),
                            (source, runtime, closure + b"x")):
                with self.subTest(changed=changed):
                    with self.assertRaises(ValueError):
                        VERIFY.check_pins(*changed)

    def test_manifest_authenticates_ir_and_shared_library(self) -> None:
        files = {
            "./source-once/compiler/private.ll": b"define void @sample() {}\n",
            "./source-once/compiler/libprivate.so": b"ELF-fixture",
            "./source-once/observed-source-closure.txt": b"source\n",
        }
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            run.mkdir()
            for name, data in files.items():
                target = run / name[2:]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            manifest = Path(temp) / "RUN-SHA256SUMS"
            manifest.write_text("".join(f"{sha(data)}  {name}\n"
                                        for name, data in files.items()))
            with patch.object(VERIFY, "IR_SHA256", sha(files["./source-once/compiler/private.ll"])), \
                 patch.object(VERIFY, "LIB_SHA256", sha(files["./source-once/compiler/libprivate.so"])), \
                 patch.object(VERIFY, "SOURCE_CLOSURE_SHA256", sha(files["./source-once/observed-source-closure.txt"])):
                VERIFY.verify_run_manifest(run, manifest)
                (run / "source-once/compiler/private.ll").write_bytes(b"altered")
                with self.assertRaisesRegex(ValueError, "hash"):
                    VERIFY.verify_run_manifest(run, manifest)
                (run / "source-once/compiler/private.ll").write_bytes(
                    files["./source-once/compiler/private.ll"])
                (run / "source-once/compiler/libprivate.so").write_bytes(b"altered")
                with self.assertRaisesRegex(ValueError, "hash"):
                    VERIFY.verify_run_manifest(run, manifest)

    def test_manifest_path_and_fresh_output_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest"
            for row in ("0" * 64 + "  ./source-once/../escape\n",
                        "0" * 64 + "  ./same\n" + "1" * 64 + "  ./same\n"):
                manifest.write_text(row)
                with self.assertRaisesRegex(ValueError, "unsafe or duplicate"):
                    VERIFY.manifest_rows(manifest)
            existing = Path(temp) / "existing"
            existing.mkdir()
            result = subprocess.run([str(SCRIPT), str(Path(temp) / "absent"),
                                     str(Path(temp) / "absent-review"),
                                     str(Path(temp)), str(existing)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output must be fresh", result.stderr)

    def test_symlinked_intermediate_directory_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            genuine = root / "genuine"
            genuine.mkdir()
            (genuine / "input").write_bytes(b"sealed")
            (root / "alias").symlink_to(genuine, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked"):
                VERIFY.require_hash(root / "alias/input", sha(b"sealed"))

    def test_duplicate_claim_preserves_first_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "outside-output"
            first = subprocess.run([str(CLAIM), str(root), str(output)],
                                   capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            claim = root / "e3-section9-j-ir-object-fb31-fe9"
            before = [(path.name, path.stat().st_mode, path.read_bytes())
                      for path in sorted(claim.iterdir())]
            mode = claim.stat().st_mode
            duplicate = subprocess.run([str(CLAIM), str(root), str(output)],
                                       capture_output=True, text=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertEqual(claim.stat().st_mode, mode)
            self.assertEqual([(path.name, path.stat().st_mode, path.read_bytes())
                              for path in sorted(claim.iterdir())], before)

    def test_capture_rejects_overlap_and_symlinked_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "accepted-j"
            run.mkdir()
            review = root / "review"
            review.mkdir()
            claim_root = root / "claims"
            claim_root.mkdir()
            for output in (run / "new-output", claim_root / "new-output"):
                result = subprocess.run([str(SCRIPT), str(run),
                                         str(review), str(claim_root), str(output)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("overlaps", result.stderr)
                self.assertFalse(output.exists())
            (root / "alias").symlink_to(run, target_is_directory=True)
            result = subprocess.run([str(SCRIPT), str(root / "alias"),
                                     str(review), str(claim_root),
                                     str(root / "output")], capture_output=True,
                                    text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("free of symlinks", result.stderr)

    def test_static_capture_seal_checks_without_self_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "capture"
            out.mkdir()
            (out / "raw.txt").write_bytes(b"raw evidence\n")
            result = subprocess.run([str(SEAL), str(out), "7"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            check = subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS"],
                                   cwd=out, capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertEqual((out / "capture.exit").read_text(), "7\n")
            self.assertFalse(list(out.glob("SHA256SUMS.tmp.*")))
            subprocess.run(["chmod", "-R", "u+w", str(out)], check=True)


if __name__ == "__main__":
    unittest.main()
