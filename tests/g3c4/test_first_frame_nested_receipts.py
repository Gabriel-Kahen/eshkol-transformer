#!/usr/bin/env python3
"""Real-process controls for opt-in first-frame nested child capture."""

from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NestedReceiptTests(unittest.TestCase):
    def run_common(self, script: str, root: Path) -> subprocess.CompletedProcess:
        env = dict(os.environ, G3C4_FIRST_FRAME_RECEIPT_MODE="1",
                   G3C4_FIRST_FRAME_NESTED_ROOT=str(root))
        return subprocess.run(["bash", "-c", f'source "{ROOT}/scripts/common.sh"; {script}'],
                              cwd=ROOT, env=env, capture_output=True)

    def test_failure_and_suppressed_failure_keep_raw_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_common(
                "common_receipt_exec child-fail -- bash -c "
                "'printf out; printf err >&2; exit 7' || true", root)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b"out")
            self.assertEqual(result.stderr, b"err")
            receipt, = root.glob("common-child-fail.*")
            self.assertEqual((receipt / "exit").read_text(), "7\n")
            self.assertEqual((receipt / "stdout.raw").read_bytes(), b"out")
            self.assertEqual((receipt / "stderr.raw").read_bytes(), b"err")

    def test_default_common_path_runs_directly_without_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ)
            env.pop("G3C4_FIRST_FRAME_RECEIPT_MODE", None)
            env["G3C4_FIRST_FRAME_NESTED_ROOT"] = str(root)
            result = subprocess.run(
                ["bash", "-c", f'source "{ROOT}/scripts/common.sh"; '
                 "common_receipt_exec default -- bash -c "
                 "'printf out; printf err >&2; exit 7' || true"],
                cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b"out")
            self.assertEqual(result.stderr, b"err")
            self.assertFalse(list(root.iterdir()))

    def test_pipeline_and_command_substitution_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_common(
                "value=$(common_receipt_exec substitute -- printf 'word\\n'); "
                "printf '%s|' \"$value\"; "
                "common_receipt_exec pipeline -- printf 'tail\\n' | cat", root)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b"word|tail\n")
            self.assertEqual(len(list(root.iterdir())), 2)

    def test_bare_compiler_name_on_relative_path_records_resolver_branch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "bin/clang-21"
            binary.parent.mkdir()
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            receipts = root / "receipts"
            receipts.mkdir()
            env = dict(os.environ, PATH="./bin:" + os.environ["PATH"],
                       G3C4_FIRST_FRAME_RECEIPT_MODE="1",
                       G3C4_FIRST_FRAME_NESTED_ROOT=str(receipts))
            script = (f'source "{ROOT}/scripts/common.sh"; '
                      'found=$(command -v clang-21); '
                      '[[ $found == ./bin/clang-21 ]] || exit 2; '
                      'resolve_executable_into resolved_cc clang-21 test; '
                      'printf "%s" "$resolved_cc"')
            result = subprocess.run(["bash", "-c", script], cwd=root, env=env,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.decode(), str(binary))
            self.assertEqual(len(list(receipts.glob("common-resolve-resolved_cc-*"))), 2)
            evidence = root / "runner-evidence"
            runner_env = dict(env, ESHKOL_SOURCE_DIR="/nonexistent-first-frame-test",
                              ESHKOL_ALLOW_UNSUPPORTED_HOST="1", CC="clang-21")
            runner = subprocess.run(
                ["bash", str(ROOT / "scripts/test-g3c4-p2g2-first-frame.sh"), str(evidence)],
                cwd=root, env=runner_env, capture_output=True)
            self.assertNotEqual(runner.returncode, 0)
            self.assertIn("cc_relative\t1\n", (evidence / "nested-profile.tsv").read_text())

    def test_early_closing_consumer_does_not_hide_replay_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_common(
                "common_receipt_exec large -- python3 -c "
                "'import sys; sys.stdout.write(\"x\" * 10000000)' | head -c 1 >/dev/null",
                root)
            self.assertNotEqual(result.returncode, 0)
            receipt, = root.glob("common-large.*")
            self.assertEqual((receipt / "exit").read_text(), "0\n")
            self.assertEqual(len((receipt / "stdout.raw").read_bytes()), 10000000)

    def test_llvm_child_failure_is_captured_before_toolchain_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = root / "llvm-config-fail"
            tool.write_text("#!/bin/sh\nprintf 'bad compiler' >&2\nexit 7\n")
            tool.chmod(0o755)
            env = dict(os.environ, G3C4_FIRST_FRAME_RECEIPT_MODE="1",
                       G3C4_FIRST_FRAME_NESTED_ROOT=str(root),
                       LLVM_CONFIG_EXECUTABLE=str(tool))
            result = subprocess.run(["bash", "-c", f'source "{ROOT}/scripts/common.sh"; verify_llvm'],
                                    cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(result.returncode, 7)
            receipt, = root.glob("common-llvm-version.*")
            self.assertEqual((receipt / "exit").read_text(), "7\n")
            self.assertEqual((receipt / "stderr.raw").read_bytes(), b"bad compiler")

    def test_caller_toolchain_export_is_restored_with_opt_in_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provenance = root / "eshkol-transformer-provenance.tsv"
            provenance.write_text("cxx_path\t/opt/pinned/clang++-21\n")
            receipts = root / "receipts"
            receipts.mkdir()
            env = dict(os.environ, ESHKOL_BUILD_DIR=str(root),
                       G3C4_FIRST_FRAME_RECEIPT_MODE="1",
                       G3C4_FIRST_FRAME_NESTED_ROOT=str(receipts))
            script = (f'source "{ROOT}/scripts/common.sh"; '
                      f'source "{ROOT}/scripts/g3c4_terminal_command_receipt.sh"; '
                      'terminal_restore_toolchain_export; '
                      "bash -c 'printf %s \"$ESHKOL_CXX_COMPILER\"'")
            result = subprocess.run(["bash", "-c", script], cwd=ROOT,
                                    env=env, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, b"/opt/pinned/clang++-21")
            self.assertEqual(len(list(receipts.glob("common-tsv-cxx_path-*"))), 2)

    def test_failing_static_child_stops_next_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            static = scripts / "check-g3c4-p2g2-first-frame.py"
            static.write_bytes((ROOT / "scripts/check-g3c4-p2g2-first-frame.py").read_bytes())
            child = scripts / "check-g3c4-output-reservation.py"
            child.write_text("import sys\nsys.stderr.write('child failure')\nsys.exit(3)\n")
            receipt_root = root / "receipts"
            receipt_root.mkdir()
            env = dict(os.environ, G3C4_FIRST_FRAME_STATIC_CHILD_ROOT=str(receipt_root))
            result = subprocess.run([sys.executable, str(static)], env=env,
                                    capture_output=True)
            self.assertEqual(result.returncode, 1)
            receipt, = receipt_root.glob("static-output-reservation.*")
            self.assertEqual((receipt / "exit").read_text(), "3\n")
            self.assertEqual((receipt / "stderr.raw").read_bytes(), b"child failure")
            self.assertFalse(list(receipt_root.glob("static-p2g1-pending.*")))

    def test_early_runner_failure_retains_prefix_and_original_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            env = dict(os.environ, ESHKOL_SOURCE_DIR="/nonexistent-first-frame-test",
                       ESHKOL_ALLOW_UNSUPPORTED_HOST="1")
            result = subprocess.run(["bash", str(ROOT / "scripts/test-g3c4-p2g2-first-frame.sh"),
                                     str(evidence)], cwd=ROOT, env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((evidence / "postfailure-audit.exit").read_text(), "0\n")
            inventory = (evidence / "postfailure-inventory.json").read_text()
            self.assertIn("HOLD/INCOMPLETE", inventory)
            self.assertIn("missing common:", inventory)
            self.assertFalse(list(evidence.glob("build.*")))

    def test_late_exit_trap_retains_build_raw_and_original_exit(self):
        runner = (ROOT / "scripts/test-g3c4-p2g2-first-frame.sh").read_text()
        cleanup = re.search(r"cleanup\(\) \{\n.*?\n\}\ntrap cleanup EXIT", runner,
                            re.DOTALL)
        self.assertIsNotNone(cleanup)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            command_root = evidence / "commands.test"
            command_root.mkdir()
            (evidence / "commands-path.txt").write_text(str(command_root) + "\n")
            build = evidence / "build.test"
            build.mkdir()
            shell = (f'source "{ROOT}/scripts/g3c4_terminal_command_receipt.sh"\n'
                     f'PROJECT_ROOT="{ROOT}"\n'
                     f'evidence="{evidence}"\n'
                     f'terminal_commands_root="{command_root}"\n'
                     'terminal_command_index=0\n'
                     f'tmp="{build}"\n'
                     f'{cleanup.group()}\n'
                     'exit 9\n')
            result = subprocess.run(["bash", "-c", shell], cwd=ROOT,
                                    capture_output=True)
            self.assertEqual(result.returncode, 9)
            self.assertTrue(build.is_dir())
            self.assertEqual((evidence / "postfailure-audit.exit").read_text(), "0\n")
            self.assertIn("HOLD/INCOMPLETE",
                          (evidence / "postfailure-inventory.json").read_text())
            audit, = command_root.glob("*-postfailure-audit")
            self.assertTrue((audit / "stdout.raw").is_file())
            self.assertTrue((audit / "stderr.raw").is_file())
            self.assertEqual((audit / "exit").read_text(), "0\n")


if __name__ == "__main__":
    unittest.main()
