"""Offline command-capture controls; no Eshkol build or Docker."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


HELPER = Path(__file__).resolve().parents[2] / "scripts/g3c4_proof_command.py"
ROOT = HELPER.parents[1]


class ProofCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, *argv, **kwargs):
        return subprocess.run([sys.executable, str(HELPER), str(self.root), "--", *argv],
                              capture_output=True, **kwargs)

    def receipt(self):
        paths = list(self.root.glob("command.*/receipt.json"))
        self.assertEqual(len(paths), 1)
        return paths[0].parent, json.loads(paths[0].read_text())

    def test_large_pipe_replays_every_byte(self):
        payload = b"A" * (2 * 1024 * 1024 + 73)
        code = "import sys;sys.stdout.buffer.write(b'A'*(2*1024*1024+73))"
        run = self.invoke(sys.executable, "-c", code)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout, payload)
        path, meta = self.receipt()
        self.assertEqual((path / "stdout.raw").read_bytes(), payload)
        self.assertEqual(meta["stdout_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(meta["stdout_bytes"], len(payload))
        self.assertGreaterEqual(meta["wall_ns"], 0)

    def test_nonzero_exit_and_stderr_are_preserved(self):
        run = self.invoke(sys.executable, "-c",
                          "import sys;sys.stderr.write('failure');sys.exit(27)")
        self.assertEqual((run.returncode, run.stderr), (27, b"failure"))
        path, meta = self.receipt()
        self.assertEqual(meta["exit"], 27)
        self.assertEqual((path / "stderr.raw").read_bytes(), b"failure")

    def test_site_marker_does_not_leak_to_nested_command(self):
        env = dict(os.environ, G3C4_PROOF_SITE="outer.sh:7")
        run = self.invoke(sys.executable, "-c",
                          "import os;print(os.getenv('G3C4_PROOF_SITE', ''))",
                          env=env)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(run.stdout, b"\n")
        _, meta = self.receipt()
        self.assertEqual(meta["site"], "outer.sh:7")

    def test_environment_hash_is_spawned_child_environment(self):
        code = ("import hashlib,os;"
                "print(hashlib.sha256(b'\\0'.join(os.fsencode(k+'='+v) "
                "for k,v in sorted(os.environ.items()))).hexdigest())")
        run = self.invoke(sys.executable, "-c", code)
        self.assertEqual(run.returncode, 0)
        path, _ = self.receipt()
        self.assertEqual(run.stdout, (path / "environment.sha256").read_bytes())

    def test_python_subprocess_run_gets_its_own_receipt(self):
        code = ("import subprocess,sys;"
                "subprocess.run([sys.executable,'-c','print(17)'],check=True)")
        run = self.invoke(sys.executable, "-c", code)
        self.assertEqual((run.returncode, run.stdout), (0, b"17\n"))
        receipts = [json.loads(path.read_text()) for path in
                    self.root.glob("command.*/receipt.json")]
        self.assertEqual(len(receipts), 2)
        self.assertIn("<string>:1", {receipt["site"] for receipt in receipts})

    def test_python_direct_popen_fails_closed(self):
        code = ("import subprocess,sys;"
                "subprocess.Popen([sys.executable,'-c','print(17)'])")
        run = self.invoke(sys.executable, "-c", code)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn(b"unreceipted Python child process", run.stderr)
        self.assertEqual(len(list(self.root.glob("command.*/receipt.json"))), 1)

    def test_python_stdin_option_fails_closed(self):
        code = "import subprocess;subprocess.run(['cat'],input=b'x')"
        run = self.invoke(sys.executable, "-c", code)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn(b"unsupported subprocess.run process options", run.stderr)
        self.assertEqual(len(list(self.root.glob("command.*/receipt.json"))), 1)

    def test_nested_bash_wrapper_uses_new_site(self):
        inner = self.root / "inner.sh"
        outer = self.root / "outer.sh"
        prelude = f"source {ROOT}/scripts/common.sh\nsource {ROOT}/scripts/g3c4_proof_command.sh\n"
        inner.write_text(prelude + 'proof_command python3 -c "print(17)"\n')
        outer.write_text(prelude + f"proof_command bash {inner}\n")
        env = dict(os.environ, G3C4_PROOF_RECEIPTS=str(self.root),
                   G3C4_PROOF_SCRIPT="test-g3c4-p2g2-terminal-close.sh")
        run = subprocess.run(["bash", str(outer)], cwd=ROOT, env=env,
                             capture_output=True)
        self.assertEqual((run.returncode, run.stdout), (0, b"17\n"), run.stderr)
        receipts = [json.loads(path.read_text()) for path in
                    self.root.glob("command.*/receipt.json")]
        self.assertEqual(len(receipts), 2)
        self.assertEqual(len({receipt["site"] for receipt in receipts}), 2)

    def test_stream_overflow_fails_closed(self):
        code = "import sys;sys.stdout.buffer.write(b'B'*(16*1024*1024+1))"
        run = self.invoke(sys.executable, "-c", code)
        self.assertEqual(run.returncode, 86)
        path, meta = self.receipt()
        self.assertEqual(meta["stdout_bytes"], 16 * 1024 * 1024)
        self.assertEqual(len(run.stdout), 16 * 1024 * 1024)
        self.assertEqual((path / "stdout.raw").stat().st_size, 16 * 1024 * 1024)

    def test_marker_fd_is_inherited(self):
        marker = self.root / "marker"
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            run = self.invoke("bash", "-c", "printf Z >&3", pass_fds=(fd,),
                              preexec_fn=lambda: os.dup2(fd, 3))
        finally:
            os.close(fd)
        self.assertEqual(run.returncode, 0)
        self.assertEqual(marker.read_bytes(), b"Z")

    def test_missing_and_symlink_roots_fail_before_spawn(self):
        missing = self.root / "missing"
        run = subprocess.run([sys.executable, str(HELPER), str(missing), "--",
                              sys.executable, "-c", "pass"], capture_output=True)
        self.assertEqual(run.returncode, 86)
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        run = subprocess.run([sys.executable, str(HELPER), str(alias), "--",
                              sys.executable, "-c", "pass"], capture_output=True)
        self.assertEqual(run.returncode, 86)
        self.assertEqual(list(self.root.glob("command.*")), [])

    def test_descendant_holding_pipe_open_hits_deadline(self):
        env = dict(os.environ, G3C4_PROOF_COMMAND_LIMIT_MS="100")
        start = time.monotonic()
        run = self.invoke("bash", "-c", "sleep 0.5 &", env=env)
        self.assertEqual(run.returncode, 86)
        self.assertLess(time.monotonic() - start, 0.45)
        _, meta = self.receipt()
        self.assertEqual(meta["exit"], 86)


if __name__ == "__main__":
    unittest.main()
