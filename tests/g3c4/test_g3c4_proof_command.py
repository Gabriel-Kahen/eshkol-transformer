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
        code = ("import subprocess,sys;"
                "subprocess.Popen([sys.executable,'-c','import time;time.sleep(0.5)'])")
        env = dict(os.environ, G3C4_PROOF_COMMAND_LIMIT_MS="100")
        start = time.monotonic()
        run = self.invoke(sys.executable, "-c", code, env=env)
        self.assertEqual(run.returncode, 86)
        self.assertLess(time.monotonic() - start, 0.45)
        _, meta = self.receipt()
        self.assertEqual(meta["exit"], 86)


if __name__ == "__main__":
    unittest.main()
