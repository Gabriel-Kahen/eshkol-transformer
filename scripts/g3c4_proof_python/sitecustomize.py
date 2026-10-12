"""Capture supported Python child commands during the private G3-C4 proof."""

import os
import pathlib
import subprocess
import sys


if os.environ.get("G3C4_PROOF_RECEIPTS") and not os.environ.get("G3C4_PROOF_HELPER"):
    _original_run = subprocess.run
    _inside_receipt_spawn = False
    _root = os.environ["G3C4_PROOF_RECEIPTS"]
    _helper = str(pathlib.Path(__file__).resolve().parents[1] /
                  "g3c4_proof_command.py")

    def _audit(event, args):
        if event == "subprocess.Popen" and not _inside_receipt_spawn:
            raise RuntimeError("unreceipted Python child process")
        if event in {"os.system", "os.fork", "os.forkpty", "os.posix_spawn",
                     "os.exec", "os.spawn"} and not _inside_receipt_spawn:
            raise RuntimeError("unsupported Python child process route")

    def _run(*popenargs, **kwargs):
        global _inside_receipt_spawn
        if len(popenargs) != 1 or "args" in kwargs:
            raise RuntimeError("unsupported subprocess.run argument form")
        argv = popenargs[0]
        if not isinstance(argv, (list, tuple)) or not argv:
            raise RuntimeError("unsupported subprocess.run command form")
        allowed = {"check", "capture_output", "cwd", "env", "stdout", "stderr",
                   "text", "encoding", "errors", "universal_newlines"}
        if set(kwargs) - allowed or kwargs.get("stderr") == subprocess.STDOUT:
            raise RuntimeError("unsupported subprocess.run process options")
        if kwargs.get("shell") or kwargs.get("timeout") is not None or \
                kwargs.get("executable") is not None or \
                kwargs.get("preexec_fn") is not None or \
                kwargs.get("pass_fds") or kwargs.get("start_new_session"):
            raise RuntimeError("unsupported subprocess.run process options")
        command = [os.fspath(item) for item in argv]
        if any(not isinstance(item, (str, bytes)) for item in command):
            raise RuntimeError("unsupported subprocess.run argv item")
        frame = sys._getframe(1)
        site = f"{pathlib.Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
        environment = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
        environment["G3C4_PROOF_RECEIPTS"] = _root
        environment["G3C4_PROOF_SCRIPT"] = os.environ.get("G3C4_PROOF_SCRIPT", "")
        environment["G3C4_PROOF_SITE"] = site
        environment["G3C4_PROOF_HELPER"] = "1"
        kwargs["env"] = environment
        try:
            _inside_receipt_spawn = True
            return _original_run([sys.executable, _helper, _root, "--", *command],
                                 **kwargs)
        finally:
            _inside_receipt_spawn = False

    sys.addaudithook(_audit)
    subprocess.run = _run
