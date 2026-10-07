"""Strict verdict for the first genuine warmup's intentional model-image abort."""

from __future__ import annotations

import signal


SELECTORS = {"absent-to-present": "absent", "seeded-bit": "present",
             "seeded-metadata": "present"}
LABEL = b"E3-RETENTION-FAIL: success-model-image\n"


def verify(*, selector: str, gradient: str, horizon: int, package: str,
           compile_status: int, link_status: int, runtime_status: int,
           runtime_signal: int | None, timed_out: bool, survivors: bool,
           stdout: bytes, stderr: bytes) -> None:
    if selector not in SELECTORS or gradient != SELECTORS[selector]:
        raise ValueError("wrong selector or gradient state")
    if horizon not in (1024, 8192) or package != "retention-first-eval-mutant":
        raise ValueError("wrong horizon or package")
    if compile_status != 0 or link_status != 0:
        raise ValueError("compile or link failed")
    if timed_out or survivors:
        raise ValueError("timeout or surviving descendant")
    if runtime_signal != signal.SIGABRT or runtime_status != -signal.SIGABRT:
        raise ValueError("runtime did not terminate by actual SIGABRT")
    if stdout != LABEL:
        raise ValueError("missing, extra, truncated or different detector label")
    if stderr:
        raise ValueError("runtime emitted unrelated or sanitizer diagnostics")
