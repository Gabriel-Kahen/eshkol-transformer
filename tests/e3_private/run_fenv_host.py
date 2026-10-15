"""Supported-host C observer checks; private Eshkol gate is separate."""

from pathlib import Path
import os
import subprocess
import tempfile


SOURCE = Path(__file__).with_name("fenv_observer_host.c")


def run(command, *, expected=0, env=None):
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.returncode != expected:
        raise AssertionError(f"{command}: rc={result.returncode}, expected={expected}\n"
                             f"stdout={result.stdout}\nstderr={result.stderr}")
    return result


with tempfile.TemporaryDirectory(prefix="e3-fenv-host-") as temporary:
    out = Path(temporary)
    for compiler in ("clang", "gcc"):
        binary = out / compiler
        run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1",
             str(SOURCE), "-lm", "-o", str(binary)])
        assert "E3-FENV-OBSERVER-HOST-PASS" in run([str(binary)]).stdout

    sanitized = out / "sanitized"
    run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1", "-g",
         "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
         str(SOURCE), "-lm", "-o", str(sanitized)])
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
    assert "E3-FENV-OBSERVER-HOST-PASS" in run([str(sanitized)], env=env).stdout

    mutant = out / "mxcsr-mutant"
    run(["clang", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O1",
         "-DET_E3_FENV_MXCSR_MUTANT", "-DET_E3_FENV_EXPECT_KILL",
         str(SOURCE), "-lm", "-o", str(mutant)])
    killed = run([str(mutant)], expected=42)
    assert killed.stderr == "E3-FENV-FAIL: ambient-state\n"

print("E3-FENV-HOST-PASS clang gcc asan-ubsan mxcsr-mutant")
