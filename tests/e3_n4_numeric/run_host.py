#!/usr/bin/env python3
"""Build the isolated direct-provider witness and require compiled mutant kills."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests/e3_n4_numeric/direct_k1.c"
E3 = ROOT / "native/e3_evaluation_metrics_provider.c"
L3S = ROOT / "native/l3s_masked_objective_provider.c"
K1 = ROOT / "native/kernel_abi.c"
FLAGS = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
         "-Wno-unknown-pragmas",  # GCC ignores the providers' STDC FP pragmas.
         "-ffp-contract=off", "-fexcess-precision=standard", "-fno-fast-math",
         "-frounding-math", "-fstack-protector-all", "-I", str(ROOT / "include")]


def build(cc, target, e3=E3, l3s=L3S, sanitized=False):
    mode = ["-O1", "-g", "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer"] if sanitized else ["-O2"]
    command = [cc, *FLAGS, *mode, str(TEST), str(e3), str(l3s), str(K1),
               "-lm", "-o", str(target)]
    subprocess.run(command, check=True)


def run(target, case, expected=0, sanitized=False):
    env = os.environ.copy()
    if sanitized:
        env.update(ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                   UBSAN_OPTIONS="halt_on_error=1")
    completed = subprocess.run([str(target), case], capture_output=True,
                               text=True, env=env, timeout=20)
    if completed.returncode != expected:
        raise AssertionError(f"{target.name} {case}: exit {completed.returncode}, "
                             f"expected {expected}\n{completed.stdout}{completed.stderr}")
    if expected == 0:
        assert completed.stdout == f"E3 N4 direct K1 {case} PASS\n"
        assert not completed.stderr
    else:
        assert "tests/e3_n4_numeric/direct_k1.c:" in completed.stderr
        assert not completed.stdout


def mutated(source, old, new, target):
    text = source.read_text()
    if text.count(old) != 1:
        raise AssertionError(f"mutation anchor must occur once in {source}: {old}")
    target.write_text(text.replace(old, new, 1))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="e3-n4-direct-k1-") as tmp:
        work = Path(tmp)
        baseline = work / "baseline"
        build(args.cc, baseline)
        for case in ("bits", "first", "bool", "fenv"):
            run(baseline, case)
        print("normal: bits, first, bool, fenv PASS")

        sanitized = work / "sanitized"
        build(args.cc, sanitized, sanitized=True)
        for case in ("bits", "first", "bool", "fenv"):
            run(sanitized, case, sanitized=True)
        print("ASan/UBSan: bits, first, bool, fenv PASS (leak detection disabled)")

        sign = "return x == 0.0f ? 0.0f : x;"
        targets = [
            ("e3-sign", E3, sign, "return x;", "bits"),
            ("l3s-sign", L3S, sign, "return x;", "bits"),
            ("first-error", E3, "if (targets[i] < 0 || targets[i] >= 256)",
             "if (0)", "first"),
            ("masked-nonfinite", E3,
             "if (!isfinite(logits[i*256+j])) return domain(e,op);",
             "if (0) return domain(e,op);", "first"),
            ("e3-control", E3, "if (!fp_supported())",
             "if (fp_supported() && 0)", "fenv"),
            ("l3s-control", L3S, "if (!fp_supported())",
             "if (fp_supported() && 0)", "fenv"),
            ("bool-invalid", L3S, "if (byte > 1u) return 0;",
             "if (byte > 2u) return 0;", "bool"),
            ("bool-masked-ce", L3S,
             "if (!nonnegative(value)) return 0;\n"
             "      ce[i] = positive_zero(value);",
             "if (!nonnegative(value) && m[i] != 0.0f) return 0;\n"
             "      ce[i] = m[i] == 0.0f ? 0.0f : positive_zero(value);", "bool"),
            ("output-guard", E3,
             "*(float *)output(c,0)->data = result.correct.correct;",
             "*(float *)output(c,0)->data = result.correct.correct; "
             "((unsigned char *)output(c,0)->data)[4] = 0;", "bits"),
        ]
        for name, source, old, new, case in targets:
            variant = mutated(source, old, new, work / f"{name}.c")
            executable = work / name
            build(args.cc, executable,
                  e3=variant if source == E3 else E3,
                  l3s=variant if source == L3S else L3S)
            run(executable, case, expected=1)
            print(f"mutant {name}: compiled and killed in {case}")


if __name__ == "__main__":
    main()
