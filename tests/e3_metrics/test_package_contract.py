"""Exact E3 metrics archive, dependency, strict arithmetic and source audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).with_name("expected")
SOURCE = "native/e3_evaluation_metrics_provider.c"
HEADER = "include/eshkol_transformer/e3_evaluation_metrics_abi.h"
OBJECT = "e3_evaluation_metrics_provider.o"
ACCESSOR = "et_e3_metrics_kernel_provider_v1"


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def manifest(name: str) -> list[str]:
    return (EXPECTED / f"{name}.txt").read_text().splitlines()


def check_report(path: Path) -> None:
    report = json.loads(path.read_text())
    expected = json.loads((EXPECTED / "provider_report.json").read_text())
    assert report == expected, report
    assert path.read_bytes() == (EXPECTED / "provider_report.json").read_bytes()


def check_order(source: str) -> None:
    # These admitted operations have equivalent local reorderings. Numerical
    # kills cannot establish their mandated traversal; audit source as well.
    groups = (
        ("const int64_t winner0 = argmax(logits);", "const int64_t winner1 = argmax(logits + 256);",
         "const float p0 = weight0 * hit0;", "const float p1 = weight1 * hit1;",
         "float correct = 0.0f;", "correct = correct + p0;", "correct = correct + p1;"),
        ("const float next_n = n + nb;", "const float next_w = w + wb;", "const float next_c = correct + cb;"),
        ("const float loss = n / w;", "const float accuracy = correct / w;", "const float perplexity = expf(loss);"),
    )
    for statements in groups:
        positions = []
        for statement in statements:
            assert source.count(statement) == 1, statement
            positions.append(source.index(statement))
        assert positions == sorted(positions), "specified binary32 arithmetic order changed"
    assert "for (int64_t j = 1; j < 256; ++j)" in source
    assert "if (row[j] > maximum)" in source


def check(artifact: Path, ir: Path) -> None:
    obj = artifact / OBJECT
    archive = artifact / "libeshkol_transformer_e3_metrics.a"
    assert output("ar", "t", str(archive)).splitlines() == [OBJECT]
    symbols = output("nm", "-g", "--defined-only", "--format=posix", str(obj))
    assert sorted(line.split()[0] for line in symbols.splitlines()) == [ACCESSOR], symbols
    undefined = output("nm", "-u", "--format=posix", str(obj))
    assert sorted({line.split()[0] for line in undefined.splitlines()}) == manifest("undefined_symbols"), undefined
    assert subprocess.check_output(["ar", "p", str(archive), OBJECT]) == obj.read_bytes()
    depfile = (artifact / "e3_evaluation_metrics_provider.d").read_text().replace("\\\n", "")
    target, inputs = depfile.split(":", 1)
    assert target == OBJECT
    closure = sorted({Path(n).resolve().relative_to(ROOT).as_posix() for n in shlex.split(inputs)})
    assert closure == sorted([SOURCE, HEADER, "include/eshkol_transformer/kernel_abi.h"]), closure
    actual = []
    for directory in ("include", "native", "src", "lib", "internal"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in (".c", ".h", ".esk") and re.search(
                    r"ET_E3_EVALUATION_METRICS_ABI|et_e3_metrics_kernel_provider_v1", path.read_text()):
                actual.append(path.relative_to(ROOT).as_posix())
    assert sorted(actual) == sorted([SOURCE, HEADER]), actual
    for name in (SOURCE, HEADER):
        source = (ROOT / name).read_text()
        assert not re.search(r"\b(python|pytorch|torch|dlopen|dlsym|getenv|malloc|calloc|realloc|free|double)\b", source, re.I), name
        assert "test_aot" not in source and "eshkol_transformer_kernel_provider_v1" not in source
    source = (ROOT / SOURCE).read_text()
    check_order(source)
    reversed_source = source.replace("correct = correct + p0;", "correct = correct + TEMP;").replace(
        "correct = correct + p1;", "correct = correct + p0;").replace(
        "correct = correct + TEMP;", "correct = correct + p1;")
    try:
        check_order(reversed_source)
    except AssertionError:
        pass
    else:
        raise AssertionError("structural order audit admitted reversed terms")
    assert "#pragma STDC FENV_ACCESS ON" in source
    assert "#pragma STDC FP_CONTRACT OFF" in source
    recipe = (ROOT / "scripts/build-e3-metrics.sh").read_text()
    for flag in ("-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-ffp-contract=off",
                 "-fexcess-precision=standard", "-fno-fast-math", "-frounding-math", "-O2",
                 "-O1", "-fsanitize=address,undefined", "-MMD", "-MF", "-MT"):
        assert flag in recipe, flag
    llvm = ir.read_text()
    assert not re.search(r"llvm\.(fma|fmuladd)|(^|[,( ])double([, )]|$)", llvm, re.M)
    assert not re.search(r"\b(fast|reassoc|afn|arcp)\b", llvm)
    for op in ("fadd", "fmul", "fdiv"):
        assert f"llvm.experimental.constrained.{op}.f32" in llvm, op
    assert "@expf(" in llvm
    assembly = output("objdump", "-d", str(obj))
    assert not re.search(r"\b(v?fmadd|v?fmsub|v?fnmadd|v?fnmsub)", assembly)
    for line in (EXPECTED / "predecessor_sources.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    print("E3-METRICS package PASS: exact archive, symbol/source/depfile closures, strict f32 IR and unchanged predecessors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("ir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    check(args.artifact.resolve(), args.ir)
    if args.report:
        check_report(args.report)
