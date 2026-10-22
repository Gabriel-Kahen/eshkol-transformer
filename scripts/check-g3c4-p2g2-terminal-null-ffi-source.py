#!/usr/bin/env python3
"""Audit the exact pinned Eshkol #f -> C NULL extern-ptr lowering path."""

import argparse
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock = dict(line.split("\t", 1) for line in (root / "toolchain/eshkol.lock").read_text().splitlines()
         if "\t" in line and not line.startswith("#"))
pin = lock["eshkol_commit"]
parser = argparse.ArgumentParser()
parser.add_argument("--eshkol-repo", type=Path, required=True,
                    help="local Git repository containing the exact Eshkol lock commit")
args = parser.parse_args()
result = subprocess.run(
    ["git", "-C", str(args.eshkol_repo), "show",
     f"{pin}:lib/backend/llvm_codegen.cpp"],
    capture_output=True, text=True, check=True,
)
source = result.stdout
guard = source[source.index("void emitFfiPointerArgGuard("):]
guard = guard[:guard.index("Function* err_fn =")]
assert "`#f` is exempt: it is how Eshkol spells a NULL pointer argument" in source
for exact in (
    "builder->CreateICmpEQ(type_byte,",
    "ConstantInt::get(int8_type, ESHKOL_VALUE_BOOL)",
    "builder->CreateICmpEQ(payload, ConstantInt::get(int64_type, 0))",
    "rejected = builder->CreateAnd(rejected, builder->CreateNot(is_false));",
):
    assert exact in guard, exact
conversion = source[source.index("if (expected_type->isPointerTy()) {",
                                 source.index("emitFfiPointerArgGuard(")):]
conversion = conversion[:conversion.index("} else if (expected_type->isDoubleTy())")]
for exact in (
    "emitFfiPointerArgGuard(original_tagged, func_name,",
    "Value* data_i64 = unpackInt64FromTaggedValue(original_tagged);",
    "arg = builder->CreateIntToPtr(data_i64, expected_type);",
):
    assert exact in conversion, exact
assert conversion.index("emitFfiPointerArgGuard(original_tagged") < conversion.index(
    "arg = builder->CreateIntToPtr(data_i64, expected_type);")
print(f"G3-C4 terminal first-EOS #f-to-C-NULL compiler source PASS: {pin}")
