#!/usr/bin/env python3
"""Extract and conservatively inspect one sanitized E3 frame-bind LLVM body."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


SYMBOL = b'@"e3-frame-bind!__eshkol_internal_abi"'
TAGGED = rb"%eshkol_tagged_value"
SSA = rb"(%[A-Za-z0-9_.-]+)"
MAX_IR_BYTES = 4 * 1024**3
MAX_FUNCTION_BYTES = 16 * 1024**2


def fail(message: str) -> None:
    raise ValueError(message)


def extract(path: Path) -> tuple[bytes, str]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        with os.fdopen(fd, "rb", buffering=1024 * 1024) as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                fail("IR must be one regular, unlinked source file")
            if not 0 < info.st_size <= MAX_IR_BYTES:
                fail("IR size outside reviewed bound")
            digest = hashlib.sha256()
            body = bytearray()
            inside = False
            found = 0
            for line in stream:
                digest.update(line)
                if line.startswith(b"define ") and SYMBOL in line:
                    found += 1
                    if found != 1 or inside:
                        fail("duplicate E3 frame-bind definition")
                    inside = True
                if inside:
                    body.extend(line)
                    if len(body) > MAX_FUNCTION_BYTES:
                        fail("E3 frame-bind function exceeds reviewed bound")
                    if line.strip() == b"}":
                        inside = False
            if inside or found != 1:
                fail("missing or unterminated E3 frame-bind definition")
            return bytes(body), digest.hexdigest()
    except BaseException:
        # fdopen closes the descriptor after entry; this handles entry failure.
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def inspect(body: bytes) -> dict[str, object]:
    header = body.split(b"\n", 1)[0]
    if not header.startswith(b"define ") or SYMBOL not in header:
        fail("wrong function body")
    if b" sanitize_address {" not in header:
        fail("frame-bind lacks address sanitizer instrumentation")
    if re.search(rb"%model(?:\.[0-9]+)?\.borrow_check\b", body):
        fail("old model binding still has a borrow check")
    if re.search(rb"%root-model(?:\.[0-9]+)?\.borrow_check\b", body):
        fail("root model binding has a borrow check")

    # The compiler may keep an immutable argument in SSA instead of spilling it
    # to %model1. Both forms are safe when no old binding is used after readback.
    old_slot = re.search(rb"%model1 = alloca " + TAGGED + rb",", body)
    old_stores = list(re.finditer(rb"store " + TAGGED + rb" (%[A-Za-z0-9_.-]+), ptr %model1,", body))
    if old_slot:
        if len(old_stores) != 1 or old_stores[0][1] != b"%model":
            fail("old model binding was reassigned or its storage shape changed")
    elif re.search(rb"%model1(?![A-Za-z0-9_.-])", body):
        fail("old model binding has an unrecognized storage shape")
    if not re.search(rb"%root-model = alloca " + TAGGED + rb",", body):
        fail("fresh root-model binding absent")
    root_stores = list(re.finditer(rb"store " + TAGGED + rb" (%vref_result[0-9]+), ptr %root-model,", body))
    if len(root_stores) != 1:
        fail("root-model is not bound once from a vector readback")
    root_store = root_stores[0]
    if old_stores and old_stores[0].start() >= root_store.start():
        fail("old model store occurs after root readback")
    # This excludes direct old-name uses. A value copied before readback can
    # carry the old binding under another SSA name; the verdict requires the
    # independent IR dataflow review named below.
    if re.search(rb"%model(?:1)?(?![A-Za-z0-9_.-])", body[root_store.end():]):
        fail("old model binding is used after root readback")
    # The nearby canonical load and literal index 2 connect this SSA result to
    # canonical frame slot 2. Exact dataflow still receives independent review.
    context = body[max(0, root_store.start() - 24000):root_store.start()]
    if not re.search(rb"%canonical\.load[0-9]* = load " + TAGGED + rb", ptr %canonical,", context):
        fail("root-model readback has no nearby canonical frame load")
    if not re.search(rb"(?:icmp sge i64 2,|getelementptr i64, ptr [^\n]+, i64 2\b)", context):
        fail("root-model readback has no canonical slot-2 check")

    model_loads = re.findall(SSA + rb" = load " + TAGGED + rb", ptr %root-model,", body)
    setup_loads = re.findall(SSA + rb" = load " + TAGGED + rb", ptr %root-setup-box,", body)
    calls = re.findall(rb"call " + TAGGED + rb" @\"p1-e3-mode-bind!__eshkol_internal_abi\"\(" +
                       TAGGED + rb" (\%[A-Za-z0-9_.-]+), " + TAGGED + rb" (\%[A-Za-z0-9_.-]+)\)", body)
    if len(calls) != 1 or calls[0][0] not in model_loads or calls[0][1] not in setup_loads:
        fail("P1 bind does not consume the fresh canonical model and setup box")
    if body.find(b"ptr %root-model,") >= body.find(b'@"p1-e3-mode-bind!__eshkol_internal_abi"'):
        fail("root-model load does not precede P1 binding")
    return {
        "verdict": "PASS_PATTERN_PENDING_INDEPENDENT_IR_REVIEW",
        "function_sha256": hashlib.sha256(body).hexdigest(),
        "function_bytes": len(body),
        "old_model_storage": "stack_slot" if old_slot else "ssa_parameter",
        "root_model_store": root_stores[0][1].decode("ascii"),
        "p1_model_ssa": calls[0][0].decode("ascii"),
        "p1_setup_ssa": calls[0][1].decode("ascii"),
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: inspect-frame-bind-ir.py RAW_LL FUNCTION_LL RESULT_JSON")
    raw, function, result = map(Path, sys.argv[1:])
    if not raw.is_absolute() or not function.is_absolute() or not result.is_absolute():
        raise SystemExit("all paths must be absolute")
    if function.exists() or function.is_symlink() or result.exists() or result.is_symlink():
        raise SystemExit("proof output already exists")
    body, input_sha = extract(raw)
    report = inspect(body)
    report["input_sha256"] = input_sha
    function.write_bytes(body)
    result.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
