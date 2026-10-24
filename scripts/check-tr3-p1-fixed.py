#!/usr/bin/env python3
"""Pin the private TR3 fixed-set leaf without executing trusted code."""

from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "internal/p1/lib/transformer/module.esk"
WRAPPER = ROOT / "native/tr3_p1_fixed_set_extension.esk"
MANIFEST = ROOT / "native/tr3_p1_fixed_set_source_closure.txt"
# The reviewed current trusted root retains the original TR3 slots and adds
# the closed CPU4 context-four validators in slots 73/74.
COMPOSED_SOURCE_SHA256 = "e1249050c340fdd246f337a688f329462a1e1aa94720881d0cc5f9a3e8c47a94"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def definition(text, name):
    marker = f"(define ({name}"
    start = text.find(marker)
    require(start >= 0, f"missing lexical helper {name}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == ";":
            newline = text.find("\n", index)
            if newline < 0:
                break
            continue
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError(f"unbalanced lexical helper {name}")


def check():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-e3-p1-contract.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    text = SOURCE.read_text()
    wrapper = WRAPPER.read_text()
    manifest = MANIFEST.read_text().splitlines()

    require(manifest == [
        "internal/p1/lib/transformer/module.esk",
        "native/tr3_p1_fixed_set_extension.esk",
    ], "TR3 P1 private source closure changed")

    capture = definition(text, "tr3-fixed-capture!")
    recheck = definition(text, "tr3-fixed-recheck!")
    capture_context = definition(text, "tr3-fixed-capture-context!")
    recheck_context = definition(text, "tr3-fixed-recheck-context!")
    cpu4_capture = definition(text, "cpu4-fixed-capture!")
    cpu4_recheck = definition(text, "cpu4-fixed-recheck!")
    capture_module = definition(text, "tr3-fixed-capture-module!")
    recheck_module = definition(text, "tr3-fixed-recheck-module")
    capture_modules = definition(text, "tr3-fixed-capture-modules!")
    capture_handles = definition(text, "tr3-fixed-capture-handles!")
    require("let loop" not in capture_modules and "let loop" not in capture_handles,
            "capture writes must remain unrolled")
    for index in range(17):
        require(capture_modules.count(
            f"tr3-fixed-capture-module! root provider modules {index})") == 1,
            f"module capture index {index} is missing or duplicated")
    for index in range(14):
        require(capture_handles.count(
            "tr3-fixed-capture-handle! supplied modules handles carriers "
            f"provider {index})") == 1,
            f"handle capture index {index} is missing or duplicated")

    leaf_start = text.index(";;; TR3 fixed-profile read-only validators.")
    leaf_end = text.index("\n\n(vector\n", leaf_start)
    leaf = text[leaf_start:leaf_end]
    for forbidden in (
        "shell-for-raw", "raw-for-shell", "finalization-plan",
        "module-parameters", "parameter-tree", "construction-begin",
        "construction-seal", "shell-register", "make-vector", "(vector ",
        "(cons ", "(list ", "p1-native-", "tensor-provider-call",
    ):
        require(forbidden not in leaf,
                f"TR3 fixed-set leaf gained forbidden operation: {forbidden}")
    require(all("(fail " not in body for body in
                (capture, recheck, capture_context, recheck_context)),
            "public exception transport entered status-only leaf")
    require(capture_context.count("(eq? token-handle head-handle)") == 1 and
            recheck_context.count("(eq? token-handle head-handle)") == 1,
            "capture/recheck no longer enforce the head/token raw-handle tie")
    require("model supplied modules handles carriers 2" in capture and
            "model modules handles carriers 2" in recheck and
            "model supplied modules handles carriers 4" in cpu4_capture and
            "model modules handles carriers 4" in cpu4_recheck,
            "TR3/CPU4 wrappers no longer use their exact context")
    for helper_name, helper in (
        ("capture", capture_module), ("recheck", recheck_module)):
        require("(vector-ref node 7) #t" in helper and
                "(vector-ref node 1) e3-mode-train" in helper,
                f"{helper_name} no longer requires every visited module to be finalized/train")

    require(wrapper.count("(vector-ref p1-trusted-surface 69)") == 1,
            "capture wrapper no longer calls exact slot 69")
    require(wrapper.count("(vector-ref p1-trusted-surface 70)") == 1,
            "recheck wrapper no longer calls exact slot 70")

    vector_append = (
        "  ;; Append-only source-private TR3 fixed-set validation, slots 69..70.\n"
        "  tr3-fixed-capture!\n"
        "  tr3-fixed-recheck!\n")
    require(text.count(vector_append) == 1,
            "TR3 P1 surface append is missing or duplicated")
    require(sha256(text.encode()).hexdigest() == COMPOSED_SOURCE_SHA256,
            "reviewed current trusted P1 source changed")

    print("TR3-P1 FIXED STRUCTURE PASS: slots=69,70 modules=17 handles=14 carriers=14")
    for path in (SOURCE, WRAPPER, MANIFEST):
        relative = path.relative_to(ROOT)
        print(f"sha256 {sha256(path.read_bytes()).hexdigest()} {relative}")


if __name__ == "__main__":
    try:
        check()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"TR3-P1 FIXED STRUCTURE FAIL: {error}", file=sys.stderr)
        sys.exit(1)
