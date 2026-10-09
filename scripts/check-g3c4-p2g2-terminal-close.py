#!/usr/bin/env python3
"""Bounded source/closure guard for the private Eshkol P2/G2 close."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/g3c4_p2g2_terminal_close_extension.esk"
BASE = ROOT / "native/g3c4_p2g2_t1_prefix_coordinator_source_closure.txt"
CLOSURE = ROOT / "native/g3c4_p2g2_terminal_close_source_closure.txt"
ADDITIONS = [
    "native/g3c4_p2g2_terminal_close_extension.esk",
    "tests/g3c4/p2_g2_terminal_close_test.esk",
    "tests/g3c4/p2_g2_terminal_close_feature_off_test.esk",
    "tests/g3c4/p2_g2_terminal_close_native.c",
    "tests/g3c4/terminal_close_allocation_shim.cpp",
    "tests/g3c4/call_entry_allocation_shim.cpp",
    "tests/g3c4/check_p2_g2_terminal_close_oracle.py",
    "tests/g3s/reference.py",
    "tests/g3c4/test_p2_g2_terminal_close_oracle.py",
    "tests/g3c4/test_p2_g2_terminal_close_source.py",
    "scripts/check-g3c4-p2g2-terminal-close.py",
    "scripts/test-g3c4-p2g2-terminal-close.sh",
    "docs/g3/G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_SOURCE_LEAF.md",
    "native/g3c4_p2g2_terminal_close_source_closure.txt",
]
DIRECT_DEPENDENCIES = {
    "tests/g3c4/call_entry_allocation_shim.cpp":
        "53fa37f63e1780437d9021e9f0752e4c35862e4d6db9950ff34182f544bf966f",
    "tests/g3s/reference.py":
        "dfef749ccf1aff32f268e2db7b981a993e2bd3023e27f0044aeb412fdbec53dd",
}


def forms(source: str) -> list[object]:
    """Read just the Scheme structure; strings and comments are opaque."""
    tokens: list[str] = []
    i = 0
    while i < len(source):
        char = source[i]
        if char.isspace():
            i += 1
        elif char == ";":
            end = source.find("\n", i)
            i = len(source) if end < 0 else end + 1
        elif source.startswith("'()", i):
            tokens.append("'()")
            i += 3
        elif char == '"':
            start = i
            i += 1
            while i < len(source):
                if source[i] == "\\":
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise AssertionError("unterminated string")
            tokens.append(source[start:i])
        elif char in "()":
            tokens.append(char)
            i += 1
        else:
            start = i
            while i < len(source) and not source[i].isspace() and source[i] not in "();":
                i += 1
            tokens.append(source[start:i])

    cursor = 0

    def read() -> object:
        nonlocal cursor
        if cursor >= len(tokens):
            raise AssertionError("truncated form")
        token = tokens[cursor]
        cursor += 1
        if token == "(":
            result: list[object] = []
            while cursor < len(tokens) and tokens[cursor] != ")":
                result.append(read())
            if cursor >= len(tokens):
                raise AssertionError("unclosed form")
            cursor += 1
            return result
        if token == ")":
            raise AssertionError("unexpected close")
        return token

    result = []
    while cursor < len(tokens):
        result.append(read())
    return result


def walk(value: object):
    if isinstance(value, list):
        yield value
        for child in value:
            yield from walk(child)


def definition(parsed: list[object], name: str) -> list[object]:
    matches = [form for form in parsed if isinstance(form, list)
               and len(form) >= 3 and form[0] == "define"
               and isinstance(form[1], list) and form[1]
               and form[1][0] == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def calls(form: object, name: str) -> list[list[object]]:
    return [part for part in walk(form) if part and part[0] == name]


def validate_artifact_retention(script: str) -> None:
    assert "rm -rf" not in script, "compiled artifact cleanup returned"
    for required in (
        'printf \'%s\\n\' "$tmp" >"$evidence/compiled-path.txt"',
        '"$tmp/$mode/libg3c4_p2g2_terminal_close.a"',
        '"$tmp/$mode/p2g2-terminal-feature-off"',
        '"$tmp/$mode/p2g2-terminal-protected"',
        '"$tmp/$mode"/objects/*.o',
        '"$evidence/compiled-artifacts.sha256"',
    ):
        assert required in script, required


def validate(source: str, closure: list[str], base: list[str],
             dependencies: dict[str, bytes] | None = None) -> None:
    assert closure == base + ADDITIONS
    assert len(closure) == len(set(closure))
    assert all((ROOT / item).is_file() for item in closure)
    if dependencies is None:
        dependencies = {name: (ROOT / name).read_bytes()
                        for name in DIRECT_DEPENDENCIES}
    assert set(dependencies) == set(DIRECT_DEPENDENCIES)
    for name, digest in DIRECT_DEPENDENCIES.items():
        assert name in closure
        assert hashlib.sha256(dependencies[name]).hexdigest() == digest, name
    validate_artifact_retention(
        (ROOT / "scripts/test-g3c4-p2g2-terminal-close.sh").read_text())
    enabled = (ROOT / "tests/g3c4/p2_g2_terminal_close_test.esk").read_text()
    disabled = (ROOT / "tests/g3c4/p2_g2_terminal_close_feature_off_test.esk").read_text()
    enabled_forms = forms(enabled)
    disabled_forms = forms(disabled)
    for fixture in (enabled_forms, disabled_forms):
        externs = [form[2] for form in fixture
                   if isinstance(form, list) and len(form) >= 3
                   and form[0] == "extern"]
        assert len(externs) == len(set(externs)), "duplicate fixture ABI"
    assert any(form[:3] == ["extern", "i64", "terminal-tail-disarm"]
               for form in enabled_forms if isinstance(form, list))
    assert any(form[:3] == ["extern", "i64", "terminal-test-reference"]
               for form in enabled_forms if isinstance(form, list))
    gate_load = '(load "g3c4_p2g2_ownership_extension.esk")'
    assert gate_load in enabled and gate_load not in disabled
    parsed = forms(source)
    for symbol in (
        "et_g3c4_private_p2g2_eos_first_frame_carrier_v1",
        "et_g3c4_private_p2g2_second_frame_carrier_v1",
        "et_g3c4_private_p2g2_terminal_commit_v1",
        "et_g3c4_private_p2g2_terminal_snapshot_v1",
        "et_g3c4_private_p2g2_terminal_ids_clone_v1",
        "et_g3c4_private_p2g2_terminal_ids_clone_copy_v1",
        "et_g3c4_private_p2g2_terminal_ids_clone_release_v1",
    ):
        assert source.count(":real " + symbol) == 1, symbol
    close = definition(parsed, "g3c4-p2g2-terminal-close-core-private")
    probe = definition(parsed, "g3c4-p2g2-terminal-close-probed-private")
    assert calls(probe, "g3c4-p2g2-terminal-close-core-private") == [
        ["g3c4-p2g2-terminal-close-core-private", "generator", "input",
         "before-terminal", "repair-lease", "#f", "#f", "#f"]]
    published = definition(parsed, "g3c4-p2g2-terminal-close-private")
    assert calls(published, "g3c4-p2g2-terminal-close-probed-private") == [
        ["g3c4-p2g2-terminal-close-probed-private", "generator", "input",
         ["lambda", ["call", "row"], "#t"],
         ["lambda", ["call", "row"], "#f"]]]
    origin = definition(parsed, "g3c4-p2g2-terminal-close-origin-private")
    assert calls(origin, "g3c4-p2g2-terminal-close-core-private") == [
        ["g3c4-p2g2-terminal-close-core-private", "generator", "input",
         ["lambda", ["call", "row"], "#t"],
         ["lambda", ["call", "row"], "#f"], "#t", "#f", "#f"]]
    origin_check = definition(parsed,
                              "g3c4-p2g2-terminal-origin-prezero-check")
    assert ["procedure?", ["cdr", "binding"]] in calls(
        origin_check, "procedure?")
    assert ["cdr", "binding"] in calls(origin_check, "cdr")
    for name in ("generator", "generator-entry", "input", "input-entry",
                 "row", "anchor"):
        assert ["eq?", ["vector-ref", "origin-row",
                        {"generator": "3", "generator-entry": "4",
                         "input": "5", "input-entry": "6", "row": "7",
                         "anchor": "8"}[name]], name] in calls(origin_check, "eq?")
    assert len(calls(close, "g3c4-p2g2-terminal-origin-prezero-check")) == 1
    before_index = source.index("(before-terminal call row)")
    assert before_index < source.index(
        "(g3c4-p2g2-terminal-origin-prezero-check\n", before_index) < source.index(
        "(if (= (g3c4-p2g2-native-terminal\n", before_index)
    for root in ("rows", "anchors"):
        symbol = "g3c4-p2g2-terminal-origin-" + root
        assert ["vector-set!", symbol, "0", "old-origin-" + root] in calls(
            close, "vector-set!")
        assert ["vector-set!", symbol, "0",
                ["cons", "origin-" + ("row" if root == "rows" else "anchor"),
                 "old-origin-" + root]] in calls(close, "vector-set!")
    assert ["vector-set!", "g3c4-p2g2-terminal-origin-bindings", "0",
            "old-origin-bindings"] in calls(close, "vector-set!")
    assert ["vector-set!", "g3c4-p2g2-terminal-origin-bindings", "0",
            ["cons", "origin-binding", "old-origin-bindings"]] in calls(
        close, "vector-set!")
    for slot, bound in (("3", "bound-generator"),
                        ("4", "bound-generator-entry"),
                        ("5", "bound-input"),
                        ("6", "bound-input-entry")):
        for owner in ("candidate-row", "candidate-anchor"):
            assert ["eq?", ["vector-ref", owner, slot], bound] in calls(
                close, "eq?")
    origin_live = definition(parsed, "g3c4-p2g2-terminal-origin-live")
    assert len(calls(origin_live, "g3c4-p2g2-terminal-live")) == 1
    assert ["procedure?", ["cdr", "binding"]] in calls(
        origin_live, "procedure?")
    assert ["eq?", ["vector-ref", "row", "7"], "terminal"] in calls(
        origin_live, "eq?")
    assert ["eq?", ["vector-ref", "row", "9"], "'live"] in calls(
        origin_live, "eq?")
    for origin_slot, terminal_slot in (("10", "14"), ("11", "15"),
                                       ("12", "16")):
        for owner in ("row", "anchor"):
            assert ["=", ["vector-ref", owner, origin_slot],
                    ["vector-ref", "terminal", terminal_slot]] in calls(
                origin_live, "=")
    assert len(calls(close, "m3-call")) == 1
    carriers = definition(parsed, "g3c4-p2g2-terminal-carriers?")
    extents = {"first-raw": "1", "first-stage": "8",
               "second-raw": "1", "second-stage": "8",
               "text-one": "1", "text-two": "2",
               "pending-raw": "2", "pending-stage": "16"}
    for name, size in extents.items():
        assert any(call == ["=", ["bytevector-length", name], size]
                   for call in calls(carriers, "=")), (name, size)
    for size in ("1", "2", "8"):
        assert calls(close, "make-bytevector")
        assert any(call[1:3] == [size, "0"]
                   for call in calls(close, "make-bytevector")), size
    assert len(calls(close, "g3c4-p2g2-native-eos-first")) == 1
    assert len(calls(close, "g3c4-p2g2-native-second")) == 1
    assert len(calls(close, "t1-private-g3-decode-raw-into!")) == 2
    assert len(calls(close, "g3c4-p2g2-native-prefix-commit")) == 1
    assert len(calls(close, "g3c4-p2g2-native-terminal")) == 1
    assert len(calls(close, "before-terminal")) == 1
    assert len(calls(close, "repair-lease")) == 1
    active = definition(parsed, "g3c4-p2g2-terminal-active-check")
    for slot, terminal_slot in (("11", "3"), ("12", "4"),
                                ("9", "17"), ("10", "18")):
        assert ["eq?", ["vector-ref", "record", slot],
                ["vector-ref", "row", terminal_slot]] in calls(active, "eq?")
    for slot in range(15):
        row_slot = (9, 10, 11, 12, 3, 4, 5, 6, 7, 8,
                    17, 18, 0, 1, 2)[slot]
        assert ["eq?", ["vector-ref", "row", str(row_slot)],
                ["vector-ref", "originals", str(slot)]] in calls(active, "eq?")
    native = calls(close, "g3c4-p2g2-native-terminal")[0]
    assert native[3:5] == ["first-raw", "9"]
    assert native[5] == ["if", ["=", "count", "1"], "'()", "second-raw"]
    assert native[6] == ["if", ["=", "count", "1"], "0", "9"]
    assert calls(close, "g3c4-p2g2-terminal-active-check")

    barriers = [part for part in walk(close)
                if part == ["vector-set!", "barrier", "0", "#t"]]
    assert len(barriers) == 1
    parents = [part for part in walk(close)
               if part and part[0] == "begin" and barriers[0] in part]
    assert len(parents) == 1
    tail = parents[0]
    assert tail[1] == barriers[0]
    # The irreversible tail may only read captured vector values, compare
    # fixed count, and write captured vectors. No allocation, native call,
    # registry lookup, decode, validation, or recoverable callback follows.
    approved = {"begin", "vector-set!", "vector-ref", "bytevector-u8-set!",
                "if", "="}
    for part in walk(tail):
        if part and isinstance(part[0], str):
            assert part[0] in approved, part[0]
    assert calls(tail, "vector-set!")
    assert ["vector-set!", "output", "1", "'terminal-output"] in tail
    assert ["vector-set!", "output", "2", "'live"] in tail
    assert ["vector-set!", "anchor", "13", "'live"] in tail
    assert ["vector-set!", "origin-row", "9", "'live"] in calls(
        tail, "vector-set!")
    assert ["vector-set!", "origin-anchor", "9", "'live"] in calls(
        tail, "vector-set!")
    for slot, value in (("14", "count"), ("15", "selected-first"),
                        ("16", "selected-second")):
        assert ["vector-set!", "anchor", slot, value] in tail
    assert calls(tail, "bytevector-u8-set!") == [
        ["bytevector-u8-set!", "first-raw", "0", "0"],
        ["bytevector-u8-set!", "second-raw", "0", "0"]]
    assert ["vector-ref", "output", "0"] == tail[-1]
    live = definition(parsed, "g3c4-p2g2-terminal-live")
    for slot in ("14", "15", "16"):
        assert ["=", ["vector-ref", "row", slot],
                ["vector-ref", "anchor", slot]] in calls(live, "=")

    snapshot = definition(parsed, "g3c4-p2g2-terminal-snapshot-private")
    assert any(call[-2:] == ["destination", "72"]
               for call in calls(snapshot, "g3c4-p2g2-native-snapshot"))
    assert any(call[1:3] == ["64", "0"]
               for call in calls(snapshot, "make-bytevector"))
    copy = definition(parsed, "g3c4-p2g2-terminal-ids-clone-copy-private")
    assert calls(copy, "g3c4-p2g2-native-ids-copy") == [
        ["g3c4-p2g2-native-ids-copy", "native", "count", "bytes",
         ["+", "8", ["*", "count", "8"]]]]
    assert calls(copy, "make-bytevector")
    release = definition(parsed, "g3c4-p2g2-terminal-release-private!")
    assert len(calls(release, "g3c4-native-output-release")) == 1
    assert len(calls(release, "g3c4-p2g2-terminal-origin-live")) == 1
    assert any(part[:2] == ["if", ["or", "origin-row", "origin-anchor",
                                  "origin-binding"]]
               for part in walk(release) if isinstance(part, list))
    assert ["vector-set!", "origin-row", "9", "'dead"] in calls(
        release, "vector-set!")
    assert ["vector-set!", "origin-anchor", "9", "'dead"] in calls(
        release, "vector-set!")
    release_source = source[source.index(
        "(define (g3c4-p2g2-terminal-release-private! shell)"):source.index(
        ";;; Clone record:")]
    assert release_source.index(
        "(g3c4-p2g2-terminal-origin-live shell operation)") < release_source.index(
        "(g3c4-native-output-release native)")
    assert release_source.index(
        "(g3c4-p2g2-terminal-origin-without-binding shell)") < release_source.index(
        "(g3c4-native-output-release native)") < release_source.index(
        "(vector-set! g3c4-p2g2-terminal-origin-bindings")
    assert len(calls(definition(parsed, "g3c4-p2g2-terminal-ids-clone-private"),
                     "g3c4-p2g2-native-ids-clone")) == 1


def main() -> None:
    validate(SOURCE.read_text(encoding="utf-8"),
             CLOSURE.read_text(encoding="utf-8").splitlines(),
             BASE.read_text(encoding="utf-8").splitlines())
    print("G3-C4 P2/G2 terminal-close source/zero-tail PASS")


if __name__ == "__main__":
    main()
