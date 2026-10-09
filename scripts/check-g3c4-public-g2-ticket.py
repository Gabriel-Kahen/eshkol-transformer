#!/usr/bin/env python3
"""Source-only admission for the private public-G2 prezero ticket seam."""

from __future__ import annotations

import hashlib
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parse = SourceFileLoader(
    "terminal_source_shape", str(ROOT / "scripts/check-g3c4-p2g2-terminal-close.py")
).load_module()
BASE = ROOT / "native/g3c4_p2g2_terminal_close_source_closure.txt"
CLOSURE = ROOT / "native/g3c4_public_g2_ticket_source_closure.txt"
ADDITIONS = [
    "native/g3c4_public_output_adapter_extension.esk",
    "native/g3g_manual_package_root.esk",
    "native/g3g_manual_package_source_closure.txt",
    "scripts/check-g3g-manual-package.py",
    "scripts/check-g3c4-public-g2-ticket.py",
    "tests/g3c4/test_public_g2_ticket_source.py",
    "tests/g3c4/p2_g1_g2_prefix_stub_negative_test.esk",
    "tests/g3c4/p2_g1_g2_core_stub_negative_test.esk",
    "tests/g3c4/p2_g2_public_ticket_close_test.esk",
    "docs/g3/G3_C4_PUBLIC_G2_PREZERO_TICKET_SOURCE_LEAF.md",
    "native/g3c4_public_g2_ticket_source_closure.txt",
]
ADAPTER_SHA256 = "3d55f8737b6fcfc280a9c4094362d69eaa1d005002cf1b604f2464d6f40d2139"
PRIVATE_LOADS = [
    "g3c4_p2g2_ownership_extension.esk",
    "g3c4_p2g2_protected_scope_extension.esk",
    "g3c4_p2g2_t1_prefix_coordinator_extension.esk",
    "g3c4_p2g2_terminal_close_extension.esk",
]


def top_define(forms: list[object], name: str) -> list[object]:
    matches = [form for form in forms if isinstance(form, list)
               and len(form) >= 3 and form[0] == "define"
               and (form[1] == name or
                    (isinstance(form[1], list) and form[1][0] == name))]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def validate(dispatch: str, terminal: str, call_entry: str, adapter: bytes,
             fixture: str, g2_close_fixture: str,
             package_root: str, package_sources: list[str],
             prefix_negative: str, core_negative: str,
             closure: list[str], base: list[str]) -> None:
    assert closure == base + ADDITIONS
    assert len(closure) == len(set(closure))
    assert all((ROOT / item).is_file() for item in closure)
    assert hashlib.sha256(adapter).hexdigest() == ADAPTER_SHA256
    parsed = parse.forms(dispatch)
    top_define(parsed, "g3c4-public-profile-config")
    creator = top_define(parsed, "g3c4-public-generator-create")
    original = top_define(parsed, "g3c4-public-g2-profile-original")
    assert "'diagnostic-c4-p2g2" in dispatch
    assert len(parse.calls(creator, "g3c4-public-g2-generator-create")) == 1
    assert len(parse.calls(original, "g3c4-public-g2-root-find")) == 3
    assert "(<= (vector-ref policy 6) 255)" in dispatch
    assert "(eq? (vector-ref row 6) 'diagnostic-c4-p2g2)" in dispatch
    assert not any(isinstance(form, list) and form[:2] == ["define", "g3c4-public-g2-bridge"]
                   for form in parsed)
    generate = top_define(parsed, "g3c4-public-generator-generate")
    assert generate == ["define", ["g3c4-public-generator-generate", "generator", "input"],
                        ["exit", "134"]]
    route = top_define(parsed, "g3c4-p2g2-terminal-close-public-private")
    assert route[1] == "g3c4-p2g2-terminal-close-public-private"
    assert route[2][0] == "let"
    assert route[2][1] == [["secret", ["vector", "'g3c4-public-g2-call-secret"]]]
    setters = [part for part in parse.walk(route) if part[:2] ==
               ["set!", "g3c4-public-generator-generate"]]
    assert len(setters) == 1 and setters[0][2][0] == "lambda"
    assert setters[0][2][1] == ["generator", "input"]
    assert parse.calls(setters[0], "bridge") == [[
        "bridge", "'generate", "generator", "input", "#f", "#f", "#f", "#f",
        "0", "-1", "-1", "'generator-generate!"]]
    budget_branches = [part for part in parse.walk(setters[0])
                       if part and part[0] == "if" and len(part) >= 4
                       and part[1] == ["=", ["vector-ref",
                                                 ["vector-ref", "generator-entry", "7"], "5"],
                                       "2"]]
    assert len(budget_branches) == 1
    assert len(parse.calls(budget_branches[0][2], "bridge")) == 1
    assert not parse.calls(budget_branches[0][3], "bridge")
    assert len(parse.calls(budget_branches[0][3], "g3c4-native-p2g1-preflight")) == 1
    close = top_define(parsed, "g3c4-public-generator-close")
    dead_branches = [part for part in parse.walk(close)
                     if part and part[0] == "if" and len(part) >= 4
                     and part[1] == ["eq?", ["vector-ref", "member", "2"], "'dead"]]
    assert len(dead_branches) == 1
    assert dead_branches[0][2][-1] == ["g3c4-generator-close-internal!", "generator"]
    assert len(parse.calls(dead_branches[0][2], "g3c4-native-generator-close")) == 0
    assert len(parse.calls(dead_branches[0][3], "g3c4-public-g2-profile-live")) == 1
    assert len(parse.calls(dead_branches[0][3], "g3c4-generator-entry-live")) == 1
    internal = top_define(parse.forms(call_entry), "g3c4-generator-close-internal!")
    conditions = [part for part in parse.walk(internal) if part and part[0] == "cond"]
    assert len(conditions) == 1
    assert conditions[0][1] == [["g3c4-dead-entry?", "entry", "'generator"], "#t"]
    assert len(parse.calls(route, "g3c4-p2g2-terminal-close-core-private")) == 1
    assert len(parse.calls(route, "g3c4-p2g2-terminal-close-public-private")) == 1
    assert [part for part in parse.walk(route) if part[:2] == ["bridge", "'admit"]]
    assert [part for part in parse.walk(route) if part[:2] == ["bridge", "'verify"]] == []
    assert "(g3c4-p2g2-terminal-close-public-private\n               generator input issued)" in dispatch
    assert dispatch.count("(eq? (vector-ref ticket 0) secret)") == 2
    assert dispatch.count("(eq? (vector-ref ticket 12) 'pending)") == 2
    assert "(eq? (g3c4-public-g2-profile-original" in dispatch
    assert "(eq? (g3c4-input-entry-live input operation)" in dispatch
    assert "g3c4-public-g2-output-bindings 0))\n                                    0)" in dispatch

    terminal_forms = parse.forms(terminal)
    core = top_define(terminal_forms, "g3c4-p2g2-terminal-close-core-private")
    assert core[1] == ["g3c4-p2g2-terminal-close-core-private", "generator", "input",
                       "before-terminal", "repair-lease", "origin-enabled",
                       "public-ticket", "public-stage"]
    assert "g3c4-public-g2-bridge" not in terminal
    assert "g3c4-p2g2-terminal-close-public-private" not in terminal
    zero = terminal.index("(g3c4-p2g2-native-terminal\n")
    for action in ("'restore", "'bind", "'verify"):
        assert "(public-stage " + action in terminal
        assert terminal.index("(public-stage " + action) < zero
    assert "public-stage" not in terminal[zero:terminal.index("(define (g3c4-p2g2-terminal-close-probed-private")]
    assert ["#f", "#f", "#f"] == parse.calls(
        top_define(terminal_forms, "g3c4-p2g2-terminal-close-probed-private"),
        "g3c4-p2g2-terminal-close-core-private")[0][-3:]

    order = ["g3c4_p2g1_coordinator_extension.esk", *PRIVATE_LOADS,
             "g3c4_public_output_adapter_extension.esk",
             "g3c4_public_generation_dispatch_extension.esk"]
    assert all(f'(load "{name}")' in package_root for name in order)
    positions = [package_root.index(f'(load "{name}")') for name in order]
    assert positions == sorted(positions)
    assert all(package_root.count(f'(load "{name}")') == 1 for name in order)
    assert len(package_sources) == 56 and len(package_sources) == len(set(package_sources))
    assert [f"native/{name}" for name in order] == [
        item for item in package_sources if item in {f"native/{name}" for name in order}]
    assert all((ROOT / item).is_file() for item in package_sources)
    assert "g3c4-p2g2-prefix-preflight" in fixture
    assert "(check \"authentic dead P2/G1 generator close stays idempotent\"\n  (g3c4-public-generator-close greedy))" in fixture
    assert "malformed dead P2/G1 generator rejects" in fixture
    assert "invalid phase P2/G1 generator rejects" in fixture
    assert "authentic dead public G2 generator close stays idempotent" in g2_close_fixture
    assert "malformed dead public G2 generator rejects" in g2_close_fixture
    assert "missing public G2 profile root blocks live close" in g2_close_fixture
    assert parse.forms(g2_close_fixture)[0] == ["load", '"g3g_manual_package_root.esk"']
    fixture_forms = parse.forms(fixture)
    for name, arguments in (
        ("g3c4-p2g2-prefix-preflight", ["generator", "input"]),
        ("g3c4-p2g2-terminal-close-core-private",
         ["generator", "input", "before-terminal", "repair-lease",
          "origin-enabled", "public-ticket", "public-stage"]),
    ):
        stub = top_define(fixture_forms, name)
        assert stub == ["define", [name, *arguments], ["exit", "134"]]
        assert name not in package_root
    assert prefix_negative.strip().endswith("(g3c4-p2g2-prefix-preflight #f #f)")
    assert parse.forms(core_negative)[-1] == [
        "g3c4-p2g2-terminal-close-core-private", "#f", "#f",
        ["lambda", ["call", "row"], "#f"],
        ["lambda", ["call", "row"], "#f"], "#f", "#f", "#f"]
    assert all("p2_g1_public_generation_test.esk" in text
               for text in (prefix_negative, core_negative))


def main() -> None:
    read = lambda path: (ROOT / path).read_text()
    validate(
        read("native/g3c4_public_generation_dispatch_extension.esk"),
        read("native/g3c4_p2g2_terminal_close_extension.esk"),
        read("native/g3c4_call_entry_extension.esk"),
        (ROOT / "native/g3c4_public_output_adapter_extension.esk").read_bytes(),
        read("tests/g3c4/p2_g1_public_generation_test.esk"),
        read("tests/g3c4/p2_g2_public_ticket_close_test.esk"),
        read("native/g3g_manual_package_root.esk"),
        read("native/g3g_manual_package_source_closure.txt").splitlines(),
        read("tests/g3c4/p2_g1_g2_prefix_stub_negative_test.esk"),
        read("tests/g3c4/p2_g1_g2_core_stub_negative_test.esk"),
        CLOSURE.read_text().splitlines(), BASE.read_text().splitlines())
    print("G3-C4 public G2 prezero ticket source PASS")


if __name__ == "__main__":
    main()
