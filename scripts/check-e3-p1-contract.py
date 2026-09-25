#!/usr/bin/env python3
"""Development-only structural checks; not a mode/lifetime runtime proof."""

from hashlib import sha256
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "dd4f1d4b090fb7d4ace907733a0232575be99997"
SOURCE = "internal/p1/lib/transformer/module.esk"
WRAPPERS = "native/e3_p1_modes_extension.esk"
TR3_WRAPPERS = "native/tr3_p1_fixed_set_extension.esk"
G3C4_WRAPPERS = "native/g3c4_p1_construction_extension.esk"
INHERITED_CLOSURES = {
    "p1_package": 4, "c1_checkpoint": 8, "t1_wave1": 10,
    "t2_wave2": 12, "t2_wave2_d1_test": 13, "i2_wave2": 13,
    "o2_wave2": 15, "d2_wave2": 18, "k2_wave2": 15,
    "c2_wave2": 32, "m3t_package": 15, "m3_package": 18,
    "c2_checkpoint_load": 25, "c2_checkpoint_save": 27,
    "c2_d2_cursor_pair": 19, "c2_model_encode": 14,
    "c2_o2_encode": 17, "c2_training_state": 25,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def forms(text):
    """Read balanced source spans, without evaluating any Eshkol code."""
    stack, result = [], []
    for match in re.finditer(r';[^\n]*|"(?:\\.|[^"\\])*"|[()]|[^\s();"]+', text):
        token = match.group()
        if token.startswith(";"):
            continue
        if token == ")":
            require(stack, "unmatched closing parenthesis")
            stack.pop()[1] = match.end()
            continue
        node = [match.start(), match.end(), [] if token == "(" else token]
        (stack[-1][2] if stack else result).append(node)
        if token == "(":
            stack.append(node)
    require(not stack, "unclosed source form")
    return result


def atom(node):
    return node[2] if isinstance(node[2], str) else None


def digest(text, nodes):
    return sha256("\0".join(text[n[0]:n[1]] for n in nodes).encode()).hexdigest()


def signature(node):
    return node[2] if isinstance(node[2], str) else [signature(n) for n in node[2]]


def externs(nodes):
    for node in nodes:
        if isinstance(node[2], list):
            if node[2] and atom(node[2][0]) == "extern":
                yield node
            yield from externs(node[2])


def check():
    text = (ROOT / SOURCE).read_text()
    nodes = forms(text)
    surfaces = [n for n in nodes if isinstance(n[2], list) and len(n[2]) > 1
                and atom(n[2][0]) == "define" and atom(n[2][1]) == "p1-trusted-surface"]
    require(len(surfaces) == 1, "expected one canonical lexical P1 surface")
    surface = surfaces[0]
    lexical = surface[2][2]
    require(atom(lexical[2][0]) == "let", "P1 authority must remain lexical")
    vector = lexical[2][-1]
    require(atom(vector[2][0]) == "vector", "P1 surface must end with its vector")
    require(len(vector[2]) == 74, "expected exactly 73 trusted closure slots")
    require(digest(text, vector[2][1:65]) ==
            "d7d3d3a9aff533aa5f3422130d702f730b3cf227c3ecdd0a56423a634fec7d5e",
            "inherited closure 0..63 changed from accepted #121 base")
    require(digest(text, vector[2][65:70]) ==
            "f2a34d77e5d6db808dcc7bf5b5eeeacbfcda8da274827aa4b9e94ac0fdfa0097",
            "accepted E3 closure slots 64..68 changed")
    require(digest(text, vector[2][70:72]) ==
            "2bbc3bfad143d3f17ca21db4180cde453622d86cc7dff267a0276b2bd6e66a15",
            "accepted TR3 closure slots 69..70 changed")
    g3c4_slots = [
        ["lambda", ["identity"],
         ["construction-prepare-eval-guarded!", "identity"]],
        ["lambda", ["identity"],
         ["construction-seal-prepared!", "identity"]],
    ]
    require([signature(n) for n in vector[2][72:74]] == g3c4_slots,
            "G3-C4 private closures differ from exact slot71/72 calls")
    e3_definitions = []
    for node in lexical[2][1:-1]:
        if not (isinstance(node[2], list) and len(node[2]) > 1
                and atom(node[2][0]) == "define"):
            continue
        target = node[2][1]
        name = atom(target)
        if name is None and isinstance(target[2], list) and target[2]:
            name = atom(target[2][0])
        if name and name.startswith("e3-mode-"):
            e3_definitions.append(node)
    # ca21861 added the reviewed active-record removal helper and changed
    # enrollment to consult the bounded active head. The lexical E3 slice is
    # identical from that commit through the current P1 index integration.
    require(len(e3_definitions) == 24 and digest(text, e3_definitions) ==
            "8dcb04c2adb943374c40da939637f3331af574b5cf1a07ae3976e8211f1d669a",
            "accepted E3 lexical definitions changed")
    g3c4_externs = [
        ["extern", "i64", "p1-native-construction-prepare", "ptr", "ptr",
         ":real", "et_p1_private_construction_prepare_v1"],
        ["extern", "i64", "p1-native-construction-commit-prepared", "ptr",
         "ptr", ":real", "et_p1_private_construction_commit_prepared_v1"],
        ["extern", "i64", "p1-native-construction-abort-prepared", "ptr",
         "ptr", ":real", "et_p1_private_construction_abort_prepared_v1"],
    ]
    require(sum(signature(n) in g3c4_externs for n in externs(nodes)) == 3,
            "G3-C4 native extern delta is missing or duplicated")
    require(digest(text, (n for n in nodes if n is not surface
                          and signature(n) not in g3c4_externs)) ==
            "505b1c59f9f2734e4ce457423129ca426fd300c748076e0fac03c9d84762d34f",
            "P1 top-level authority/provides/wrappers changed from accepted #121 base")
    require(digest(text, (n for n in externs(nodes)
                          if signature(n) not in g3c4_externs)) ==
            "616535438e9ebcb1b5adc240d96a6f837e4d32f45de1ee59983c1b91dd05518e",
            "P1 native extern authority changed from accepted #121 base")
    for path, expected in {
        "lib/transformer/module.esk":
            "38be7a65d467753dc53abccf5195107d939d842e5988783d8f0073f4b9c9f46c",
        "tools/p1/module_surface.tsv":
            "e302fb8c7545d450b5365460393c41b61c9bba8840de27b3c3f6abfcec714104",
    }.items():
        require(sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                f"unchanged inherited public/named surface drifted: {path}")
    template = (ROOT / "templates/p1/module_roots.esk.tmpl").read_text()
    trusted = template.split("@@P1_TRUSTED_BEGIN@@\n", 1)[1].split(
        "@@P1_TRUSTED_END@@", 1)[0]
    require(trusted == text, "canonical trusted source differs from generated template")
    wrapper_text = (ROOT / WRAPPERS).read_text()
    expected = []
    for slot, operation in enumerate(("bind!", "prepare!", "enter!", "restore!", "unbind!"), 64):
        args = ["model", "setup-box"] if slot == 64 else ["token"]
        expected.append(["define", ["p1-e3-mode-" + operation, *args],
                         [["vector-ref", "p1-trusted-surface", str(slot)], *args]])
    require([signature(n) for n in forms(wrapper_text)] == expected,
            "private wrappers differ from the five exact slot/arity calls")
    tr3_wrapper_text = (ROOT / TR3_WRAPPERS).read_text()
    tr3_expected = [
        ["define", ["tr3-p1-fixed-capture-internal", "model", "handles",
                    "modules", "retained-handles", "carriers"],
         [["vector-ref", "p1-trusted-surface", "69"], "model", "handles",
          "modules", "retained-handles", "carriers"]],
        ["define", ["tr3-p1-fixed-recheck-internal", "model", "modules",
                    "retained-handles", "carriers"],
         [["vector-ref", "p1-trusted-surface", "70"], "model", "modules",
          "retained-handles", "carriers"]],
    ]
    require([signature(n) for n in forms(tr3_wrapper_text)] == tr3_expected,
            "TR3 private wrappers differ from exact slot69/70 arities")
    g3c4_wrapper_text = (ROOT / G3C4_WRAPPERS).read_text()
    g3c4_expected = [
        ["define", ["module-construction-prepare-eval-internal!", "identity"],
         [["vector-ref", "p1-trusted-surface", "71"], "identity"]],
        ["define", ["module-construction-seal-prepared-internal!", "identity"],
         [["vector-ref", "p1-trusted-surface", "72"], "identity"]],
    ]
    require([signature(n) for n in forms(g3c4_wrapper_text)] == g3c4_expected,
            "G3-C4 private wrappers differ from exact slot71/72 arities")
    for prefix, count in INHERITED_CLOSURES.items():
        path = ROOT / f"native/{prefix}_source_closure.txt"
        paths = path.read_text().splitlines()
        require(len(paths) == count and paths.count(SOURCE) == 1,
                f"predecessor source count/canonical P1 identity drifted: {path.name}")
        require(WRAPPERS not in paths,
                f"predecessor source closure gained E3-only wrappers: {path.name}")
        require(G3C4_WRAPPERS not in paths,
                f"predecessor source closure gained G3-C4-only wrappers: {path.name}")
    print(f"E3-P1 STRUCTURE PASS: base={BASE} first71=accepted-exact total=73 public=unchanged")
    for path in (SOURCE, WRAPPERS, TR3_WRAPPERS, G3C4_WRAPPERS):
        print(f"sha256 {sha256((ROOT / path).read_bytes()).hexdigest()} {path}")


if __name__ == "__main__":
    try:
        check()
    except (ValueError, IndexError, OSError) as error:
        print(f"E3-P1 STRUCTURE FAIL: {error}", file=sys.stderr)
        sys.exit(1)
