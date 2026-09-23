#!/usr/bin/env python3
"""Development-only structural checks; not a mode/lifetime runtime proof."""

from hashlib import sha256
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "ba0e37d06076d0a16c473ff742ab95723cb2cb89"
SOURCE = "internal/p1/lib/transformer/module.esk"
WRAPPERS = "native/e3_p1_modes_extension.esk"
TR3_WRAPPERS = "native/tr3_p1_fixed_set_extension.esk"
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
    require(len(vector[2]) == 72, "expected exactly 71 trusted closure slots")
    require(digest(text, vector[2][1:65]) ==
            "0f054d3f4e2f5466c77514f98d38aed5421790d8d9a7e6a73a9fc684775326f5",
            "inherited closure 0..63 changed from accepted base")
    require(digest(text, vector[2][65:70]) ==
            "f2a34d77e5d6db808dcc7bf5b5eeeacbfcda8da274827aa4b9e94ac0fdfa0097",
            "accepted E3 closure slots 64..68 changed")
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
    require(len(e3_definitions) == 23 and digest(text, e3_definitions) ==
            "69d85636fffe58568c649308796a40d870d7fea7d78d54e88d0c0cb20149da02",
            "accepted E3 lexical definitions changed")
    require(digest(text, (n for n in nodes if n is not surface)) ==
            "0c8621e2248ad188553e25c9a3a6d103d0cfe122617545847eb15e450d90b098",
            "P1 top-level authority/provides/wrappers changed from accepted base")
    require(digest(text, externs(nodes)) ==
            "d970a2c162b05acd64ce94214ac0f56fb37817c8d51bbd6e78aa31b54db9985b",
            "P1 native extern authority changed from accepted base")
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
    for prefix, count in INHERITED_CLOSURES.items():
        path = ROOT / f"native/{prefix}_source_closure.txt"
        paths = path.read_text().splitlines()
        require(len(paths) == count and paths.count(SOURCE) == 1,
                f"predecessor source count/canonical P1 identity drifted: {path.name}")
        require(WRAPPERS not in paths,
                f"predecessor source closure gained E3-only wrappers: {path.name}")
    print(f"E3-P1 STRUCTURE PASS: base={BASE} first69=unchanged total=71 public=unchanged")
    for path in (SOURCE, WRAPPERS, TR3_WRAPPERS):
        print(f"sha256 {sha256((ROOT / path).read_bytes()).hexdigest()} {path}")


if __name__ == "__main__":
    try:
        check()
    except (ValueError, IndexError, OSError) as error:
        print(f"E3-P1 STRUCTURE FAIL: {error}", file=sys.stderr)
        sys.exit(1)
