#!/usr/bin/env python3
"""Source-only successor check for the feature-local P1 candidate handshake."""

from hashlib import sha256
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
legacy = runpy.run_path(str(ROOT / "scripts/check-e3-p1-contract.py"))
forms, atom, digest, signature = (
    legacy[name] for name in ("forms", "atom", "digest", "signature")
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check():
    subprocess.run(["bash", str(ROOT / "scripts/generate-p1-roots.sh"), "--check"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
    nodes = forms(source)
    surfaces = [node for node in nodes if isinstance(node[2], list)
                and len(node[2]) > 1 and atom(node[2][0]) == "define"
                and atom(node[2][1]) == "p1-trusted-surface"]
    require(len(surfaces) == 1, "expected one lexical P1 surface")
    vector = surfaces[0][2][2][2][-1]
    require(atom(vector[2][0]) == "vector" and len(vector[2]) == 77,
            "expected exactly 76 trusted slots")
    for start, stop, expected in (
        (1, 65, "d7d3d3a9aff533aa5f3422130d702f730b3cf227c3ecdd0a56423a634fec7d5e"),
        (65, 70, "f2a34d77e5d6db808dcc7bf5b5eeeacbfcda8da274827aa4b9e94ac0fdfa0097"),
        (70, 72, "2bbc3bfad143d3f17ca21db4180cde453622d86cc7dff267a0276b2bd6e66a15"),
        (74, 77, "76dc9c821ed3d5d3ad52697e34bac9863e3b9a49b69eb2780839d342bee1a2a3"),
    ):
        require(digest(source, vector[2][start:stop]) == expected,
                f"trusted slot span {start - 1}..{stop - 2} changed")
    require([signature(node) for node in vector[2][72:74]] == [
        ["lambda", ["identity"], ["construction-prepare-eval-guarded!", "identity"]],
        ["lambda", ["identity"], ["construction-seal-prepared!", "identity"]],
    ], "legacy prepared slots 71..72 changed")

    provides = [node for node in nodes if isinstance(node[2], list)
                and node[2] and atom(node[2][0]) == "provide"]
    require(len(provides) == 1, "expected one trusted provide")
    names = [atom(node) for node in provides[0][2][1:]]
    manifest = [line.split("\t", 1)[1] for line in
                (ROOT / "tools/p1/module_surface.tsv").read_text().splitlines()]
    require(names == [*manifest, "p1-trusted-surface"],
            "ordinary trusted package export manifest changed")
    for path, expected in (
        ("lib/transformer/module.esk", "38be7a65d467753dc53abccf5195107d939d842e5988783d8f0073f4b9c9f46c"),
        ("tools/p1/module_surface.tsv", "e302fb8c7545d450b5365460393c41b61c9bba8840de27b3c3f6abfcec714104"),
    ):
        require(sha256((ROOT / path).read_bytes()).hexdigest() == expected,
                f"inherited package surface changed: {path}")
    require("et_p1_private_candidate_" not in source,
            "feature-off root directly links candidate native symbols")

    extension = (ROOT / "native/g3r_p1_candidate_construction_extension.esk").read_text()
    extension_forms = [signature(node) for node in forms(extension)]
    require(len(extension_forms) == 5
            and extension_forms[0][-1] == "et_p1_private_candidate_construction_begin_v1"
            and extension_forms[1][-1] == "et_p1_private_candidate_graph_preflight_v1"
            and extension_forms[2][1][0] == "module-candidate-construction-begin-internal"
            and extension_forms[3][1][0] == "module-candidate-construction-preflight-internal!"
            and extension_forms[4][0][1] == "p1-trusted-surface",
            "candidate wrappers are not feature-local and exact")
    print("G3-R P1 candidate STRUCTURE PASS: legacy slots 0..72, exact slots 73..75, unchanged package provides")


if __name__ == "__main__":
    try:
        check()
    except (ValueError, IndexError, OSError, subprocess.CalledProcessError) as error:
        print(f"G3-R P1 candidate STRUCTURE FAIL: {error}", file=sys.stderr)
        sys.exit(1)
