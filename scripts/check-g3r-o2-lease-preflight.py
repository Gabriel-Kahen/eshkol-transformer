#!/usr/bin/env python3
"""Source closure checks for the private G3-R O2/TR3 admission leaf."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
scheme = (ROOT / "native/g3r_candidate_o2_lease_preflight_extension.esk").read_text()
fixture = (ROOT / "tests/g3r/o2_candidate_lease_preflight_test.esk").read_text()
o2_header = (ROOT / "native/o2_optimizer_internal.h").read_text()
o2_native = (ROOT / "native/o2_optimizer.c").read_text()
m3t_header = (ROOT / "src/eshkol_transformer/m3t_transport.h").read_text()
m3t_native = (ROOT / "src/eshkol_transformer/m3t_transport.c").read_text()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def balanced(source: str) -> bool:
    depth = 0
    quoted = False
    escaped = False
    commented = False
    for char in source:
        if commented:
            if char == "\n":
                commented = False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == ";":
            commented = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted


for name, source in (("extension", scheme), ("fixture", fixture)):
    require(balanced(source), f"{name} has unbalanced Scheme delimiters")

require(
    ":real et_m3t_private_candidate_o2_graph_preflight_v1" in scheme,
    "candidate O2/M3T status ABI is missing",
)
for name in (
    "g3r-candidate-o2-enroll-internal!",
    "g3r-candidate-o2-lease-preflight-internal!",
    "g3r-candidate-tr3-lease-clear-internal!",
):
    require(f"(define ({name}" in scheme, f"private {name} missing")
require("tr3-lease-record-overlap?" in scheme, "TR3 exact overlap missing")
require("tr3-p1-fixed-capture-internal" in scheme, "TR3 graph capture missing")
require("g3r-candidate-graph-identity-preflight-internal!" in scheme,
        "accepted candidate graph preflight missing")
require("o2-require-optimizer" in scheme, "registered O2 shell lookup missing")
require("et_o2_private_candidate_retire_preflight_v1(candidate, error)" in o2_native,
        "native O2 liveness admission missing")
require("et_o2_private_candidate_graph_entry_match_v1" in o2_header + o2_native,
        "native O2 per-entry identity seam missing")
require("et_m3t_private_candidate_o2_graph_preflight_v1" in m3t_header + m3t_native,
        "native M3T/O2 graph status seam missing")
require("#ifdef ET_G3R_O2_GRAPH_PREFLIGHT_PRIVATE" in o2_header,
        "O2 feature guard missing")
require("#ifdef ET_G3R_O2_GRAPH_PREFLIGHT_PRIVATE" in m3t_header,
        "M3T feature guard missing")
require("(load \"tr3_lease_root.esk\")" in fixture,
        "fixture lacks genuine TR3 root")
require("(load \"g3r_candidate_o2_lease_preflight_extension.esk\")" in fixture,
        "fixture lacks candidate extension")
require("(define-public" not in scheme and "(export" not in scheme,
        "private source exposes a package API")
print("G3-R O2/lease source contract PASS")
