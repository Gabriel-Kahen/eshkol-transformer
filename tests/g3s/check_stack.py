"""Conservative normal-object provider-only stack ceiling, with compiler spills.

The acyclic internal call graph can have each emitted function live at most once.
Summing *all* emitted static frames overcounts mutually exclusive paths. Add 128
bytes red-zone and 8 bytes return-address allowance per emitted function. External
libc/libm and caller/K1 frames are excluded; runtime_stack separately measures a
painted stack including those external calls.
"""
import re
import sys
from pathlib import Path

usage, ir_path = map(Path, sys.argv[1:])
frames = {}
for line in usage.read_text().splitlines():
    location, size, kind = line.split("\t")
    assert kind == "static", f"unbounded/dynamic stack: {line}"
    name = location.rsplit(":", 1)[1]
    assert name not in frames
    frames[name] = int(size)
assert frames and "provider_validate" in frames and "provider_invoke" in frames, frames
ir = ir_path.read_text()
functions = dict(re.findall(r"^define\b[^\n]*?@([\w.]+)\([^\n]*\)[^\n]*\{\n(.*?)^}", ir, re.M | re.S))
assert functions
assert set(frames) <= set(functions), f"emitted frames absent from audited IR: {set(frames) - set(functions)}"
calls = {}
for name, body in functions.items():
    edges = set()
    for line in body.splitlines():
        if not re.search(r"\b(call|invoke)\b", line):
            continue
        if " asm " in line:
            continue  # reviewed x87/MXCSR observation instructions only
        match = re.search(r"@([\w.]+)\(", line)
        assert match, f"unreviewed indirect call: {line}"
        if match[1] in functions:
            edges.add(match[1])
    calls[name] = edges

def visit(name, active):
    assert name not in active, f"recursive provider graph: {active} -> {name}"
    for child in calls[name]:
        visit(child, active | {name})

for name in calls:
    visit(name, set())
bound = sum(frames.values()) + len(frames) * (128 + 8)
assert bound <= 16384, (frames, bound)
for name, size in sorted(frames.items()):
    print(f"G3S compiler frame {name}: {size} bytes (static, includes spills)")
print(f"G3S compiler simultaneous bound: {bound}/16384 bytes; conservative sum of {len(frames)} acyclic frames plus red-zone/return allowances")
print("G3S compiler exclusions: caller/K1 and external libc/libm frames")
