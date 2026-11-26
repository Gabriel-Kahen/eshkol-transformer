#!/usr/bin/env python3
"""Source-contract checks for the standalone constructor-owned prerequisite."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
source = (root / "native/g3c4_p2g2_terminal_carriers_extension.esk").read_text()
fixture = (root / "tests/g3c4/p2_g2_terminal_carriers_test.esk").read_text()
additions = [
    "native/g3c4_p2g2_terminal_carriers_extension.esk",
    "tests/g3c4/p2_g2_terminal_carriers_test.esk",
    "tests/g3c4/call_entry_allocation_shim.cpp",
    "scripts/check-g3c4-terminal-carrier-failures.py",
    "scripts/check-g3c4-p2g2-terminal-carriers.py",
    "scripts/test-g3c4-p2g2-terminal-carriers.sh",
    "docs/g3/G3_C4_P2_G2_TERMINAL_CARRIERS_LEAF.md",
    "native/g3c4_p2g2_terminal_carriers_source_closure.txt",
]
base = (root / "native/g3c4_p2g2_terminal_pending_anchor_source_closure.txt").read_text().splitlines()
manifests = (
    "native/m3_package_source_closure.txt",
    "native/g3c4_i2_construction_source_closure.txt",
    "native/m3_package_native_source_closure.txt",
)
eshkol_inputs = (
    "tests/g3c4/p2_g2_terminal_carriers_test.esk",
    "tests/g3c4/p2_g2_terminal_pending_anchor_test.esk",
    "tests/g3c4/p2_g2_terminal_pending_anchor_feature_off_test.esk",
)
c_inputs = (
    "tests/g3c4/p2_g2_protected_scope_native.c",
    "tests/m3t/kernel_fail_allocator.c",
    "native/i2_wave2_package_bridge.c",
    "tests/g3c4/call_entry_allocation_shim.cpp",
    "native/p1_identity.c", "native/t1_i64_shell.c",
    "src/eshkol_transformer/m3_i64_integration.c",
    "src/eshkol_transformer/m3_model.c",
) + tuple(f"native/{name}.c" for name in (
    "data_io", "checkpoint_io", "n3k_primitives_provider",
    "g3n_primitives_provider", "n2_primitives_provider",
    "g3c4_primitives_provider", "g3s_sampling_provider",
    "a2_attention_provider", "a2_kv_cache",
))

def discover(seeds, pattern, include_dirs, require_all):
    seen = set()
    def visit(path):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        for name in re.findall(pattern, path.read_text()):
            candidates = [path.parent / name] + [root / d / name for d in include_dirs]
            found = next((p for p in candidates if p.is_file()), None)
            if found is not None:
                visit(found)
            elif require_all:
                raise AssertionError((path, name))
    for seed in seeds:
        visit(root / seed)
    return {str(p.relative_to(root)) for p in seen}

loads = discover(eshkol_inputs, r'(?m)^\s*\(load\s+"([^"]+)"',
                 ("internal/p1/lib", "internal/c1/lib", "internal/t1/lib",
                  "src", "lib", "native"), True)
native = discover(c_inputs, r'(?m)^\s*#\s*include\s*[<"]([^>"\n]+)[>"]',
                  ("include", "native", "src", "src/eshkol_transformer"), False)
expected = set(base + additions + list(manifests) + ["toolchain/eshkol.lock"])
for manifest in manifests:
    # Runtime-owned .deps inputs are sealed separately with compiler/image IDs.
    expected.update(p for p in (root / manifest).read_text().splitlines()
                    if not p.startswith(".deps/"))
expected.update(loads | native)
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + sorted(expected - set(base))
assert len(set(closure)) == len(closure)
assert all((root / name).is_file() for name in closure)
names = ("first-raw", "first-staging", "pending-raw", "pending-staging",
         "second-raw", "second-staging", "text-one", "text-two")
sizes = (1, 8, 2, 16, 1, 8, 1, 2)
required = [
    f"({name} (make-bytevector {size} 0))" for name, size in zip(names, sizes)
] + [f"(eq? (vector-ref bundle {i}) {name})" for i, name in enumerate(names)] + [
    "(let ((roots (vector '())))",
    "(eq? bundle (car (car remaining)))",
    "(extents (vector 9 16 10 24 9 16 9 10))",
    "(eq? (vector-ref bundle 8) extents)",
    "(m3t-exact-i64? (vector-ref extents i))",
    "(vector-set! roots 0 remaining)",
]
def contract(candidate):
    return (all(item in candidate for item in required)
            and candidate.count("(make-bytevector ") == 8
            and "(extern " not in candidate
            and "native-terminal" not in candidate)
assert contract(source)
for item in required:
    assert not contract(source.replace(item, "#t", 1)), item
assert source.index("(remaining\n") < source.index("(proof #t operation)")
assert source.index("(proof #t operation)") < source.index("(vector-set! roots 0 remaining)")
for item in ("paired same-length originals swapped", "duplicate alias rejected",
             "length fault injected", "equal-length genuine foreign carrier rejected",
             "retirement scrubs original bytes", "other constructor unaffected"):
    assert item in fixture
print(f"G3-C4 P2/G2 terminal carriers static PASS: mutants={len(required)} loads={len(loads)} native={len(native)}")
