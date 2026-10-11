#!/usr/bin/env python3
"""Source-only regression for the composite IR checker's SSA predicates."""

import ast
from pathlib import Path
import re

checker = Path(__file__).with_name("check-p1-registry-publication.sh").read_text()
anchor = "python3 - \"${tmp}/aot/registry-proof.ll\" \"${tmp}/publication-witness.txt\" <<'PY'\n"
body = checker.split(anchor, 1)[1].split("\nPY\n", 1)[0]
parsed = ast.parse(body)
names = {"ssa_definition", "pair_fields", "checked_pair_root_value",
         "checked_witness_store"}
helpers = ast.Module(
    body=[node for node in parsed.body
          if isinstance(node, ast.FunctionDef) and node.name in names],
    type_ignores=[],
)
assert len(helpers.body) == len(names)
namespace = {"re": re}
exec(compile(helpers, "composite-ir-predicates", "exec"), namespace)

pair = [
    "  %alloc = call ptr @arena_allocate_vector_with_header(ptr %arena, i64 2)",
    "  %base = getelementptr i8, ptr %alloc, i64 8",
    "  %slot0 = getelementptr %eshkol_tagged_value, ptr %base, i64 0",
    "  %slot1 = getelementptr %eshkol_tagged_value, ptr %base, i64 1",
    "  %next-registry.load = load %eshkol_tagged_value, ptr %next-registry, align 8",
    "  %next-witnesses.load = load %eshkol_tagged_value, ptr %next-witnesses, align 8",
    "  store %eshkol_tagged_value %next-registry.load, ptr %slot0, align 8",
    "  store %eshkol_tagged_value %next-witnesses.load, ptr %slot1, align 8",
    "  %ptrint = ptrtoint ptr %alloc to i64",
    "  %tag = insertvalue %eshkol_tagged_value { i8 8, i64 undef }, i64 %ptrint, 4",
    "  %next-pair = alloca %eshkol_tagged_value, align 16",
    "  store %eshkol_tagged_value %tag, ptr %next-pair, align 8",
]

witness = [
    "  %witness.load = load %eshkol_tagged_value, ptr %witness, align 8",
    "  %bits = extractvalue %eshkol_tagged_value %witness.load, 4",
    "  %object = inttoptr i64 %bits to ptr",
    "  %data = getelementptr i8, ptr %object, i64 8",
    "  %field = getelementptr %eshkol_tagged_value, ptr %data, i64 3",
    "  %shell.load = load %eshkol_tagged_value, ptr %shell, align 8",
    "  store %eshkol_tagged_value %shell.load, ptr %wb_val_slot, align 8",
    "  %status = call i32 @eshkol_region_write_barrier_checked_v1(ptr %wb_out_slot, ptr %object, ptr %wb_val_slot)",
    "wb_success:",
    "  %wb_result = load %eshkol_tagged_value, ptr %wb_out_slot, align 8",
    "  store %eshkol_tagged_value %wb_result, ptr %field, align 8",
    "  br label %vset_merge",
    "wb_failure:",
    "  unreachable",
    "vset_merge:",
]

def rejects(thunk):
    try:
        thunk()
    except SystemExit as error:
        assert str(error).startswith("IR proof failed:")
    else:
        raise AssertionError("IR checker accepted mutated SSA")

assert namespace["pair_fields"](pair, "synthetic") == [
    "%next-registry.load", "%next-witnesses.load"]
rejects(lambda: namespace["pair_fields"](
    [line.replace("i64 2)", "i64 3)") for line in pair], "synthetic"))
rejects(lambda: namespace["pair_fields"](
    [line.replace("%next-witnesses.load, ptr %slot1",
                  "%next-registry.load, ptr %slot1") for line in pair], "synthetic"))
pair_publication = pair + [
    "  %vref_result = load %eshkol_tagged_value, ptr %stage_slot0, align 8",
    "  store %eshkol_tagged_value %vref_result, ptr %next-pair, align 8",
    "  %next-pair.load = load %eshkol_tagged_value, ptr %next-pair, align 8",
    "  store %eshkol_tagged_value %next-pair.load, ptr %root_value_slot, align 8",
    "  %root_status = call i32 @eshkol_region_write_barrier_checked_v1(ptr %out, ptr %root, ptr %root_value_slot)",
]
root_barrier = len(pair_publication) - 1
namespace["checked_pair_root_value"](pair_publication, root_barrier, "synthetic")
overwritten_pair = pair_publication.copy()
overwritten_pair[-3:-3] = [
    "  %wrong = load %eshkol_tagged_value, ptr %other, align 8",
    "  store %eshkol_tagged_value %wrong, ptr %next-pair, align 8",
]
rejects(lambda: namespace["checked_pair_root_value"](
    overwritten_pair, len(overwritten_pair) - 1, "overwritten-pair"))
wrong_type_pair = pair_publication.copy()
wrong_type_pair[-3:-3] = [
    "  store i64 0, ptr %next-pair, align 8",
]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_type_pair, len(wrong_type_pair) - 1, "wrong-type-pair"))
postload_pair = pair_publication.copy()
postload_pair[-2:-2] = [
    "  store %eshkol_tagged_value %tag, ptr %next-pair, align 8",
]
rejects(lambda: namespace["checked_pair_root_value"](
    postload_pair, len(postload_pair) - 1, "postload-pair"))
overwritten_root_slot = pair_publication.copy()
overwritten_root_slot[-1:-1] = [
    "  store i64 0, ptr %root_value_slot, align 8",
]
rejects(lambda: namespace["checked_pair_root_value"](
    overwritten_root_slot, len(overwritten_root_slot) - 1,
    "overwritten-root-slot"))
barrier = witness.index(next(line for line in witness if "call i32 @eshkol_region" in line))
flow = {"success_index": barrier + 1, "successor_index": len(witness) - 1}
namespace["checked_witness_store"](witness, barrier, flow, "synthetic")
rejects(lambda: namespace["checked_witness_store"](
    [line.replace("%shell.load, ptr %wb_val_slot",
                  "%wrong.load, ptr %wb_val_slot") for line in witness],
    barrier, flow, "synthetic"))
overwritten_witness = witness.copy()
overwritten_witness[barrier:barrier] = [
    "  %wrong.load = load %eshkol_tagged_value, ptr %wrong, align 8",
    "  store %eshkol_tagged_value %wrong.load, ptr %wb_val_slot, align 8",
]
rejects(lambda: namespace["checked_witness_store"](
    overwritten_witness, barrier + 2,
    {"success_index": barrier + 3, "successor_index": len(overwritten_witness) - 1},
    "overwritten-value"))
wrong_type_witness = witness.copy()
wrong_type_witness[barrier:barrier] = [
    "  store i64 0, ptr %wb_val_slot, align 8",
]
rejects(lambda: namespace["checked_witness_store"](
    wrong_type_witness, barrier + 1,
    {"success_index": barrier + 2, "successor_index": len(wrong_type_witness) - 1},
    "wrong-type-value"))
rejects(lambda: namespace["checked_witness_store"](
    [line.replace("ptr %data, i64 3", "ptr %data, i64 2")
     for line in witness], barrier, flow, "synthetic"))
rejects(lambda: namespace["checked_witness_store"](
    [line.replace("%witness.load, 4", "%record.load, 4")
     for line in witness], barrier, flow, "synthetic"))
print("P1 composite IR SSA source regression: PASS")
