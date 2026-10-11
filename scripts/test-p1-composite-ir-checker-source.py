#!/usr/bin/env python3
"""Source-only regression for the composite IR checker's SSA predicates."""

import ast
from pathlib import Path
import re

checker = Path(__file__).with_name("check-p1-registry-publication.sh").read_text()
anchor = "python3 - \"${tmp}/aot/registry-proof.ll\" \"${tmp}/publication-witness.txt\" <<'PY'\n"
body = checker.split(anchor, 1)[1].split("\nPY\n", 1)[0]
parsed = ast.parse(body)
names = {"ssa_definition", "local_writes", "vector_ref_local",
         "kind_slot_routes", "staged_pair_readback", "pair_fields",
         "checked_pair_root_value", "checked_witness_store"}
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
    "  %witness = alloca %eshkol_tagged_value, align 16",
    "  store %eshkol_tagged_value %initial_witness, ptr %witness, align 8",
    "  %next-pair = alloca %eshkol_tagged_value, align 16",
    "  store %eshkol_tagged_value %vref_result, ptr %next-pair, align 8",
    "  %next-pair.load = load %eshkol_tagged_value, ptr %next-pair, align 8",
    "  %pair_bits = extractvalue %eshkol_tagged_value %next-pair.load, 4",
    "  %pair_ptr = inttoptr i64 %pair_bits to ptr",
    "  %pair_data = getelementptr i8, ptr %pair_ptr, i64 8",
    "  %pair_slot1 = getelementptr %eshkol_tagged_value, ptr %pair_data, i64 1",
    "  %spine_value = load %eshkol_tagged_value, ptr %pair_slot1, align 8",
    "vref_bounds_ok:",
    "  %vref_result2 = phi %eshkol_tagged_value [ %tensor_result, %vref_tensor_merge ], [ %spine_value, %vref_bounds_ok ]",
    "  %spine_bits = extractvalue %eshkol_tagged_value %vref_result2, 4",
    "  %spine_ptr = inttoptr i64 %spine_bits to ptr",
    "  %car_direct = load %eshkol_tagged_value, ptr %spine_ptr, align 8",
    "  %car_result = phi %eshkol_tagged_value [ %car_direct, %car_subtype_ok ], [ %unused, %car_other ]",
    "  store %eshkol_tagged_value %car_result, ptr %witness, align 8",
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
    "  %active-record-stage.load = load %eshkol_tagged_value, ptr %active-record-stage_cap, align 8",
    "  %active-slot = alloca %eshkol_tagged_value, align 16",
    "  br label %cond_check_state",
    "cond_check_state:",
    "  %kind.load0 = load %eshkol_tagged_value, ptr %kind_cap, align 8",
    "  %kind.bits0 = extractvalue %eshkol_tagged_value %kind.load0, 4",
    "  %kind.state = icmp eq i64 %kind.bits0, 0",
    "  br i1 %kind.state, label %cond_then, label %cond_next",
    "cond_next:",
    "  %kind.load1 = load %eshkol_tagged_value, ptr %kind_cap, align 8",
    "  %kind.bits1 = extractvalue %eshkol_tagged_value %kind.load1, 4",
    "  %kind.module = icmp eq i64 %kind.bits1, 1",
    "  br i1 %kind.module, label %cond_then1, label %cond_next2",
    "cond_next2:",
    "  %kind.load2 = load %eshkol_tagged_value, ptr %kind_cap, align 8",
    "  %kind.bits2 = extractvalue %eshkol_tagged_value %kind.load2, 4",
    "  %kind.tensor = icmp eq i64 %kind.bits2, 4",
    "  br i1 %kind.tensor, label %cond_then2, label %cond_next3",
    "cond_then:",
    "  br label %cond_done",
    "cond_then1:",
    "  br label %cond_done",
    "cond_then2:",
    "  br label %cond_done",
    "cond_next3:",
    "  br label %cond_done",
    "cond_done:",
    "  %cond_result = phi %eshkol_tagged_value [ { i8 1, i8 16, i16 0, i32 0, i64 0 }, %cond_then ], [ { i8 1, i8 16, i16 0, i32 0, i64 1 }, %cond_then1 ], [ { i8 1, i8 16, i16 0, i32 0, i64 4 }, %cond_then2 ], [ { i8 3, i8 0, i16 0, i32 0, i64 0 }, %cond_next3 ]",
    "  store %eshkol_tagged_value %cond_result, ptr %active-slot, align 8",
    "  %active-slot.load = load %eshkol_tagged_value, ptr %active-slot, align 8",
    "  %next-pair.load0 = load %eshkol_tagged_value, ptr %next-pair, align 8",
    "  %stage.fn.bits = extractvalue %eshkol_tagged_value %active-record-stage.load, 4",
    "  %stage.fn = inttoptr i64 %stage.fn.bits to ptr",
    "  %stage.call = call %eshkol_tagged_value %stage.fn(%eshkol_tagged_value %active-slot.load, %eshkol_tagged_value %next-pair.load0)",
    "  %call_result = phi %eshkol_tagged_value [ %active-record-stage.load, %return_as_is ], [ %stage.call, %do_direct_call ]",
    "  store %eshkol_tagged_value %call_result, ptr %stage, align 8",
    "  %stage.load = load %eshkol_tagged_value, ptr %stage, align 8",
    "  %stage.bits = extractvalue %eshkol_tagged_value %stage.load, 4",
    "  %stage.ptr = inttoptr i64 %stage.bits to ptr",
    "  %stage.base = getelementptr i8, ptr %stage.ptr, i64 8",
    "  %stage.slot0 = getelementptr %eshkol_tagged_value, ptr %stage.base, i64 0",
    "  %stage.value = load %eshkol_tagged_value, ptr %stage.slot0, align 8",
    "  %vref_result = phi %eshkol_tagged_value [ %tensor_result, %vref_tensor_merge ], [ %stage.value, %vref_bounds_ok ]",
    "  store %eshkol_tagged_value %vref_result, ptr %next-pair, align 8",
    "  %next-pair.load1 = load %eshkol_tagged_value, ptr %next-pair, align 8",
    "  store %eshkol_tagged_value %next-pair.load1, ptr %root_value_slot, align 8",
    "  %root_status = call i32 @eshkol_region_write_barrier_checked_v1(ptr %out, ptr %root, ptr %root_value_slot)",
]
root_barrier = len(pair_publication) - 1
namespace["checked_pair_root_value"](pair_publication, root_barrier, "synthetic")
state_entry_publication = [line.replace(
    "%eshkol_tagged_value %active-slot.load, %eshkol_tagged_value %next-pair.load0)",
    "%eshkol_tagged_value { i8 1, i8 16, i16 0, i32 0, i64 5 }, %eshkol_tagged_value %next-pair.load0)")
    for line in pair_publication]
namespace["checked_pair_root_value"](
    state_entry_publication, len(state_entry_publication) - 1,
    "state-entry", "state-entry-shell")
rejects(lambda: namespace["checked_pair_root_value"](
    pair_publication, root_barrier, "generic-as-state-entry",
    "state-entry-shell"))
wrong_state_slot = [line.replace("i64 5 }, %eshkol_tagged_value %next-pair.load0)",
                                 "i64 4 }, %eshkol_tagged_value %next-pair.load0)")
                    for line in state_entry_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_state_slot, len(wrong_state_slot) - 1,
    "wrong-state-slot", "state-entry-shell"))
wrong_stage = [line.replace("ptr %stage.slot0, align 8",
                            "ptr %unrelated_slot0, align 8")
               for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_stage, len(wrong_stage) - 1, "unrelated-stage"))
wrong_stage_slot = [line.replace("ptr %stage.base, i64 0",
                                 "ptr %stage.base, i64 1")
                    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_stage_slot, len(wrong_stage_slot) - 1, "wrong-stage-slot"))
wrong_stage_owner = [line.replace("ptr %active-record-stage_cap,",
                                  "ptr %other-stage_cap,")
                     for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_stage_owner, len(wrong_stage_owner) - 1, "wrong-stage-owner"))
wrong_stage_pointer = [line.replace(
    "extractvalue %eshkol_tagged_value %active-record-stage.load, 4",
    "extractvalue %eshkol_tagged_value %other-stage.load, 4")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_stage_pointer, len(wrong_stage_pointer) - 1,
    "wrong-stage-pointer"))
wrong_stage_call = [line.replace("%eshkol_tagged_value %next-pair.load0)",
                                 "%eshkol_tagged_value %other.arg)")
                    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_stage_call, len(wrong_stage_call) - 1, "unrelated-stage-call"))
wrong_position = [line.replace(
    "%eshkol_tagged_value %active-slot.load, %eshkol_tagged_value %next-pair.load0)",
    "%eshkol_tagged_value %next-pair.load0, %eshkol_tagged_value %wrong_pair)")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_position, len(wrong_position) - 1, "pair-in-slot-position"))
pair_both = [line.replace(
    "%eshkol_tagged_value %active-slot.load, %eshkol_tagged_value %next-pair.load0)",
    "%eshkol_tagged_value %next-pair.load0, %eshkol_tagged_value %next-pair.load0)")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    pair_both, len(pair_both) - 1, "pair-in-both-positions"))
wrong_first_slot = [line.replace(
    "%eshkol_tagged_value %active-slot.load, %eshkol_tagged_value %next-pair.load0)",
    "%eshkol_tagged_value %wrong-slot.load, %eshkol_tagged_value %next-pair.load0)")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_first_slot, len(wrong_first_slot) - 1, "unrelated-first-slot"))
wrong_slot_load = [line.replace(
    "%active-slot.load = load %eshkol_tagged_value, ptr %active-slot,",
    "%active-slot.load = load %eshkol_tagged_value, ptr %other-slot,")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_slot_load, len(wrong_slot_load) - 1, "wrong-slot-load"))
wrong_slot_selection = [line.replace(
    "store %eshkol_tagged_value %cond_result, ptr %active-slot,",
    "store %eshkol_tagged_value %wrong-slot, ptr %active-slot,")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_slot_selection, len(wrong_slot_selection) - 1,
    "wrong-slot-selection"))
wrong_slot_case = [line.replace(
    "i64 4 }, %cond_then2", "i64 9 }, %cond_then2")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_slot_case, len(wrong_slot_case) - 1, "wrong-slot-case"))
swapped_case_predecessors = [line.replace(
    "i64 0 }, %cond_then ], [ { i8 1, i8 16, i16 0, i32 0, i64 1 }, %cond_then1",
    "i64 0 }, %cond_then1 ], [ { i8 1, i8 16, i16 0, i32 0, i64 1 }, %cond_then")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    swapped_case_predecessors, len(swapped_case_predecessors) - 1,
    "swapped-kind-predecessors"))
wrong_branch_input = [line.replace(
    "extractvalue %eshkol_tagged_value %kind.load0, 4",
    "extractvalue %eshkol_tagged_value %unrelated, 4")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    wrong_branch_input, len(wrong_branch_input) - 1,
    "branch-bypasses-kind"))
alias_slot = pair_publication.copy()
alias_slot[alias_slot.index(
    "  %active-slot.load = load %eshkol_tagged_value, ptr %active-slot, align 8"):
    alias_slot.index(
    "  %active-slot.load = load %eshkol_tagged_value, ptr %active-slot, align 8")] = [
    "  %active-slot.alias = getelementptr i8, ptr %active-slot, i64 0",
    "  store %eshkol_tagged_value { i8 1, i8 16, i16 0, i32 0, i64 9 }, ptr %active-slot.alias, align 8",
]
rejects(lambda: namespace["checked_pair_root_value"](
    alias_slot, len(alias_slot) - 1, "alias-slot-overwrite"))
overwritten_slot = pair_publication.copy()
overwritten_slot.insert(overwritten_slot.index(
    "  %active-slot.load = load %eshkol_tagged_value, ptr %active-slot, align 8"),
    "  store %eshkol_tagged_value %wrong-slot, ptr %active-slot, align 8")
rejects(lambda: namespace["checked_pair_root_value"](
    overwritten_slot, len(overwritten_slot) - 1, "overwritten-slot"))
extra_argument = [line.replace(
    "%eshkol_tagged_value %next-pair.load0)",
    "%eshkol_tagged_value %next-pair.load0, %eshkol_tagged_value %wrong_pair)")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    extra_argument, len(extra_argument) - 1, "extra-stage-argument"))
missing_argument = [line.replace(
    "%eshkol_tagged_value %active-slot.load, %eshkol_tagged_value %next-pair.load0)",
    "%eshkol_tagged_value %next-pair.load0)")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    missing_argument, len(missing_argument) - 1, "missing-slot-argument"))
capture_argument = [line.replace(
    "%eshkol_tagged_value %next-pair.load0)",
    "%eshkol_tagged_value %next-pair.load0, ptr %cap_ptr_0)")
    for line in pair_publication]
namespace["checked_pair_root_value"](
    capture_argument, len(capture_argument) - 1, "stage-with-capture")
capture_before_pair = [line.replace(
    "%eshkol_tagged_value %next-pair.load0)",
    "ptr %cap_ptr_0, %eshkol_tagged_value %next-pair.load0)")
    for line in pair_publication]
rejects(lambda: namespace["checked_pair_root_value"](
    capture_before_pair, len(capture_before_pair) - 1,
    "capture-before-pair"))
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
wrong_spine_slot = [line.replace("ptr %pair_data, i64 1",
                                 "ptr %pair_data, i64 0")
                    for line in witness]
rejects(lambda: namespace["checked_witness_store"](
    wrong_spine_slot, barrier, flow, "wrong-witness-spine-slot"))
wrong_car_source = [line.replace("extractvalue %eshkol_tagged_value %vref_result2, 4",
                                 "extractvalue %eshkol_tagged_value %unrelated, 4")
                    for line in witness]
rejects(lambda: namespace["checked_witness_store"](
    wrong_car_source, barrier, flow, "wrong-witness-car-source"))
wrong_witness = witness.copy()
load_at = next(i for i, line in enumerate(wrong_witness)
               if "%witness.load = load" in line)
wrong_witness[load_at:load_at] = [
    "  %wrong_witness = load %eshkol_tagged_value, ptr %other, align 8",
    "  store %eshkol_tagged_value %wrong_witness, ptr %witness, align 8",
]
rejects(lambda: namespace["checked_witness_store"](
    wrong_witness, barrier + 2,
    {"success_index": barrier + 3, "successor_index": len(wrong_witness) - 1},
    "overwritten-witness-local"))
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
