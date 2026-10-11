#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command ar
require_command python3
require_command rg
require_command timeout
verify_toolchain

runner="$(eshkol_build_dir)/eshkol-run"
provenance="$(eshkol_build_dir)/eshkol-transformer-provenance.tsv"
cc="$(tsv_value "${provenance}" cc_path)"
cxx="$(tsv_value "${provenance}" cxx_path)"
compiler_timeout="${P1_REGISTRY_PROOF_COMPILER_TIMEOUT_SECONDS:-360}"
[[ "${compiler_timeout}" =~ ^[1-9][0-9]*$ ]] || \
  die "P1_REGISTRY_PROOF_COMPILER_TIMEOUT_SECONDS must be a positive integer"

template="${PROJECT_ROOT}/templates/p1/module_roots.esk.tmpl"
generated="${PROJECT_ROOT}/internal/p1/lib/transformer/module.esk"
fixture="${PROJECT_ROOT}/tests/p1/registry_publication_region_test.esk"

"${PROJECT_ROOT}/scripts/generate-p1-roots.sh" --check

require_exact_count() {
  local expected=$1 pattern=$2 source=$3 actual
  actual="$(rg -F -c -- "${pattern}" "${source}" || true)"
  [[ "${actual}" == "${expected}" ]] || \
    die "expected ${expected} occurrences of '${pattern}' in ${source}; found ${actual}"
}

for source in "${template}" "${generated}"; do
  require_exact_count 1 "(define shell-registry-root (vector (vector '() '())))" "${source}"
  require_exact_count 1 "(define active-record-root (vector '() '() #t #f '() '()))" "${source}"
  require_exact_count 2 "(vector-ref shell-registry-root 0)" "${source}"
  require_exact_count 2 \
    "(vector-set! shell-registry-root 0 next-pair)" "${source}"
  require_exact_count 1 "(define (issuance-witness-current)" "${source}"
  require_exact_count 1 "(vector-set! witness 3 shell)" "${source}"
  require_exact_count 2 \
    "(vector-ref (car (shell-registry-current)) 0)" "${source}"
  require_exact_count 1 \
    "(canonical-record (shell-entry shell))" "${source}"
  require_exact_count 1 \
    "(vector-ref canonical-record 2)" "${source}"
  require_exact_count 1 \
    "(define tensor-provider-registry-root (vector '()))" "${source}"
  require_exact_count 1 \
    "(vector-ref tensor-provider-registry-root 0)" "${source}"
  require_exact_count 1 \
    "(vector-set! tensor-provider-registry-root 0" "${source}"
  require_exact_count 4 \
    "(tensor-provider-registry-current)" "${source}"
  require_exact_count 2 \
    "(raw-for-shell 'module-register-parameter-internal!" "${source}"
  require_exact_count 1 \
    "(vector-set! plan 1 canonical-handle)" "${source}"
  require_exact_count 1 \
    "(vector-set! (car (vector-ref plan 2)) 2 canonical-handle)" "${source}"
  if rg -F -e "(define shell-registry '())" \
      -e "(set! shell-registry next-registry)" \
      -e "(define tensor-providers '())" \
      -e "(set! tensor-providers next-providers)" "${source}" >/dev/null; then
    die "legacy captured registry mutation remains in ${source}"
  fi
done

python3 - "${generated}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
publication = source.split("(define (shell-for-raw kind raw)", 1)[1].split(
    "(define (state-entry-shell-for-raw", 1)[0]
ordered = (
    "(shell-for-raw 'state (vector-ref raw 1))",
    "(construction (active-construction))",
    "(witness (if scoped?",
    "(next-registry (cons entry (shell-registry-current)))",
    "(next-pair (vector next-registry next-witnesses))",
    "(active-record-stage active-slot next-pair)",
    "(native-create-shell kind raw construction)",
    "(set! next-pair (vector-ref stage 0))",
    "(set! next-registry (vector-ref next-pair 0))",
    "(if witness (set! witness (car (vector-ref next-pair 1))))",
    "(vector-set! entry 0 shell)",
    "(vector-set! witness 3 shell)",
    "(vector-set! active-record-root 2 #f)",
    "(vector-set! shell-registry-root 0 next-pair)",
    "(vector-set! active-record-root\n",
    "(vector-set! active-record-root 2 (vector-ref stage 2))",
)
positions = [publication.index(witness) for witness in ordered]
if positions != sorted(positions):
    raise SystemExit("active-record staging/publication order changed")
if "((or (eq? kind 'module) (eq? kind 'handle)) 1)" not in publication:
    raise SystemExit("scoped handle lost active-root pre-native stage")
if publication.index("(if (and scoped? shell)") > \
        publication.index("(set! stage (active-record-stage") or \
        publication.index("(construction-abort! construction)") > \
        publication.index("(set! stage (active-record-stage"):
    raise SystemExit("prepublication native success lost scoped abort guard")
entry = source.split("(define (state-entry-shell-for-raw", 1)[1].split(
    "(define (raw-for-shell", 1)[0]
entry_ordered = (
    "(active-record-source 5)",
    "(next-pair (vector next-registry (issuance-witness-current)))",
    "(active-record-stage 5 next-pair)",
    "(p1-native-state-entry-create native-context state-shell)",
    "(set! next-pair (vector-ref stage 0))",
    "(set! next-registry (vector-ref next-pair 0))",
    "(vector-set! record 0 shell)",
    "(vector-set! active-record-root 2 #f)",
    "(vector-set! shell-registry-root 0 next-pair)",
    "(vector-set! active-record-root 5 (vector-ref stage 1))",
    "(vector-set! active-record-root 2 (vector-ref stage 2))",
)
positions = [entry.index(witness) for witness in entry_ordered]
if positions != sorted(positions):
    raise SystemExit("state-entry staging/publication order changed")
if "(eq? (vector-ref (car records) 2) raw)" not in entry or \
        "(eq? (vector-ref (car records) 3) state)" not in entry:
    raise SystemExit("state-entry exact raw/owner identity check changed")
if "(active-record-source 4)" not in source or \
        "(shell-registry-current)" not in source.split(
            "(define (active-record-source slot)", 1)[1].split(
                "(define (active-record-stage", 1)[0]:
    raise SystemExit("active raw index lost its registry fallback")
compaction = source.split("(define (compact-released-state-shells! state)", 1)[1].split(
    "(define (bounded-acyclic-list-spine?", 1)[0]
if compaction.index("(active-record-unlink-record! 4 record)") > \
        compaction.index("(vector-set! record 2 dead-state-tensor-raw)") or \
        compaction.index("(active-record-unlink-record! 5 record)") > \
        compaction.index("(vector-set! record 2 dead-state-entry-raw)"):
    raise SystemExit("active raw records survive state release")
PY

tmp="$(mktemp -d "${TMPDIR:-/tmp}/p1-registry-publication.XXXXXX")"
cleanup() {
  if [[ "${P1_REGISTRY_PROOF_KEEP_TMP:-0}" == 1 ]]; then
    printf 'P1 registry publication evidence preserved: %s\n' "${tmp}" >&2
  else
    rm -rf -- "${tmp}"
  fi
}
trap cleanup EXIT

mkdir -p "${tmp}/native" "${tmp}/aot" "${tmp}/overlay/transformer" \
  "${tmp}/fallback"
"${cc}" -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion \
  -Wsign-conversion -Wshadow -fPIC -fvisibility=hidden -fno-common \
  -DET_P1_TRUSTED_BUILD=1 -I "${PROJECT_ROOT}/native" \
  -c "${PROJECT_ROOT}/native/p1_identity.c" \
  -o "${tmp}/native/p1_identity.o"
ar rcsD "${tmp}/native/libeshkol_transformer_p1_identity.a" \
  "${tmp}/native/p1_identity.o"

(cd "${tmp}/aot" &&
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${tmp}/cache" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${compiler_timeout}s" "${runner}" \
      --strict-types --optimize 0 --dump-ir --no-stdlib \
      -I "${PROJECT_ROOT}/internal/p1/lib" -I "${PROJECT_ROOT}/lib" \
      -I "${PROJECT_ROOT}/native" \
      -I "${PROJECT_ROOT}/tests/p1/providers" -L "${tmp}/native" \
      --lib eshkol_transformer_p1_identity "${fixture}" \
      -o "${tmp}/aot/registry-proof" \
      >"${tmp}/aot/compile.stdout" 2>"${tmp}/aot/compile.stderr")

test -x "${tmp}/aot/registry-proof"
test -s "${tmp}/aot/registry-proof.ll"
test ! -s "${tmp}/aot/compile.stderr"

python3 - "${tmp}/aot/registry-proof.ll" "${tmp}/publication-witness.txt" <<'PY'
import collections
import hashlib
import re
import sys

ir_path, witness_path = sys.argv[1:]
history = collections.deque(maxlen=48)
root_context = None
current = None
publication_functions = []

with open(ir_path, "r", encoding="utf-8") as stream:
    for number, text in enumerate(stream, 1):
        line = text.rstrip("\n")
        if ("store %eshkol_tagged_value" in line
                and "ptr %shell-registry-root_letrecstar" in line
                and "zeroinitializer" not in line
                and root_context is None):
            root_context = list(history) + [(number, line)]
        if line.startswith("define "):
            current = [(number, line)] if "shell-registry-root_cap" in line else None
        elif current is not None:
            current.append((number, line))
            if line == "}":
                body = "\n".join(item[1] for item in current)
                if "%next-registry" in body:
                    publication_functions.append(current)
                current = None
        history.append((number, line))

if root_context is None:
    raise SystemExit("IR proof failed: root registry vector initialization not found")
root_text = "\n".join(line for _, line in root_context)
root_store = root_context[-1][1]
root_value_match = re.search(
    r"store %eshkol_tagged_value (%[-A-Za-z0-9_.]+), "
    r"ptr %shell-registry-root_letrecstar", root_store)
if root_value_match is None:
    raise SystemExit("IR proof failed: registry root store has no SSA value")
root_value = root_value_match.group(1)
pointer_value_match = re.search(
    rf"{re.escape(root_value)} = insertvalue %eshkol_tagged_value .* "
    r"i64 (%[-A-Za-z0-9_.]+), 4", root_text)
if pointer_value_match is None:
    raise SystemExit("IR proof failed: registry root store is not a heap vector")
pointer_value = pointer_value_match.group(1)
vector_pointer_match = re.search(
    rf"{re.escape(pointer_value)} = ptrtoint ptr (%[-A-Za-z0-9_.]+) to i64",
    root_text)
if vector_pointer_match is None:
    raise SystemExit("IR proof failed: registry root vector pointer is absent")
vector_pointer = vector_pointer_match.group(1)
arena_match = re.search(
    rf"{re.escape(vector_pointer)} = call ptr "
    r"@arena_allocate_vector_with_header\(ptr (%[-A-Za-z0-9_.]+), i64 1\)",
    root_text)
if arena_match is None:
    raise SystemExit("IR proof failed: registry holder is not the one-slot vector")
if not re.search(
        rf"{re.escape(arena_match.group(1))} = load ptr, "
        r"ptr @__global_arena", root_text):
    raise SystemExit("IR proof failed: registry holder is not allocated from global arena")
if len(publication_functions) != 2:
    raise SystemExit(
        f"IR proof failed: expected two registry publication functions; "
        f"found {len(publication_functions)}")

witness = []
root_store_number = root_context[-1][0]
witness.append(f"root-vector-global-arena-store-line={root_store_number}")
roles = set()

def block(lines, label, name):
    index = next((i for i, line in enumerate(lines)
                  if line.startswith(label + ":")), None)
    if index is None:
        raise SystemExit(f"IR proof failed: {name} lacks block {label}")
    end = next((i for i in range(index + 1, len(lines))
                if re.match(r"^[-A-Za-z0-9_.]+:", lines[i])), len(lines))
    return index, lines[index:end]

def checked_barrier_flow(lines, index, name):
    call = re.search(
        r"(?P<status>%[-A-Za-z0-9_.]+) = call i32 "
        r"@eshkol_region_write_barrier_checked_v1\("
        r"ptr (?P<out>%[-A-Za-z0-9_.]+), ptr [^,]+, ptr [^)]+\)",
        lines[index])
    if call is None:
        raise SystemExit(f"IR proof failed: {name} has malformed checked barrier")
    status = call.group("status")
    branch = next((re.search(
        r"br i1 (%[-A-Za-z0-9_.]+), label %([-A-Za-z0-9_.]+), "
        r"label %([-A-Za-z0-9_.]+)", line)
        for line in lines[index + 1:index + 5]
        if line.lstrip().startswith("br i1 ")), None)
    if branch is None or not any(
            re.search(rf"{re.escape(branch.group(1))} = icmp eq i32 "
                      rf"{re.escape(status)}, 0", line)
            for line in lines[index + 1:index + 4]):
        raise SystemExit(f"IR proof failed: {name} does not branch on checked status")
    success_label, failure_label = branch.group(2), branch.group(3)
    success_index, success = block(lines, success_label, name)
    failure_index, failure = block(lines, failure_label, name)
    result = None
    for line in success:
        result = re.search(
            rf"(%[-A-Za-z0-9_.]+) = load %eshkol_tagged_value, "
            rf"ptr {re.escape(call.group('out'))}", line)
        if result is not None:
            break
    if result is None or not any(
            re.search(rf"store %eshkol_tagged_value "
                      rf"{re.escape(result.group(1))}, ptr ", line)
            for line in success):
        raise SystemExit(f"IR proof failed: {name} does not commit checked output")
    successor = next((re.search(r"br label %([-A-Za-z0-9_.]+)", line)
                      for line in success
                      if line.lstrip().startswith("br label %")), None)
    if successor is None:
        raise SystemExit(f"IR proof failed: {name} checked success has no successor")
    failure_text = "\n".join(failure)
    if (f"call void @eshkol_runtime_emergency_raise_v1(i32 {status})" not in failure_text
            or "unreachable" not in failure_text
            or any(line.lstrip().startswith("br ") for line in failure)):
        raise SystemExit(f"IR proof failed: {name} checked failure can continue")
    successor_index, successor_lines = block(
        lines, successor.group(1), name)
    return {
        "success_label": success_label,
        "success_index": success_index,
        "failure_label": failure_label,
        "failure_index": failure_index,
        "successor_label": successor.group(1),
        "successor_index": successor_index,
        "successor": successor_lines,
    }

def ssa_definition(lines, value, name):
    matches = [(i, line) for i, line in enumerate(lines)
               if line.lstrip().startswith(value + " = ")]
    if len(matches) != 1:
        raise SystemExit(f"IR proof failed: {name} has no unique {value} definition")
    return matches[0]

def pair_fields(lines, name):
    # The pre-native pair is a fresh length-two vector. Follow its actual
    # tagged pointer to both element stores; SSA names alone are insufficient.
    initial = [re.search(
        r"store %eshkol_tagged_value (%[-A-Za-z0-9_.]+), ptr %next-pair,", line)
        for line in lines]
    initial = [match.group(1) for match in initial if match]
    if not initial:
        raise SystemExit(f"IR proof failed: {name} has no initial pair store")
    tag = initial[0]
    _, tag_line = ssa_definition(lines, tag, name)
    pointer = re.search(r"insertvalue %eshkol_tagged_value .* i64 "
                        r"(%[-A-Za-z0-9_.]+), 4", tag_line)
    if pointer is None:
        raise SystemExit(f"IR proof failed: {name} pair is not tagged vector")
    _, pointer_line = ssa_definition(lines, pointer.group(1), name)
    allocation = re.search(r"ptrtoint ptr (%[-A-Za-z0-9_.]+) to i64",
                           pointer_line)
    if allocation is None:
        raise SystemExit(f"IR proof failed: {name} pair pointer is not allocated")
    allocated = allocation.group(1)
    _, allocation_line = ssa_definition(lines, allocated, name)
    if not re.search(r"call ptr @arena_allocate_vector_with_header\("
                     r"ptr %[-A-Za-z0-9_.]+, i64 2\)", allocation_line):
        raise SystemExit(f"IR proof failed: {name} pair is not length two")
    bases = [re.match(r"\s*(%[-A-Za-z0-9_.]+) = getelementptr i8, "
                      rf"ptr {re.escape(allocated)}, i64 8$", line)
             for line in lines]
    bases = [match.group(1) for match in bases if match]
    if len(bases) != 1:
        raise SystemExit(f"IR proof failed: {name} pair has no unique data base")
    values = []
    for slot in (0, 1):
        slot_pointers = [re.match(
            r"\s*(%[-A-Za-z0-9_.]+) = getelementptr %eshkol_tagged_value, "
            rf"ptr {re.escape(bases[0])}, i64 {slot}$", line)
            for line in lines]
        slot_pointers = [match.group(1) for match in slot_pointers if match]
        if len(slot_pointers) != 1:
            raise SystemExit(f"IR proof failed: {name} pair slot {slot} is ambiguous")
        stores = [re.search(
            r"store %eshkol_tagged_value (%[-A-Za-z0-9_.]+), "
            rf"ptr {re.escape(slot_pointers[0])},", line)
            for line in lines]
        stores = [match.group(1) for match in stores if match]
        if len(stores) != 1:
            raise SystemExit(f"IR proof failed: {name} pair slot {slot} store is ambiguous")
        values.append(stores[0])
    if values[0] == values[1]:
        raise SystemExit(f"IR proof failed: {name} pair spines alias in SSA")
    return values

def checked_witness_store(lines, barrier, flow, name):
    call = re.search(r"@eshkol_region_write_barrier_checked_v1\("
                     r"ptr %[-A-Za-z0-9_.]+, ptr (%[-A-Za-z0-9_.]+), "
                     r"ptr (%[-A-Za-z0-9_.]+)\)", lines[barrier])
    if call is None:
        raise SystemExit(f"IR proof failed: {name} witness barrier arguments absent")
    target, value_slot = call.groups()
    before = lines[:barrier]
    values = [re.search(
        r"store %eshkol_tagged_value (%shell\.load[0-9]*), "
        rf"ptr {re.escape(value_slot)},", line)
        for line in before[-8:]]
    if not any(values):
        raise SystemExit(f"IR proof failed: {name} witness barrier value is not shell")
    shell_value = next(match.group(1) for match in values if match)
    _, shell_load = ssa_definition(lines, shell_value, name)
    if not re.search(r"load %eshkol_tagged_value, ptr %shell,", shell_load):
        raise SystemExit(f"IR proof failed: {name} witness value is not native shell")
    target_sources = [re.search(
        r"(%[-A-Za-z0-9_.]+) = inttoptr i64 "
        r"(%[-A-Za-z0-9_.]+) to ptr", line)
        for line in before]
    target_sources = [match for match in target_sources
                      if match and match.group(1) == target]
    if len(target_sources) != 1:
        raise SystemExit(f"IR proof failed: {name} witness target pointer absent")
    _, extracted = ssa_definition(lines, target_sources[0].group(2), name)
    tagged = re.search(r"extractvalue %eshkol_tagged_value "
                       r"(%witness\.load[0-9]*), 4", extracted)
    if tagged is None:
        raise SystemExit(f"IR proof failed: {name} barrier target is not witness")
    _, witness_load = ssa_definition(lines, tagged.group(1), name)
    if not re.search(r"load %eshkol_tagged_value, ptr %witness,", witness_load):
        raise SystemExit(f"IR proof failed: {name} target is not canonical witness")
    # A checked success must commit the returned value to witness field 3.
    success_block = lines[flow["success_index"]:flow["successor_index"]]
    committed = [re.search(r"store %eshkol_tagged_value %wb_result[0-9]*, "
                           r"ptr (%[-A-Za-z0-9_.]+),", line)
                 for line in success_block]
    committed = [match.group(1) for match in committed if match]
    if len(committed) != 1:
        raise SystemExit(f"IR proof failed: {name} witness checked commit absent")
    _, slot_line = ssa_definition(lines, committed[0], name)
    slot = re.search(r"getelementptr %eshkol_tagged_value, ptr "
                     r"(%[-A-Za-z0-9_.]+), i64 3$", slot_line)
    if slot is None:
        raise SystemExit(f"IR proof failed: {name} witness commit is not field 3")
    _, base_line = ssa_definition(lines, slot.group(1), name)
    if not re.search(rf"getelementptr i8, ptr {re.escape(target)}, i64 8$",
                     base_line):
        raise SystemExit(f"IR proof failed: {name} witness field has wrong base")

for ordinal, function in enumerate(publication_functions, 1):
    lines = [line for _, line in function]
    numbers = [number for number, _ in function]
    signature = lines[0]
    name_match = re.search(r"@([^\s(]+)", signature)
    name = name_match.group(1) if name_match else f"publication-{ordinal}"
    if "ptr %kind_cap" in signature and "ptr %state_cap" not in signature:
        role = "generic-shell"
    elif "ptr %state_cap" in signature and "ptr %state-shell_cap" in signature:
        role = "state-entry-shell"
    else:
        raise SystemExit(f"IR proof failed: unrecognized publication role in {name}")
    roles.add(role)
    barriers = [i for i, line in enumerate(lines)
                if "call i32 @eshkol_region_write_barrier_checked_v1" in line]
    # The generic publication has one additional checked canonical witness
    # shell store. State-entry publication carries the existing spine through
    # the pair and retains its seven checked stores.
    expected_barriers = 8 if role == "generic-shell" else 7
    if len(barriers) != expected_barriers:
        raise SystemExit(
            f"IR proof failed: {name} has {len(barriers)} write barriers, "
            f"expected {expected_barriers}")
    barrier_flows = {
        index: checked_barrier_flow(lines, index, name) for index in barriers
    }
    registry_spine, witness_spine = pair_fields(lines, name)
    _, registry_def = ssa_definition(lines, registry_spine, name)
    if not re.search(r"load %eshkol_tagged_value, ptr %next-registry,",
                     registry_def):
        raise SystemExit(f"IR proof failed: {name} pair slot 0 is not registry")
    if role == "generic-shell":
        _, witness_def = ssa_definition(lines, witness_spine, name)
        if not re.search(r"load %eshkol_tagged_value, ptr %next-witnesses,",
                         witness_def):
            raise SystemExit(f"IR proof failed: {name} pair slot 1 is not witness spine")
    else:
        # The carried witness is the exact return SSA of the captured
        # issuance-witness-current closure, including the as-is call branch.
        _, carry_phi = ssa_definition(lines, witness_spine, name)
        captured = [re.match(
            r"\s*(%[-A-Za-z0-9_.]+) = load %eshkol_tagged_value, "
            r"ptr %issuance-witness-current_cap,", line)
            for line in lines]
        captured = [match.group(1) for match in captured if match]
        if (len(captured) != 1 or
                not re.search(r"= phi %eshkol_tagged_value", carry_phi) or
                not re.search(rf"\[ {re.escape(captured[0])}, %return_as_is[0-9]* \]",
                              carry_phi)):
            raise SystemExit(f"IR proof failed: {name} lost witness carry SSA identity")
        direct = re.search(r"\[ (%[-A-Za-z0-9_.]+), %do_direct_call[0-9]* \]",
                           carry_phi)
        if direct is None:
            raise SystemExit(f"IR proof failed: {name} witness call path absent")
        _, direct_call = ssa_definition(lines, direct.group(1), name)
        called = re.search(r"call %eshkol_tagged_value "
                           r"(%[-A-Za-z0-9_.]+)\(\)$", direct_call)
        if called is None:
            raise SystemExit(f"IR proof failed: {name} witness carry is not call result")
        _, called_pointer = ssa_definition(lines, called.group(1), name)
        bits = re.search(r"inttoptr i64 (%[-A-Za-z0-9_.]+) to ptr$",
                         called_pointer)
        if bits is None:
            raise SystemExit(f"IR proof failed: {name} witness call pointer absent")
        _, bits_line = ssa_definition(lines, bits.group(1), name)
        if not re.search(
                rf"extractvalue %eshkol_tagged_value "
                rf"{re.escape(captured[0])}, 4$", bits_line):
            raise SystemExit(f"IR proof failed: {name} witness call source changed")

    root_loads = set()
    for line in lines:
        match = re.search(
            r"(%[-A-Za-z0-9_.]+) = load %eshkol_tagged_value, "
            r"ptr %shell-registry-root_cap", line)
        if match:
            root_loads.add(match.group(1))
    root_extracts = set()
    for line in lines:
        for root_load in root_loads:
            match = re.search(
                rf"(%[-A-Za-z0-9_.]+) = extractvalue %eshkol_tagged_value "
                rf"{re.escape(root_load)}, 4", line)
            if match:
                root_extracts.add(match.group(1))
    root_pointers = set()
    for line in lines:
        for extracted in root_extracts:
            match = re.search(
                rf"(%[-A-Za-z0-9_.]+) = inttoptr i64 "
                rf"{re.escape(extracted)} to ptr", line)
            if match:
                root_pointers.add(match.group(1))

    root_barrier = None
    for index in barriers:
        prior = "\n".join(lines[max(0, index - 3):index])
        slot_match = re.search(
            r"store %eshkol_tagged_value %next-pair\.load[0-9]*, "
            r"ptr (%[-A-Za-z0-9_.]+)", prior)
        if (slot_match and slot_match.group(1) in lines[index]
                and any(pointer in lines[index] for pointer in root_pointers)):
            root_barrier = index
            break
    if root_barrier is None:
        raise SystemExit(
            f"IR proof failed: {name} lacks a root-vector next-pair barrier")
    pair_load = re.search(r"store %eshkol_tagged_value "
                          r"(%next-pair\.load[0-9]*), ptr %[-A-Za-z0-9_.]+",
                          "\n".join(lines[max(0, root_barrier - 3):root_barrier]))
    if pair_load is None:
        raise SystemExit(f"IR proof failed: {name} root value is not pair SSA")
    _, pair_load_def = ssa_definition(lines, pair_load.group(1), name)
    if not re.search(r"load %eshkol_tagged_value, ptr %next-pair,",
                     pair_load_def):
        raise SystemExit(f"IR proof failed: {name} root value bypasses pair local")
    canonical_pair_stores = [i for i, line in enumerate(lines[:root_barrier])
                             if re.search(r"store %eshkol_tagged_value "
                                          r"%vref_result[0-9]*, ptr %next-pair,",
                                          line)]
    if len(canonical_pair_stores) != 1:
        raise SystemExit(f"IR proof failed: {name} lacks staged canonical pair")
    entry_barrier = barriers[1]
    if entry_barrier >= root_barrier:
        raise SystemExit(f"IR proof failed: {name} root publication ordering changed")
    entry_prior = "\n".join(lines[max(0, entry_barrier - 3):entry_barrier])
    if "%shell.load" not in entry_prior:
        raise SystemExit(f"IR proof failed: {name} does not publish shell into record first")

    root_flow = barrier_flows[root_barrier]
    successor_label = root_flow["successor_label"]
    successor_index = root_flow["successor_index"]
    successor = root_flow["successor"]
    canonical_tail = lines[root_barrier + 1:]
    if not any("ptr %shell-registry-current_cap" in line
               and " load " in f" {line} " for line in canonical_tail):
        raise SystemExit(
            f"IR proof failed: {name} root barrier does not enter canonical readback")
    # The staged active node and validity marker are root-owned before
    # publication. Their three checked stores follow the registry store.
    # Each has the normal checked success/failure control flow, and the
    # caller still returns a registry-read shell after that tail.
    root_ordinal = 4 if role == "generic-shell" else 3
    if barriers[root_ordinal] != root_barrier:
        raise SystemExit(f"IR proof failed: {name} active tail ordering changed")
    if role == "generic-shell":
        if not entry_barrier < barriers[2] < root_barrier:
            raise SystemExit(f"IR proof failed: {name} witness store is not before root")
        checked_witness_store(lines, barriers[2], barrier_flows[barriers[2]], name)
        witness_local_stores = [i for i, line in enumerate(lines[:barriers[2]])
                                if re.search(r"store %eshkol_tagged_value "
                                             r"%[-A-Za-z0-9_.]+, ptr %witness,",
                                             line)]
        if len(witness_local_stores) < 2:
            raise SystemExit(f"IR proof failed: {name} witness target was not staged")
    for index in barriers[root_ordinal + 1:]:
        if not any("ptr %active-record-root_cap" in line
                   for line in lines[max(0, index - 256):index]):
            raise SystemExit(f"IR proof failed: {name} active tail lost its root")
    if any("%shell.load" in line for line in lines[root_barrier + 1:]):
        raise SystemExit(f"IR proof failed: {name} reuses pre-barrier shell identity")
    returns = [(i, line) for i, line in enumerate(lines)
               if line.lstrip().startswith("ret %eshkol_tagged_value")]
    if len(returns) != 1:
        raise SystemExit(f"IR proof failed: {name} does not return canonical branch result")
    return_match = re.search(
        r"ret %eshkol_tagged_value (%cond_result[0-9]*)$", returns[0][1])
    if return_match is None:
        raise SystemExit(f"IR proof failed: {name} does not return canonical branch result")
    cond_phis = [line for line in lines if re.search(
        rf"{re.escape(return_match.group(1))} = phi %eshkol_tagged_value", line)]
    if len(cond_phis) != 1 or "%vref_result" not in cond_phis[0]:
        raise SystemExit(f"IR proof failed: {name} return phi bypasses registry readback")

    returned_reads = re.findall(r"%vref_result[0-9]*", cond_phis[0])
    readback_index = next((i for i, line in enumerate(lines)
                           if any(f"{value} = phi %eshkol_tagged_value" in line
                                  for value in returned_reads)), None)
    if readback_index is None:
        raise SystemExit(f"IR proof failed: {name} lacks returned canonical readback")
    return_index = returns[0][0]
    witness.extend((
        f"publication-{ordinal}-function={name}",
        f"publication-{ordinal}-role={role}",
        f"publication-{ordinal}-entry-write-barrier-line={numbers[entry_barrier]}",
        f"publication-{ordinal}-root-write-barrier-line={numbers[root_barrier]}",
        f"publication-{ordinal}-root-barrier-success-block={root_flow['success_label']}",
        f"publication-{ordinal}-root-barrier-failure-block={root_flow['failure_label']}",
        f"publication-{ordinal}-root-barrier-successor={successor_label}",
        f"publication-{ordinal}-root-barrier-successor-line={numbers[successor_index]}",
        f"publication-{ordinal}-canonical-readback-line={numbers[readback_index]}",
        f"publication-{ordinal}-canonical-return-line={numbers[return_index]}",
        f"publication-{ordinal}-post-root-stale-shell-use=false",
    ))

if roles != {"generic-shell", "state-entry-shell"}:
    raise SystemExit(f"IR proof failed: publication roles changed: {sorted(roles)}")

digest = hashlib.sha256()
with open(ir_path, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
witness.append(f"ir-sha256={digest.hexdigest()}")
with open(witness_path, "w", encoding="utf-8") as stream:
    stream.write("\n".join(witness) + "\n")
PY

ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=2s \
  30s "${tmp}/aot/registry-proof" \
  >"${tmp}/runtime.stdout" 2>"${tmp}/runtime.stderr"
test ! -s "${tmp}/runtime.stderr"
grep -Fx "P1 REGISTRY PUBLICATION PASS: 24 checks" \
  "${tmp}/runtime.stdout" >/dev/null

# The diagnostic copy differs from the trusted root only at four asserted
# insertion sites. Its extra slot is private to this test; the production
# source, generated template, and ABI are checked above and never altered.
python3 - "${generated}" "${tmp}/overlay/transformer/module.esk" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
def inject(old, new):
    global source
    if source.count(old) != 1:
        raise SystemExit(f"diagnostic overlay lost exact anchor: {old!r}")
    source = source.replace(old, new)

inject("         module-construction-parameters-internal\n"
       "         p1-trusted-surface)\n",
       "         module-construction-parameters-internal\n"
       "         p1-index-test-control!\n"
       "         p1-trusted-surface)\n")
inject("(define active-record-root (vector '() '() #t #f '() '()))\n",
       "(define active-record-root (vector '() '() #t #f '() '()))\n"
       "(define p1-index-test-fail-stage? #f)\n")
inject("    (vector-set! active-record-root 3 stage)\n"
       "    (vector-ref active-record-root 3)))",
       "    (vector-set! active-record-root 3 stage)\n"
       "    (if p1-index-test-fail-stage?\n"
       "        (begin (set! p1-index-test-fail-stage? #f)\n"
       "               (raise 'index-stage-failed)) #t)\n"
       "    (vector-ref active-record-root 3)))")
inject("  ;; Append-only source-private G3-C4 construction split, slots 71..72.\n"
       "  (lambda (identity)\n"
       "    (construction-prepare-eval-guarded! identity))\n"
       "  (lambda (identity)\n"
       "    (construction-seal-prepared! identity))\n"
       "  ;; Append-only P1-only sealed current-root verifier, slot 73.\n"
       "  (lambda (root-shell expected-trie expected-handles)\n"
       "    (p1-fixed-sealed-topology!\n"
       "      root-shell expected-trie expected-handles))\n"
       "  )))",
       "  ;; Append-only source-private G3-C4 construction split, slots 71..72.\n"
       "  (lambda (identity)\n"
       "    (construction-prepare-eval-guarded! identity))\n"
       "  (lambda (identity)\n"
       "    (construction-seal-prepared! identity))\n"
       "  ;; Append-only P1-only sealed current-root verifier, slot 73.\n"
       "  (lambda (root-shell expected-trie expected-handles)\n"
       "    (p1-fixed-sealed-topology!\n"
       "      root-shell expected-trie expected-handles))\n"
       "  ;; Test-only slot 74 models failed advisory linkage.\n"
       "  (lambda (mode)\n"
       "    (cond ((eq? mode 'fail-stage)\n"
       "           (set! p1-index-test-fail-stage? #t) #t)\n"
       "          ((eq? mode 'pending-clear)\n"
       "           (not (vector-ref active-record-root 3)))\n"
       "          ((eq? mode 'invalidate)\n"
       "           (vector-set! active-record-root 2 #f)\n"
       "           (vector-set! active-record-root 4 '())\n"
       "           (vector-set! active-record-root 5 '()) #t)\n"
       "          ((eq? mode 'invalid)\n"
       "           (not (vector-ref active-record-root 2)))\n"
       "          (else #f)))\n"
       "  )))")
source += "\n(define (p1-index-test-control! mode)\n"
source += "  ((vector-ref p1-trusted-surface 74) mode))\n"
Path(sys.argv[2]).write_text(source)
PY

(cd "${tmp}/fallback" &&
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="${tmp}/cache-fallback" ESHKOL_LIB_DIR="${PROJECT_ROOT}/lib" \
    ESHKOL_CXX_COMPILER="${cxx}" \
    timeout --foreground --signal=TERM --kill-after=5s \
      "${compiler_timeout}s" "${runner}" \
      --strict-types --optimize 0 --no-stdlib \
      -I "${tmp}/overlay" -I "${PROJECT_ROOT}/internal/p1/lib" \
      -I "${PROJECT_ROOT}/lib" -I "${PROJECT_ROOT}/native" \
      -I "${PROJECT_ROOT}/tests/p1/providers" -L "${tmp}/native" \
      --lib eshkol_transformer_p1_identity \
      "${PROJECT_ROOT}/tests/p1/registry_index_fallback_test.esk" \
      -o "${tmp}/fallback/index-fallback" \
      >"${tmp}/fallback/compile.stdout" \
      2>"${tmp}/fallback/compile.stderr")
test -x "${tmp}/fallback/index-fallback"
test ! -s "${tmp}/fallback/compile.stderr"
ESHKOL_ARENA_POISON=1 timeout --foreground --signal=TERM --kill-after=2s \
  30s "${tmp}/fallback/index-fallback" \
  >"${tmp}/fallback/runtime.stdout" 2>"${tmp}/fallback/runtime.stderr"
test ! -s "${tmp}/fallback/runtime.stderr"
grep -Fx "P1 INDEX FALLBACK PASS: 11 checks" \
  "${tmp}/fallback/runtime.stdout" >/dev/null

cat "${tmp}/publication-witness.txt"
printf 'poison-nested-sibling-runtime=true\n'
printf 'diagnostic-invalid-index-fallback-runtime=true\n'
printf 'P1 REGISTRY PUBLICATION PROOF PASS\n'
