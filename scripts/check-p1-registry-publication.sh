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
  require_exact_count 1 "(define shell-registry-root (vector '()))" "${source}"
  require_exact_count 1 "(vector-ref shell-registry-root 0)" "${source}"
  require_exact_count 2 \
    "(vector-set! shell-registry-root 0 next-registry)" "${source}"
  require_exact_count 2 \
    "(vector-ref (car (shell-registry-current)) 0)" "${source}"
  require_exact_count 1 \
    "(canonical-record (shell-entry shell))" "${source}"
  require_exact_count 1 \
    "(vector-ref canonical-record 2)" "${source}"
  require_exact_count 1 \
    "(vector-ref (car tensor-providers) 2)" "${source}"
  require_exact_count 2 \
    "(raw-for-shell 'module-register-parameter-internal!" "${source}"
  require_exact_count 1 \
    "(vector-set! plan 1 canonical-handle)" "${source}"
  require_exact_count 1 \
    "(vector-set! (car (vector-ref plan 2)) 2 canonical-handle)" "${source}"
  if rg -F -e "(define shell-registry '())" \
      -e "(set! shell-registry next-registry)" "${source}" >/dev/null; then
    die "legacy captured registry mutation remains in ${source}"
  fi
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/p1-registry-publication.XXXXXX")"
cleanup() {
  if [[ "${P1_REGISTRY_PROOF_KEEP_TMP:-0}" == 1 ]]; then
    printf 'P1 registry publication evidence preserved: %s\n' "${tmp}" >&2
  else
    rm -rf -- "${tmp}"
  fi
}
trap cleanup EXIT

mkdir -p "${tmp}/native" "${tmp}/aot"
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
                if "call void @eshkol_region_write_barrier_into" in line]
    if len(barriers) != 2:
        raise SystemExit(
            f"IR proof failed: {name} has {len(barriers)} write barriers, expected 2")

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
            r"store %eshkol_tagged_value %next-registry\.load, "
            r"ptr (%[-A-Za-z0-9_.]+)", prior)
        if (slot_match and slot_match.group(1) in lines[index]
                and any(pointer in lines[index] for pointer in root_pointers)):
            root_barrier = index
            break
    if root_barrier is None:
        raise SystemExit(
            f"IR proof failed: {name} lacks a root-vector next-registry barrier")
    entry_barrier = barriers[0]
    if entry_barrier >= root_barrier:
        raise SystemExit(f"IR proof failed: {name} root publication ordering changed")
    entry_prior = "\n".join(lines[max(0, entry_barrier - 3):entry_barrier])
    if "%shell.load" not in entry_prior:
        raise SystemExit(f"IR proof failed: {name} does not publish shell into record first")

    branch_match = next((re.search(r"br label %([-A-Za-z0-9_.]+)", line)
                         for line in lines[root_barrier + 1:root_barrier + 7]
                         if re.search(r"br label %([-A-Za-z0-9_.]+)", line)), None)
    if branch_match is None:
        raise SystemExit(f"IR proof failed: {name} root barrier has no successor")
    successor_label = branch_match.group(1)
    successor_index = next((i for i, line in enumerate(lines)
                            if line.startswith(successor_label + ":")), None)
    if successor_index is None:
        raise SystemExit(f"IR proof failed: {name} root barrier successor is absent")
    successor_end = next((i for i in range(successor_index + 1, len(lines))
                          if re.match(r"^[-A-Za-z0-9_.]+:", lines[i])), len(lines))
    successor = lines[successor_index:successor_end]
    if not any("ptr %shell-registry-current_cap" in line
               and " load " in f" {line} " for line in successor):
        raise SystemExit(
            f"IR proof failed: {name} root barrier does not enter canonical readback")
    if not any("%vref_result = phi %eshkol_tagged_value" in line
               for line in lines):
        raise SystemExit(f"IR proof failed: {name} does not read the canonical shell slot")
    if any("%shell.load" in line for line in lines[root_barrier + 1:]):
        raise SystemExit(f"IR proof failed: {name} reuses pre-barrier shell identity")
    returns = [(i, line) for i, line in enumerate(lines)
               if line.lstrip().startswith("ret %eshkol_tagged_value")]
    if len(returns) != 1 or "%cond_result" not in returns[0][1]:
        raise SystemExit(f"IR proof failed: {name} does not return canonical branch result")
    cond_phis = [line for line in lines if "%cond_result = phi" in line]
    if len(cond_phis) != 1 or "%vref_result" not in cond_phis[0]:
        raise SystemExit(f"IR proof failed: {name} return phi bypasses registry readback")

    readback_index = next(
        i for i, line in enumerate(lines)
        if "%vref_result = phi %eshkol_tagged_value" in line)
    return_index = returns[0][0]
    witness.extend((
        f"publication-{ordinal}-function={name}",
        f"publication-{ordinal}-role={role}",
        f"publication-{ordinal}-entry-write-barrier-line={numbers[entry_barrier]}",
        f"publication-{ordinal}-root-write-barrier-line={numbers[root_barrier]}",
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
grep -Fx "P1 REGISTRY PUBLICATION PASS: 7 checks" \
  "${tmp}/runtime.stdout" >/dev/null

cat "${tmp}/publication-witness.txt"
printf 'poison-nested-sibling-runtime=true\n'
printf 'P1 REGISTRY PUBLICATION PROOF PASS\n'
