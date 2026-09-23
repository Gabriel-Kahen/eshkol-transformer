# Pinned runtime publication blocker

This is diagnostic evidence, not an E3-P1 acceptance gate. The script deliberately
returns **1** when promotion publishes a graph containing regional pointers.

Reproduce from the repository root with:

```sh
/usr/bin/bash scripts/test-e3-p1-runtime-barrier.sh
```

The default source/build paths use the repository's existing `common.sh` policy.
Overrides are `ESHKOL_SOURCE_DIR`, `ESHKOL_BUILD_DIR`, and
`E3_P1_BARRIER_EVIDENCE`. The script verifies the complete existing toolchain
provenance, including source HEAD `90cbd7130f47b8184bcc77b8d5c1b0026da980de`,
compiles only diagnostic fixtures, and records exact commands and artifact hashes.
The local run explicitly selected the canonical read-only source/build under
`/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/` with
`ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=llvm-config`.
The observed archive SHA256
is `580946c291d88e2008de02202b8e07942e7b85c0ac9c6b7f861397c1d115e240`.
This local CachyOS/LLVM22 observation is not supported Ubuntu22/LLVM21 evidence.

## Injection and observed result

Both fixtures allocate the P1 publication graph shape: a cons cell points to a
nine-slot record, which points to an inert one-slot token and three vectors of
length17. The model/frame fields contain inert values; the fixture establishes no
M3 model authority or E3 frame admission.

The target root arena is made temporarily bounded, with its current block's used
size set to `size - budget`. Budgets `0,48,208,240,528,816` allow successive graph
copy prefixes and force a deterministic `arena_allocate_aligned` NULL result.
This is synthetic allocator-failure injection, **not naturally occurring OOM**.
It does not intercept the write barrier or replace it with an exception.

`runtime_barrier_failure.cpp` calls the actual runtime barrier with a root-vector
destination and stores the result, matching the generated vector-set! sequence.
`runtime_barrier_publication.esk` compiles the actual Eshkol `vector-set!` path;
its small native control fixture only changes/restores arena capacity policy.
Within compiled `with-region`, the actual promotion target is the region's
`escape_base`; `get_global_arena_shared()` is temporarily redirected to the region.

All runs set `ESHKOL_ARENA_POISON=1`. Each unmodified successful promotion has zero
regional graph identities and its canonical graph remains valid after poisoned
region exit. Each injected case returns normally and publishes a graph containing
respectively `6,5,4,3,2,1` original regional identities. Thus neither the whole graph
nor any partially copied prefix is failure-atomic. Each failing process returns1;
the diagnostic script returns1 after recording both sets of six failures.

Failed graphs are inspected only while the source region is alive, then the root
holder is cleared before region exit. No freed memory is read. Only a graph proved
to have zero original regional object identities is accessed after successful
poisoned exit. This does not measure general allocator exhaustion or every C++
allocation failure inside the evacuation machinery.

## Source trace at the exact pin

- `lib/core/runtime_regions.cpp:1252`: `evac_raw` returns its original pointer if
  allocation returns NULL.
- `lib/core/runtime_regions.cpp:1282`: `evac_object` logs allocation failure and
  returns its original pointer.
- `inc/eshkol/logger.h:298`: `eshkol_error` calls `eshkol_printf` with ERROR level.
  `lib/core/logger.cpp:404..429` logs it; only FATAL exits. It does not raise an
  Eshkol exception, so slot64's cleanup guard cannot intercept this failure.
- `lib/core/runtime_regions.cpp:1899`: the barrier assigns the evacuation result
  to its output. It has no success/failure result to prevent publication.
- `lib/backend/collection_codegen.cpp:2199..2205`: vector-set! calls the barrier
  and stores its result. Emitted diagnostic IR reproduces this at
  `publication.o.ll:3220..3222` in the observed run.
- `lib/core/runtime_regions.cpp:1262,1289,1412..1431`: forwarding is installed
  before child traversal and persists for the region. **Source inference, not
  an executed retry test:** after a failed child copy, retrying a previously copied
  parent can return the recorded partial graph without traversing its failed
  descendants again. Any upstream correction must account for forwarding state,
  as well as prevent the final holder store.

The required bind failure-atomicity contract cannot currently be accepted. A guard
that handles an artificially injected exception before a barrier would not prove
this real NULL-allocation path safe. Root must resolve the upstream runtime/pin
prerequisite; this workstream adds no alternate P1 authority or runtime fallback.
