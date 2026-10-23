# Focused CPU4 binding handler-allocation witness (staged)

SOURCE ONLY. Actual compilation, handler-OOM injection, native lease lifecycle
and constructor failure/retry are **NOT RUN**. This six-file leaf changes no
production source, root, roster, manifest, runtime, public/native ABI, numerical
behavior, mode contract or supported launcher. Root owns the later recipe,
source composition/pins,8GiB/3CPU/one compile job/shared lock, toolchain custody,
whole-run timeout and independent result admission. Do not launch from this doc.

Staging base isfd0db7270ca0212f73fc64337379d1ada32d0154,
tree3a8294a7d550bea956e99a256282dab127f17e15. Its repaired binding blob is
`4cc9d1feb4530da9c12e3224b9978d72bf0f798adaa46931ffc6e878b8ece346`.
The immutable79ea814d994417d646835f5d09f26459869bf4a7 baseline binding blob is
`7ff90345a23204a6b182b7d12417350c07cdd9871b1a23b54dd49b62a38dd567`.
`generator.py` extracts ONLY that constructor, renames only its declaration to
`cpu4bw-baseline-create`, and verifies exact AST equality after reversing the
header rename. It emits one definition to a fresh output. Generated baseline
source belongs in gate evidence, not Git. This is baseline-constructor execution
inside the current actual union, **not** execution of the full historical79ea
union. Configuration/strict P1/native owners/runtime remain common controls.

Root's later single source identity universe must load its actual common CPU4
root, append the generated constructor, append `body.esk`, and call
`cpu4-binding-witness-run` once. The body itself contains definitions/fixture
externs only; it adds no root or driver. It constructs genuine canonical
candidate models with the actual owned-candidate API and seed1729 resolved fixed
C4 config, explicitly sets real P1 TRAIN, and obtains the actual native owner
through the existing registry validator. Each successful case retires its
candidate only after lease-release observations; no zero-memory claim follows.

`shim.cpp` adapts the reviewed actual M3 handler witness mechanics from
`tests/m3cg/handler_failure_shim.cpp` (SHA256
`11c4ee67711e976907eb96d3cd3690be9a579f3a33dfc1e7f119ce3e95c1ebe7`)
and `.esk` (`2f8ebed32876a12a664e3307137dce7d333144d5b91787a125f717b972ad98f8`).
It uses pinned fe9 exception structs/symbols. It calibrates the actual static
condition5 pointer via the runtime's own emergency raise and checks canonical
heap tag/flags/reserved bits, pointer identity/current exception, and actual
handler stack progression at every failed caught boundary. It never manufactures
an exception or substitutes the runtime push/get-raised/malloc implementation.

Root must link the actual shim with GNU wrapper resolution for `malloc`,
`eshkol_push_exception_handler`, `eshkol_get_raised_value`, and existing native
`et_g3c4_private_training_lease_{create,prepare_release,commit_release}_v1`.
Wrappers delegate to their real functions and observe owner/lease pointer
identity and returned success only. Shim's ordinary `lease_check` import is the
existing real API. Root must prove actual wrapper imports/linkage and exact
runtime/header/archive provenance in the eventual artifact, not infer it from
this source. These are fixture functions, not new production counters or ABI.

Before each cut, non-tail guarded recursion occupies recycled handlers until an
actual handler-size malloc is observed. Depth128 is a failure cutoff, not a
claimed free-list bound. Arm records the active fixture frame and requires an
observed fresh malloc. Installation cuts fail the first actual constructor
handler malloc, persistently failing handler-size allocations through the catch.
The shim requires exactly one attempted constructor push, a genuine malloc hit,
exactly one failed handler malloc, expected raw condition5 read count, expected
stack transition, and zero native lease-create attempts. Any unexpected earlier
helper guard or absent allocation must fail the witness instead of passing a
weaker interpretation.

Three fixed cases, no tuning:

1. Extracted baseline install cut: actual pending row remains; exact same-model
   baseline retry must truly reject `invalid-state/cpu4-model-binding-create`.
   No native create occurred. ONLY after that observation, explicit baseline-only
   test quarantine marks its leaked pending row dead so the fixture may retire
   the candidate. This mutation never occurs in a repaired result path.
2. Actual repaired constructor install cut: no pending authority remains; retry
   on the exact same model/native owner must create one genuine native lease.
   Actual release must prepare+commit that exact lease once; checking that same
   released lease through the actual native API must reject.
3. Actual repaired postpublication cut: explicitly replace ONLY the capture
   closure in trusted P1 slot74's pair slot0, saving its genuine closure. The
   production `cpu4-p1-fixed-capture-internal` already dynamically calls that
   actual trusted slot. Wrapper delegates genuine strict capture FIRST, checks
   success, then requests actual runtime handler reserve129 with handler malloc
   failure enabled. The real malloc hit/exact condition5/stack checks are
   mandatory:129 is only a bounded attempt, never proof of exhaustion. If reserve
   returns, the witness exits failure. Each fresh case first requires zero
   existing same-owner binding rows. Before mode3 retry, the actual global
   registry must retain exactly one same-owner row in DEAD state: deletion or
   a prior dead row cannot satisfy this check. Then exact same-owner retry/
   release must work as above. Restore the genuine
   slot closure BEFORE every retry, including all normal control cases.

Trusted-slot mutation is explicit fixture interposition, not a production P1
capability or provider change. Actual AOT invocation of the mutated slot and
restoration remains a compiled-gate obligation; source tests cannot establish
it. The postpublication cut observes current canonical cleanup; it does NOT
prove the vector's address relocated. The old constructor may also pass such a
postpublication cut when no relocation occurred. The baseline install cut is
the causal before/after control. Host replacement/promotion doubles remain
separate evidence. There is no raw row-address inspector or relocation claim.

Lease release detaches owner.training_lease and marks the lease DEAD, retaining
control/registry storage. Actual successful create/prepare/commit and same-lease
check rejection do NOT prove freed storage, zero retained bytes or leak absence.
No new lease counter, leak metric, parameter update or training claim is added.

`check.py` requires exactly three causal records plus the final fixed scope line:

```
CPU4-BINDING-WITNESS case=1 pushes=1 condition5-reads=1 failure-create-attempts=0 pending=1 retry-blocked=1 retry-create=0 prepare=0 commit=0 dead-check=0
CPU4-BINDING-WITNESS case=2 pushes=1 condition5-reads=1 failure-create-attempts=0 pending=0 retry-blocked=0 retry-create=1 prepare=1 commit=1 dead-check=1
CPU4-BINDING-WITNESS case=3 pushes=1 condition5-reads=2 failure-create-attempts=0 pending=0 retry-blocked=0 retry-create=1 prepare=1 commit=1 dead-check=1
CPU4-BINDING-WITNESS-PASS relocation=UNPROVEN retained-bytes=UNMEASURED
```

Checker input is raw UTF-8 bytes with exact LF records; missing/extra/reordered
records or altered fields/scope are failures. This is no independent native
oracle: the shim/caller observations and actual supported artifact are required.

Cheap checks: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest
 tests.cpu4_binding_witness.test_source` covers baseline generation/equality,
fixed source hashes, genuine constructor/model/P1 calls, delegation/restore and
baseline-only quarantine order, non-tail occupancy, exact condition/actual-hit
source contracts, real lease wrapper names, and each causal-field/framing
checker negative. Guile reader checks only syntax. No host mock run is called a
native witness. A host clang++ syntax-only check was performed against the fe9
header; it produced no object/binary. Its exact command/provider version/header
hash/source hash/status are retained in external CHECKS evidence; it is not a
supported compiler/runtime gate. No further compiler process is authorized here.

Remaining executable gate: root's independently reviewed focused supported fe9
normal/repeat witness, with generated-source/depfile/IR/wrapper/link/runtime/
stdout/stderr/exit/source-prepost/resource provenance and post-exit seals. Until
that gate executes and is authenticated, actual allocation failure, condition5
crossings, P1 dynamic interposition, native lifecycle/retry, and canonical cleanup
are UNPROVEN. Real address relocation and retained-byte measurement remain
unproven even after the stated witness succeeds. No broader learning/FILE/resume,
installed/final CPU-v1/sanitizer/generation acceptance follows from this fixture.
