# TR3 same-aggregate constructor candidate

Status: supported non-installed linked-package candidate. It proves the accepted A0
five-receiver call in one identity universe; it does not publish a public
trainer API, change the production TR3-C initializer-only library, or use the
separate private C factory request.

`native/tr3_public_candidate_root.esk` loads the unchanged
`tr3_c_private_package_root.esk` and defines `trainer-create/5` as a direct
call to `tr3-lease-create-internal/5`. The latter owns all validation, captures
the D2 epoch-start cursor, and enrolls the sole 22-slot trainer record. Its
failure path leaves every receiver with the caller. The candidate bridge
exports boxed X1 parse/resolve, T2 byte tokenizer, D2 open/close, M3T
initializer/model, P1 module-parameters/tree paths/tree handle, O2 create,
and TR3 create from the same prelocalized object. The source rename map has
the seven E1B error entries and those twelve operation entries. C2 and M3T
package bridges cannot both be included: each includes the I2 helper source.
This candidate instead compiles the reviewed native helpers once and uses
small wrappers with their existing fixed-arity boxed ABI.

The candidate gate pins the 47-source closure, 67 native source/header paths,
30 native objects, 173 unresolved runtime symbols, 18 public C symbols, fixed
f31/LLVM21 compiler/runtime and one combined archive member. Its linked
caller constructs a real resolved X1 value, byte tokenizer, finite D2
dataset, fixed M3T model, P1 parameter tree and 14 unique O2 paths, then
calls `trainer-create/5`. A forged model must raise the authenticated E1
`invalid-argument/trainer-create` tuple; its D2 child remains caller-owned and
closes successfully. A second call with the enrolled tuple must raise
`invalid-state/trainer-create` for overlap. Compiler depfiles must show that
the caller loads only its source and the safe E1 error facade, never private
trusted source. The exact-head `3b0f38d` linked gate passed with an empty
caller stderr under `ESHKOL_ARENA_POISON=1`. Its source compile took 3:38.92,
peaked at 3,729,092 KiB RSS and used no swap. The immutable evidence directory
is `/home/gabe/.codex/evidence/eshkol-transformer/tr3-public-candidate-3b0f38d-20260925/`;
`SHA256SUMS` has SHA-256 `1b12fc91d50ac2c1ea07915ff373456f6635105c4496bf2aa4fe646e3bc20e25`
and passed `sha256sum -c` with exit 0. The manifest pins were copied only from
this linked object and its native depfiles; the manifest follow-up does not
change the tested root, bridge or caller bytes.

## Public lifetime prerequisite

A0 says the exclusive lease ends when the trainer is unreachable. Current
`tr3-trainers` slot 0 strongly retains the record and shell until the private
`tr3-lease-unenroll-internal!` explicitly unlinks them. The package candidate
therefore proves process-lifetime enrollment only. It cannot be installed as
the A0 `trainer-create` facade on that evidence.

The pinned Eshkol `weak-ref` codegen (`lib/backend/llvm_codegen.cpp`,
`codegenWeakRef`) sets a WEAK flag on the object's header and returns the
same tagged value. It does not itself provide a separately authenticated weak
registry edge or a demonstrated callback that unenrolls this lease at
unreachability. `memory_abi_v2.h` declares an optional layout finalizer for
external resources, but declaration alone does not establish a supported
trainer finalizer in this runtime. A weak/finalizer route needs an upstream
runtime API plus a focused live/dropped-shell GC and registry-unlink proof.

The narrower source-level route is to accept an explicit public trainer
release operation and revise A0's lease-lifetime rule accordingly. The
private idle-only unenroll already authenticates the exact shell, rejects
busy/repeat/forged calls, and leaves all five receivers caller-owned and live.
That is a contract decision, not an API this candidate adds. Until one route
is accepted and tested, no installed `trainer-create` export or facade change
is warranted.
