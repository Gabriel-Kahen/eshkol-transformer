# Fresh CPU4 checkpoint factory

This source-private factory composes the actual `42438ce3` checkpoint consumer,
`de2e6d83` typed child producer, `502f0671` control byte fix, `3580136f` trainer
detach and `fefbfa6d` restore publication prerequisite. It owns a newly loaded
Scheme C2 checkpoint and its staging/access, then one fresh canonical C4
candidate, optimizer, binding and trainer. Tokenizer and D2 remain caller-owned.
The native C2 byte-image is owned/freed inside the existing reader bridge; it is
not a factory resource. No general destroy callback, owner cast, substitute model,
public API, evaluation or numerical path is added.

Private signatures:

- `cpu4-checkpoint-factory-create-internal!(path,policy,report,resolved,tokenizer,dataset,recovery-cell)`
  returns the genuine restored trainer. The exact initially empty one-slot cell
  receives its registered recovery ticket before any file/native acquisition.
- `cpu4-checkpoint-factory-retire-internal!(ticket)` retires only this factory's
  still-unpublished children and is the explicit blocked-cleanup retry path.
  Completed cleanup is idempotent; foreign tickets reject.

The caller supplies the real C2 persistence policy, genuine K2 report, exact X1
resolved configuration and existing raw-byte T2/N1-T4 D2 components. The factory
uses split C2 staging, authentic capability descriptors/report admission, and the
real reconstruction procedure. It enrolls each returned stage/image/child token
before another fallible operation. Its own recovery envelope is globally rooted
and reacquired before acquisition; handlers exist before pending/busy publication.

The existing consumer stages all fourteen parameter values, twenty-eight moment
buffers, complete topology/optimizer metadata and serialized controls under the
exact C2 borrow. All source access closes. The factory checks the supplied X1
profile, canonical UTF8 bytes by value, fingerprint and seed against that source.
It releases the owned C2 image **before child creation**, then creates its own
candidate, optimizer from the authenticated source configuration, binding and
trainer. The caller receives the same canonical C4 model after real value/moment,
optimizer/scheduler/update and D2/control restoration; initialization from the
same seed is not used as a replacement for restoring checkpoint values.

The consumer's control staging/copy precedes busy component plans. Complete
preflights precede I2 writes; commit order remains I2, O2 config/update plus
finalization, D2, then controls last. Return identities already exist before this
tail. After it returns, only fixed factory state/return stores remain. An
unexpected postwrite native validation, clone drain or parent WRITE-leave/control
close defect is a fail-stop, not a recoverable rollback. No owned image close
exists after this write frontier.

On prewrite failure, the ticket retains the first error and exact custody. Cleanup
closes subordinate source access/borrow and drains all staging/plans, releases the
owned loaded C2/staging, then asks the typed child producer to retire. C2 release
can allocate/reserve and reject before consumption: keep the rooted owner until
success or its exact registered `dead` state. A post-consumption defect may raise
after marking it dead; retry must not release it twice. Unrelated live O2 states
can still block child retirement; the ticket and authentic children remain for
explicit nonregional retry. Caller T2/D2 and caller checkpoint objects are never
released. Successful restore intentionally restores the caller D2 cursor;
prewrite failure preserves it through the genuine D2 abort plan.

CPU-v1 admits fixed **serial nonregional** entries with immutable trusted I2
provider/codec and reader seams. The actual runtime `region_current` pointer query
is the first operation in factory create, factory retire/retry, checkpoint snapshot
and checkpoint restore. Non-NULL rejects before recovery/file/owner/registry
changes. The pinned source proof shows NULL means TLS depth0; global/vector stores
then take the no-evacuation memmove path without promotion allocation. No explicit
or automatic nursery spans the fixed C1/I2/P1/C2 publication tails. This does not
admit arbitrary provider rebinding, callbacks, concurrent entry or general
regional C2 loading. The independent child layer remains unchanged and keeps its
own separate regional contract.

Host checks execute actual factory/guard Scheme with explicit native/reader/child
doubles: constructor cuts, partial O2/binding/trainer ownership, owned-image release
blocking, post-consumption release, external-state cleanup retry, original error
retention, exact caller identity/survival, config mismatch and postwrite fail-stop.
The existing consumer host suite retains its staging/control/order coverage.
These are not compiler, file-decoder, native or sanitizer admission.

`tests/cpu4_checkpoint_factory/nonregional_runtime.esk` is a prepared genuine-runtime
fragment for the root-owned aggregate. It observes actual NULL baseline, a real
outer pointer and distinct nested region pointer, requires typed unsupported
negatives for all four entries, and compares exact checkpoint registries, C2 owner/
staging, native O2/F32 counts, reader/IO event counts and native image/reader/FD
counts. It calls a deliberately nonexistent path so an IO
failure cannot pass as a region rejection. Its held-ticket helper must also be
called after the aggregate's genuine positive factory test. It is NOT RUN here.
Root owns the reader aggregate/native roster and supported locked execution.
The fragment uses the existing child native observer and C2 core, reader and
checkpoint IO test hooks (`ET_C2_CHECKPOINT_CORE_TESTING`,
`ET_C2_CHECKPOINT_READER_TESTING`, `ET_CHECKPOINT_IO_TESTING`). Reuse the genuine
K2 discovery and split reader supplied by the owning `native/c2_wave2_root.esk`
composition, without loading a second identity root. A positive FILE factory
test must SAVE a genuine CPU4 snapshot through the actual C2 writer first;
older C2 operational payloads cannot stand in for that C4 checkpoint producer.

Persisted C4 generation continuation is a separate required actual-contract gate.
A live typed multi-token/EOS chain does not establish serialized C4 sampler/history
continuation; old C2 profile1/history<=2 and training RNG cannot substitute for it.
Root owns the later mode sequence: evaluate with the same live binding, finish and
drain the frame restoring TRAIN, detach/release that binding, then put the SAME
model into eval before fresh generator creation/adoption. This factory implements
none of that evaluator/generation sequence.
