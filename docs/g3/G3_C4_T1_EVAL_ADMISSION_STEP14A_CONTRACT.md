# G3-C4 Step 14A: T1 identity and eval admission

Step 14A closes the tokenizer-identity and model-mode dependency identified by
Step 13A. It changes only the source-private Eshkol generator and call authority.
It adds no public export, native boundary, raw tokenizer pointer, prompt carrier,
result owner, decoder, EOS behavior, persistence format, or package surface.

## Accepted authority

`g3c4-generator-create-internal` has the private arity
`(model tokenizer config)`. Before native generator construction it authenticates
the exact live C4 model, looks up the tokenizer by `eq?` in the same aggregate's
T1 registry, and requires the canonical raw byte profile: V256, empty specials,
empty prefix and suffix, and fingerprint
`sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704`.
Foreign, copied, bare-core, and wrong-kind values reject as `invalid-argument`.
An authentic nonbaseline T1 profile rejects as `unsupported`. The authenticated
P1 root must report `eval` through `module-mode-internal`; train mode rejects as
`invalid-state` before native allocation.

The generator retains the exact T1 shell in slot 6 for its lifetime. No T1 core,
artifact bytes, native address, or detached fingerprint copy crosses into C. The
private call entry inherits the same shell in slot 6 and scrubs it with the rest
of its roots on success or rollback.

## Rechecks and cleanup

Call admission structurally authenticates the exact idle generator and model
cross-link, then redoes T1 registry membership, baseline fingerprint/profile,
and P1 eval checks before call-entry publication and native call acquire. It
repeats the tokenizer and eval checks after the trusted Eshkol body and directly
before native call prepare. Drift at that point follows the existing precommit
abort path and restores generator, model, aggregate guard, and native call state.
There is no callback or fallible Eshkol work between successful prepare and the
fail-stop native finish.

Abort, finish, generator close, and tombstone cleanup are never blocked by a
mode or tokenizer recheck. An idle live generator therefore remains closable in
train mode or after test-only T1 registry/fingerprint drift. Close clears slot 6;
an exact dead generator remains idempotently closable.

## Evidence and limits

The focused aggregate covers exact baseline/fingerprint retention, foreign and
copied aliases, authentic strict/prefix profiles, train-mode construction and
call rejection, registry withdrawal, fingerprint drift, post-acquire mode and
fingerprint rollback, exact call-slot inheritance, restoration, and cleanup in
train/stale states. It runs normally twice with byte-identical output and under
ASan, UBSan, and leak detection. The complete Step 6 call-entry gate remains the
failure-cut, publication-cut, retention, fail-stop, and native-regression gate.

This leaf does not authenticate prompt ownership or provide a T1-to-native i64
borrow. It does not allocate or publish generated IDs, text, lengths, cache
lengths, RNG results, or a public generation operation. Those ownership seams
remain separate successors.
