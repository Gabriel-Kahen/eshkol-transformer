# G3-R C4 trainable constructor and first workspace authority

**Proposed private contract; no constructor or workspace implementation is
admitted by this document.** The first gate establishes an exact trainable C4
model identity and a read-only workspace preflight. It does not run C4 forward,
backward, an optimizer, a trainer, C2 SAVE/LOAD, or generation.

## Existing authority and the missing transition

`native/g3c4_model_extension.esk` owns the sole C4 model registry and its
atomic seeded constructor. The 12-slot entry has fourteen distinct P1 handles,
one tied token-embedding path at head index 10, and position weight `[4,4]` at
index 13. The seeded entry is tagged `seeded`, has slot 7 `#f`, is prepared in
`eval`, and uses a native C4 owner bound to those handles once. The native
owner, P1 and I2 ledgers are existing authorities, not templates for parallel
registries. `native/g3c4_call_entry_extension.esk` currently admits only that
seeded tag and requires eval mode. Its diagnostic generator is a negative
control for trained-model identity.

The C2 M3T constructor checks context two and position `[2,4]`; its workspace
has two ID slots. It cannot be reinterpreted as C4. The accepted C4 I2 route
had only `i2-g3c4-construction-prepare-eval-internal!`, backed by P1 trusted
slot 71; slot 72 seals a prepared construction. The separate source candidate
adds **C4 train-mode prepare** in P1/I2 at append-only production slot 73.
It reuses the existing P1/I2 prepared seal and abort ledgers, without changing
the seeded path or the M3T route. The train sibling performs the same fallible
schedule, attachment, canonical-path, finalization, mode verification, and
native-prepare checks as `construction-prepare-eval!`, with final mode `train`.
The sibling at source `883cc8d6`/tree `2f5078b9` passed independent source
review (`g3r-c4-train-prepare-independent-source-review-20261025/SHA256SUMS-REVIEW`,
SHA-256 `e5191b96037cb84502911eed543980843bf8c55da4b20198536e8f71d11cf6cb`).
Its first pinned compiled attempt ended at launcher/Docker exit 126:
the generated caller could not execute from container `/tmp`, so no train
runtime or eval control ran. That nonaccepting receipt is
`g3r-c4-train-prepare-883cc8d6-attempt1/SHA256SUMS-RUN` (SHA-256
`584ad9d26190617d2cf89cd6a116ea502465e14293516b0103224938d4923d69`).
The separately reviewed v2 recipe tested an executable `/out/tmp` mount and
ran the same source/toolchain: train-prepared fixture 53 checks twice with
byte-identical output, plus inherited eval control 51 checks. Its 55-file
post-exit seal is `g3r-c4-train-prepare-883cc8d6-attempt2/SHA256SUMS-RUN`
(SHA-256 `fe3230dee056e5122a2a012cdb02093ad8ed3dda6743fc5d81c29e9e9a9da4f2`).
Independent result audit accepts only this normal/repeat P1/I2 seam. Native
sanitizers and broader inherited overlays remain unrun on this source.

## Proposed constructor authority

The private entry is a new **tagged branch in `g3c4-model-registry`**, not a
second model type or copied generator owner. Its proposed operation is
`g3c4-model-create-trainable-internal(resolved-config, tokenizer, initializer)`;
this name and signature are not implemented. Admit only the exact resolved
context-four profile: CPU f32, deterministic, vocabulary 256, hidden size 4,
one layer, two query and KV heads, head size 2, and context length 4.
Authenticate a live canonical raw V256 T1 shell with the existing T1 registry,
and require `run.seed` to equal the exact initializer seed. Preserve the X1
canonical bytes and fingerprint, T1 shell and fingerprint, initializer shell,
algorithm and four words under the entry's rooted trainable metadata. None of
these may be inferred from a matching printable fingerprint alone.

The 12-slot entry keeps existing slots 0..3 and 8..10 with their present
meanings. Slot 4 is a distinct `trainable` tag; slot 7 is a closed, fixed-shape
trainable metadata record, whereas it remains `#f` for `seeded`. Slot 11 is a
distinct trainable-C4 profile. Slots 5 and 6 retain authenticated original and
successor four-word RNG snapshots, with their transition specified and tested
before implementation. This is a tagged extension; seeded entry bytes and
ordinary feature-off admission stay unchanged. The exact metadata vector
layout and RNG transition require source review before implementation, so no
consumer may interpret slot 7 from this proposal alone.

Construction uses the existing aggregate guard, C4 model root/cleanup root,
native C4 owner, and C4 I2/P1 transaction. Validate config, T1 and initializer
before enrollment. Root a pending entry and cleanup ledger; create one native
owner; register the exact fourteen canonical ordered paths with shapes
`[4,4]` four times, `[4,8]`, `[8,4]`, `[4]` at 6..9, `[256,4]` at 10,
`[4]` at 11..12, and `[4,4]` at 13. Register `token_embedding.weight` as
the exact handle at index 10, never a fifteenth parameter. Initialize and
bind that owner once. Prepare `train` through the new P1/I2 sibling, prepare
the native owner, and complete all fallible cross-registry checks before the
existing no-failure prepared commits. Publish `live` only after both commits.
On every precommit error, revoke P1/I2 identities before aborting native
storage, mark the pending entry dead, and clear cleanup authority. A failed
attempt cannot expose a live model or a usable handle.

## First workspace gate: identity and shape only

The proposed private operation
`g3c4-trainable-workspace-preflight-internal(model)` returns a typed status,
not parameter arrays, tensors, pointers, or scratch. It admits the exact live
trainable shell in the **same C4 registry**, with matching rooted config, T1,
initializer and native owner; `train` mode; no active construction, generation
call, workspace or lease; exact fourteen ordered P1 handles and native
bindings; CPU f32 contiguous values of the listed shapes; and the exact
head/token-embedding tie. Native checking may scope the existing C4 fixed-14
pin begin/check/end operations inside one call, but must release pins on all
paths and expose only status to Eshkol. Admission leaves the model, parameter
bits and registry unchanged. No workspace allocation or forward operation is
authorized by a successful status.

Positive tests must use a newly constructed trainable owner in its original
registry. Negative controls must reject: the seeded diagnostic entry, foreign
or stale shell/owner/handle, changed config or T1 identity, context-two config,
wrong initializer seed, a swapped handle, wrong position shape or dtype,
broken tie, eval mode, active generation/workspace/lease, and incomplete
construction. Each failure must preserve the live model and unrelated
registry entries. A constructor fault matrix must separately prove rollback
before and after P1/I2 enrollment and native prepare. Feature-off builds must
retain the ordinary seeded ABI and reject private trainable symbols.

## Later gates

Trainable-to-eval must transition the **same live entry and owner in place**;
generation lookup must authenticate the new tagged entry and its exact bound
handles after that transition. Copying parameters into another seeded owner
does not count. A separate composition gate must connect the individually
reviewed C4 VJP leaves (including embedding and attention), give every
intermediate tensor an exact shape, and check complete gradients numerically.
Only later gates may add optimizer/TR3/D2, C2 SAVE and fresh LOAD into the same
registry, then G3 continuation. The isolated VJP leaves and this contract do
not establish any of those claims.
