# Exact private E3–D2 contract

**Exact design accepted by [root verdict5771845949](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/104#issuecomment-5771845949)
at reviewed head0a077a7.** This supersedes the proposal-state wording in the
source audit below; behavior is unchanged. Contract merge and explicit bounded
dispatch remain required; implementation and runtime evidence are not accepted.

Base `27c99f7`, branch `codex/e3-private-contract`. Proposal only; no implementation/build/ABI acceptance. Existing common-owner task `01a0c789-bd3f-7b21-b865-8b9365c7a282` owns canonical guard/pins and its single translation unit. This design adds no common source name, implementation or duplicate guard.

## 1. Fixed source variant, preserving the existing D2 tuple

Proposed checked-in new paths:

- `scripts/generate-e3-d2-source.py`: build-only deterministic source adapter, no runtime Python dependency.
- `native/e3_d2_source_variant.json`: exact predecessor path/SHA256, replacement-form SHA256 and output logical name; no caller-supplied substitutions.
- `internal/e3/lib/e3_d2_identity.esk`: the new fixed T1 adapter.
- `internal/e3/lib/e3_d2_restore.esk`: record/authentication/restore and three staging consumers.
- `native/e3_d2_native.h`: uninstalled declaration of the one new idle native seam.
- `native/e3_d2_native.c`: `#include "d2_native.c"` once, then the idle seam below.

Generated output path relative to private artifact build directory: `source/e3_d2_dataset.esk`. Its bytes equal the current `internal/d2/lib/d2_dataset.esk` except for **one entire exact definition** of `d2-tokenizer-identity`, replaced with:

```scheme
(define (d2-tokenizer-identity tokenizer)
  (let ((result (e3-d2-byte-identity tokenizer))) result))
```

The generator reads only those repository-owned inputs and an explicit output directory. It verifies the full predecessor SHA256, exactly one expected byte sequence spanning the existing definition, exactly one replacement definition, no remaining `t2-private-tokenizer-` reference in the result, and exact prefix/suffix preservation. It fails closed on predecessor drift, symlinked input/output, absent/duplicate form, arbitrary root/include/substitution overrides, or output escape. It writes deterministic LF bytes with no absolute-path/time header and emits source+replacement+generated hashes. It does not parse/evaluate Eshkol or copy a test carrier implementation. Checked-in exact expected form can be extracted from current lines643–654 into the manifest as UTF-8 text; the full predecessor hash prevents ambiguous extraction.

Reviewed predecessor hashes at this base:

- d2_dataset.esk: `d16224de20732fa9eaf4f9be990d33712986d2f94723d7517f7e9b653ebf2fcf`
- native/d2_native.c: `645ec4c0166b274045c9dced0bd2ee023b3fd7d75f5c2ff1821571f12be5d08c`
- native/d2_native.h: `165dc0184d288b5ea1eef0b3a6a4d5aee4c44e7a2acc1eb1d4d46908206163e8`

New private root includes M3 lineage once, existing `d2_semantic_core.esk`, `e3_d2_identity.esk`, generated `e3_d2_dataset.esk`, then `e3_d2_restore.esk` and E3 orchestration. It does **not** also load the original d2_dataset file or d2_wave2_root/T2 root. There is exactly one d2-tokenizer-identity definition and one set of D2 factories/singletons. The real compiler depfile must name the generated file; a separate checked-in provenance manifest binds its original source and generator rather than pretending it directly compiled the original. The exact private build tuple includes this deterministic preparation step. Existing D2 build scripts/source/manifests and its original T2 behavior remain byte-identical.

The private native tuple substitutes `native/e3_d2_native.c` for `native/d2_native.c`; it must never link both objects. Original D2 package continues compiling its existing TU. No new global leaks into the original D2 archive. The new private entry is localized in E3; list its header/TU and transitive d2_native.c/h in E3's actual native closure.

## 2. Exact byte-tokenizer adapter

`e3-d2-byte-identity/1 (tokenizer)` returns a newly owned two-slot vector `[fingerprint,V]`. Operation attributed to `token-dataset-open` because this replaces that operation's internal admission. Fixed steps:

1. `t1-private-require-core tokenizer 'token-dataset-open 'invalid-argument` authenticates against the one real T1 registry; it never accepts a copied core/tag or V-only claim.
2. Require current authenticated core vector length8/private tag, core[1] exactly `raw`, core[2]/[3]/[4] all empty (specials/prefix/suffix), and `t1-tokenizer-core-vocab-size` exactly256. A different authentic tokenizer profile is `unsupported`; a foreign/wrong-kind tokenizer is `invalid-argument`. No T2 access or lazy construction of a second tokenizer.
3. Obtain detached canonical fingerprint through `t1-tokenizer-core-fingerprint` and return `[fingerprint,256]`. Preserve this actual identity; never generate a digest from a guessed spelling.
4. Existing D2 open still compares V/fingerprint against D1 metadata and reports `invalid-argument` mismatch before publishing a dataset. It retains neither tokenizer nor this temporary vector.

Raw/empty byte tokenizer loaded through accepted T1 persistence is admissible if it authenticates and satisfies the same canonical profile. Merely matching V256 does not admit strict UTF-8 policy or another fingerprint. The canonical core slot use is source-private and pinned here, not a new public opaque accessor.

## 3. Exact native idle preflight

In `native/e3_d2_native.h`:

```c
int64_t et_e3_d2_dataset_idle_preflight_v1(const void *owner);
```

Implement inside the new single include-based D2 TU so it can use current static `find_dataset` and `status` without exporting raw registry pointers. `owner` is compared only, never dereferenced. Validation/result table:

| Condition, in order | Return and `et_d2_batch_last_status_v1()` |
|---|---:|
| owner NULL or no exact `find_dataset(owner)` enrollment | 1 (`ET_D2_NATIVE_STATUS_INVALID_ARGUMENT`) |
| enrolled dataset has `current_batch != NULL`, whether unpublished/sealed/borrowed | 2 (`ET_D2_NATIVE_STATUS_INVALID_STATE`) |
| enrolled dataset has `current_batch == NULL` | 0 (`ET_D2_NATIVE_STATUS_OK`) |

No other result is permitted. Active downstream leases live only in current_batch, so null current_batch implies no active lease; do not invent a separate lease field. Generation exhaustion does not defeat idle admission and no generation is changed. No reads from caller memory, allocation, error-record pointer, provider call, file I/O or resource mutation. Only existing private last_status changes. Source implementation is logically `dataset=find_dataset(owner); return status(!dataset?1:dataset->current_batch?2:0);` with explicit NULL admission. This is consistent with existing D2 close/release status meanings, not a new public C ABI.

Eshkol declaration is `extern i64 e3-d2-native-idle ptr :real et_e3_d2_dataset_idle_preflight_v1`. At initial admission map1→invalid-argument and2→invalid-state with the closed E3 operation. During cleanup, after prior authentic enrollment under exclusion, a nonzero idle result is an impossible cleanup invariant defect; terminate instead of replacing the first recoverable error or pretending rollback succeeded. Test live/unpublished batch and active lease all reject without mutation; after release idle succeeds.

## 4. Restore record authentication and lifetime

One Eshkol source-private record is allocated with its frame, never per call, never supplied as an arbitrary public value. Selected frame-facing accessor is `e3-frame-d2-record-internal/1 (frame)`: it must first perform E3's exact native/source frame authentication and returns only the frame's canonical record. This is the selected E3 frame-owned seam in this proposed exact contract, not assumed merged authority. The rooted Eshkol frame token is self-authenticated through its private source registry. D2 helpers take the authentic frame and fetch its record; they do not accept a bare vector/token supplied by the caller. Every admission requires `eq? record (e3-frame-d2-record-internal frame)`, record[1] eq record, record[2] eq frame. Reconstructing every field cannot satisfy this canonical identity check.

Exact proposed record layout, vector length14:

| Slot | Meaning |
|---:|---|
|0|one private singleton vector tag `e3-d2-restore-tag`|
|1|self, exact record identity|
|2|owning authenticated Eshkol frame identity, never the native pointer|
|3|phase symbol `idle`, `prepared`, or `restore-ready`|
|4|dataset shell; #f while idle|
|5|authenticated D2 state vector19; #f while idle|
|6|exact D2 core state vector17; #f while idle|
|7|saved next ordinal i64;0 while idle|
|8|saved M i64;0 while idle|
|9|exact normalized config object;#f while idle|
|10|exact D1 metadata object;#f while idle|
|11|exact manifest bytevector;#f while idle|
|12|one setup-allocated identity snapshot vector15 (core slots0..14), retained always; its15 entries clear to #f on idle|
|13|saved last-issued batch generation, lower-bound witness only;0 while idle|

The record/self/snapshot allocation belongs to the same proved enclosing lifetime as the frame; every dataset passed to the private fixture must already have true root-stable shell/state/core identities at native enrollment, created before any invocation/batch region. Merely living in an enclosing region is insufficient because promotion into this root record could change the native owner identity. The call's lexical batch regions are strictly younger. No dataset/shell is stored from a dying batch/invocation region. No per-call dataset promotion/copy is permitted; the trusted root-construction route and unchanged native owner identity must be proved before mutation. Cleanup cannot promote. Compiled poisoned-region proof is mandatory. Clearing slots4..11/13 and snapshot entries drops all dataset/cursor/config references after each call; record retains only itself, frame and its reusable snapshot. No process-global per-record/per-generation registry or tombstones are added.

D2 shell validation reuses existing `d2-open-dataset-state`; it checks procedure?, exact native factory kind1, guarded query, private tag/length/self and open bit before disclosure. Do not index a foreign opaque value.

## 5. Exact Eshkol restore helpers

All use fixed operation `e3-evaluate-into!`; no caller-selected operation/error/category argument.

- `e3-d2-prepare!/2 (frame,dataset)`: record must idle and frame in admitted idle evaluation phase. Authenticate dataset; require state[10] not live, state[15]=0 and native idle0. Validate configured N1/T2, unshuffled seed#f/window1, stored byte fingerprint/V256, packing boolean, semantic bound and remaining count cap before mutation. Check core tag/length, M=core[14]=state[17],0<=core[15]<=M. Fill all record bindings, core identity snapshot, saved ordinal/M and last generation only after admission. Phase becomes prepared last. Failure preserves idle record and dataset/modes/cursor; any temporary allocation remains invocation-local. No dataset mutation.
- `e3-d2-restore-preflight!/1 (frame)`: require prepared, same canonical record/frame, exact open shell/state/core/config/metadata/manifest references, unchanged core identity0..14, M consistent, saved ordinal in0..M, state[8]>=saved generation, current ordinal in0..M, status not live, borrow0 and native idle0. No filesystem/hash/cursor parse or allocation. On success mark restore-ready; this is still before restore writes. Under the closed transaction, violation after mutation is terminal invariant failure, not a recoverable error that leaves restoration abandoned.
- `e3-d2-restore-commit!/1 (frame)`: invoked only immediately after successful preflight under unchanged exclusion. Fixed nonfailing/nonallocating writes: core[15]=saved ordinal; state[10]='none; state[16]=#f. Clear record bindings/snapshot and set idle last. Return #t. Do not rewrite state[8], native next_batch_generation/next_lease_generation, state[15], native shuffle slots, corpus bytes or any other core field.
- `e3-d2-cancel-prepare!/1 (frame)`: prepared record only, known no next-batch mutation (frame flag) and current ordinal still saved; native idle must already be proved. Clear record to idle without touching dataset. Used after later preparation failure before traversal. If traversal could have changed anything, take restore preflight/commit even if final ordinal happens to match.

Identity snapshot comparisons use pointer identity for private vector/bytevector/string objects and exact immediate equality for scalar/symbol fields; no recursive copy, fingerprint recomputation or allocating `d2-core-state-valid?` during restoration (that function calls string->utf8). Immutable private identity contents plus serialized closed source call inventory are the basis of equality, not protection against malicious native mutation. Prepared-record bindings are cleared even at empty/EOS failure.

Native idle preflight is not a replacement for Eshkol source authentication, and release-preflight is never used on an idle dataset. Current D2 seek semantics are reproduced logically only: physical caches/shuffle contents and monotonic generation state are not rolled back.

## 6. Three fixed synchronous staging consumers

Proposed exact Eshkol names/arities:

- `e3-d2-inputs-consumer/2 (frame,view)`
- `e3-d2-targets-consumer/2 (frame,view)`
- `e3-d2-mask-consumer/2 (frame,view)`

`e3-stage-batch!/2 (frame,batch)` authenticates the exact current batch via existing D2 authority, verifies it belongs to record[4] and current state[8], and owns it for the interval. It obtains the three accepted tensor shells then performs, in order, `d2-token-tensor-with-view-internal` with three source-fixed unary lambda sites capturing only the authenticated frame. The callbacks invoke the corresponding named consumer. No caller callback, generic plane selector, nested borrow, escaped view, manufactured T1 shell or scalar loop exists.

Each consumer calls its corresponding fixed E3 frame-native copy entry. Selected native names are `et_e3_private_stage_inputs_v1`, `et_e3_private_stage_targets_v1`, `et_e3_private_stage_mask_v1`; each takes only authenticated frame and borrowed `const et_kernel_tensor_view_v1 *` and uses the frame contract's one diagnostic/status protocol. The E3 design selects these exact names/prototypes and the frame protocol below; they are **not additional D2 native registry/view exports**. D2 native ABI addition remains only section3's fully specified unary idle preflight.

Frame copy contract: require successive stage ordinals0→1→2→3, validate exact struct size, dtype i64/i64/bool, device cpu, dense layout, zero offset, rank2, shape[1,2], natural alignment, nonwrapping exact spans16/16/2, metadata/data disjoint from all destination/frame controls, and expected source-readability/trusted K1 pointer preconditions. Validate both IDs in[0,256) for inputs/targets and bool bytes0/1 with at least one1 for mask, including masked target position. All destination and readiness changes are atomic per copy; failures preserve that copy's destination bytes, leave previous staging unpublished, and clear whole-batch readiness during abort. Copy is an exact16/16/2-byte operation, no numeric conversion or allocation.

The original D2 wrapper authenticates owner/generation and provides the unchanged exact K1 view only during the current lease; no raw-view provenance is claimed outside that scoped call. Existing D2 guard ends the borrow even if a fixed consumer raises. A failure in any plane causes the transaction to release only its exact owned batch before its region ends, restore and keep all caller result destinations unchanged. Final stage3 alone permits forward.

## 7. Exact owned-batch release without new guard construction

Choose `e3-d2-release-owned!/2(frame,owned-cell)` in e3_d2_restore.esk. The
cell is source-private lexical storage in the current batch region, allocated
before next-batch, with exact four slots `[batch,generation,owned?,inspect-args]`.
Initial values are `[#f,0,#f,(cons #f '())]`. It never enters a root record or
escapes its region. Immediately after the trusted next-batch returns a non-EOS
shell, before another fallible query/operation, store the shell, current prepared
state[8] generation and true ownership, and set the frame's owned bit. These are
nonallocating same-region/older-value stores. Stage still authenticates the batch
normally; if that authentication fails, the already-recorded native ownership can
be released. A failed next-batch before return retains no E3 ownership; existing
D2 owns cleanup of its unpublished batch.

Release uses canonical frame/D2 record and the cell's just-acquired generation;
no public batch/dataset query, guard, closure construction, E1 reader or error
constructor is called. If owned? is false it returns#t without release. Otherwise
verify prepared record, open same root shell/state, state[8]==generation,
state[10]=='live, state[15]==0, frame ownership bit and exact cell lifetime. Call
existing `et_d2_batch_release_preflight_v1(record.dataset,generation)` through
its already-loaded D2 extern. On0 set state[10]='released, call existing
`et_d2_batch_release_v1` with those same operands, clear cell slots0/1/2 and frame
owned bit. Native release clears current_batch and frees its payload/control; it
already aborts on invalid generation/live lease. Any source/preflight/nonzero
post-release invariant defect uses the fixed source `e3-native-invariant-fail/0`
(libc abort), never an allocating error replacing the first failure. Do not modify
monotonic generations or use this helper on a caller-owned entry-live batch.

This selects the source-grounded native preflight/commit seam, not the public
`d2-token-batch-release!` validator. Its valid path queries opaque shells through
`guard`; pinned `eshkol_push_exception_handler` can malloc when its recycled frame
list is empty. Such allocation cannot be assumed absent just because a high-level
reader looks pure. The closed cleanup above creates no new guard. New E3 outer/inner handlers are installed before their own acquired work; their
allocation cost belongs to preparation/execution, with compiled failure tests.
The predecessor D2 with-view guard is installed after its native borrow begins;
its handler-allocation/runtime-exhaustion behavior is not an E3 noalloc or
recoverable-all-failures claim. Existing dependency behavior remains visible,
and runtime-exhaustion defects must not be reported as successful rollback.

For an ordinary E3 staging-native rejection, the fixed consumer snapshots its
first diagnostic/table index and returns the nonzero immediate status normally.
Existing D2 with-view then ends its lease on the normal path. The staging wrapper
explicitly tests `(= status 0)`, converts it to#t/#f, skips remaining planes on
nonzero and returns#f to the loop (all Scheme integer values are truthy);
it does not raise through the borrowed callback. Unexpected dependency exceptions
still use D2's existing borrow-end guard, then the same-region E3 classifier.
These are3 fixed consumer sites, no caller callback or new native D2 API.

## 8. Focused proof required before integration

- Deterministic generator run twice; exact one-form delta and expected hashes; wrong predecessor/duplicate/missing form, symlink and unexpected include/source inputs reject. Original D2 source, bytecode source closure and build tuple unchanged.
- Native idle table covers null/unknown/closed, open-empty, unpublished/live/sealed/borrowed batch, released batch and exhausted generation counters; receiver/control/allocation/generation snapshots unchanged, last_status exact. Same private TU owns registry.
- D2 prepare wrong/foreign/closed/live/busy/profile/boundary cases; copied record/vector/frame, cross-frame record, stale bindings and wrong phases rejected. Existing ordinary D2 aliases remain valid while allocation live.
- Success/failure/EOS/nonzero cursor restore exact canonical cursor bytes and next-batch semantics; monotonic generations preserved; fail every stage, later I/O, no batch/borrow survives; no allocating helpers in cleanup.
- Actual T1 raw/empty vs strict/special/prefix/suffix/foreign profile probes; matching V alone rejected; D1 metadata mismatch preserved.
- Three-stage transport metadata/alias/malformed/range checks, byte-preservation on failure; callbacks end lease on errors; poisoned-region and repeated-frame 1024/8192 witnesses, no retained dataset references after idle.
- Real private depfiles and localized symbol list prove original D2/T2 root and duplicate native objects absent; no test-only observer or new idle seam leaks into predecessor/public packages.

Read-only source inspection used explicit /usr/bin/bash, login=false. No production/docs edits, builds or suites. One report artifact written. Shared guard sources/ABI are deliberately not defined here; the designated common owner supplies them.

### Exact variant candidate calculated without writing generated code

A read-only Python bytes calculation over the pinned original proves the above exact replacement is unique and removes the only two T2 accessor references. Include the trailing two LF bytes after the definition in both source/replacement spans:

- Replaced definition:535 bytes, SHA256 `3833c0b46a54779cb2a587337359fb12546b739fb1f5a1d0d2044ee7c284ed27`.
- Replacement definition:102 bytes, SHA256 `2ac3329c7b432d29a1d734f4e6328a1ccec6fd309642c97849ca8b176687b0a6`.
- Generated candidate:44,982 bytes, SHA256 `d102ee50a33140271653f732f9dfea7c1c842c5c0922e99b4509ebffdc46ccaa`.

No generated source file or code change was made; these are deterministic design-candidate bytes, not compiler/runtime evidence.

E3-selected exact C prototypes within the proposed contract are:

```c
int64_t et_e3_private_stage_inputs_v1(
    void *frame, const et_kernel_tensor_view_v1 *view);
int64_t et_e3_private_stage_targets_v1(
    void *frame, const et_kernel_tensor_view_v1 *view);
int64_t et_e3_private_stage_mask_v1(
    void *frame, const et_kernel_tensor_view_v1 *view);
```

These frame entries use the frame's selected private global last_error_domain/category/code accessors, consistent with M3T. Eshkol captures diagnostics immediately into preallocated controls; native code creates no heap error. They use no new D2 status namespace. The D2-owned unary idle seam retains its independently specified 0/1/2 and D2 last_status contract. The E3 design selects the prototype/accessor split and `e3-frame-d2-record-internal/1`; only the designated common owner defines common guard/pin source. These selections are proposed exact E3 design, not root ABI acceptance.
