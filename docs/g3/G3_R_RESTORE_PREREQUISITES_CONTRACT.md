# G3-R pair admission and no-draw replay prerequisites

**Proposed private implementation contract; no restore or continuation is accepted
by this document.** This is the bounded `diagnostic-c2`, CPU-f32, batch-one,
raw-V256, context-two prerequisite in
[G3-R data-only continuation](G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md).
It retains the accepted 288-byte G3RCV1 codec and typed G3-T RNG import. It
adds no checkpoint writer, public entry, new C2 wire field, replayed-prefix
sample/commit path, C4 owner conversion, or larger generation profile. Review
the two native signatures below independently before source implementation.

## Source boundary and authoritative C2 bytes

`native/c2_checkpoint_load_bridge.h` says measure and stage each open and
validate their own C2 image; only the second image supplies content. The
existing stage implementation in `native/c2_checkpoint_load_bridge.c` obtains
`view->bytes` and `view->file_bytes` from one validated retained image, copies
eight component spans, then releases that image. Its 192-byte result does not
contain a whole-image digest. `native/c2_checkpoint_reader.c` checks regular
file size and exact EOF, but cannot make a separate read of the path identical
to the stage image. C2's domain-separated trailing digest is not SHA-256 of
the complete image. Hashing or statting the path before or after LOAD cannot
authenticate G3RCV1's bytes `224..255` when a same-size valid replacement can
occur between opens.

Under `ET_G3R_C2_STAGE_SHA_PRIVATE`, propose this **additional** native symbol;
leave `et_c2_private_checkpoint_load_stage_v1` and the public C2 LOAD ABI
unchanged:

```c
int64_t et_c2_private_checkpoint_load_stage_with_sha_v1(
    const void *path, int64_t maximum_file_bytes,
    int64_t maximum_metadata_bytes, int64_t maximum_tensor_bytes,
    int64_t maximum_tensors, int64_t enforce_operational_profile,
    const void *measurement, void *tokenizer_fingerprint,
    void *config_fingerprint, void *x1_canonical,
    void *current_cursor, void *epoch_cursor, void *model_c1,
    void *optimizer_metadata, void *optimizer_payload,
    void *result_with_sha);
```

All pointers use the existing Eshkol signed-i64 bytevector-header convention.
`result_with_sha` is an exact 224-byte private destination: offsets `0..191`
retain the existing result layout and offsets `192..223` hold the ordinary
whole-image SHA-256. The public result stays exactly 192 bytes, and
measurement remains exactly 192 bytes. This keeps the current 16-argument
Eshkol FFI arity. The Eshkol wrapper must
allocate and retain the actual bytevectors and authenticate their backing
lengths. Native can validate only header declarations and pointer/span
arithmetic; the owning Eshkol wrapper must prove the actual backing extent
of every passed bytevector before its native call. Native must reject invalid
declared spans before I/O or copy; the whole 224-byte result must
be disjoint from path, measurement and all eight destinations. No interior
alias is allowed. The signature supplies no expected digest and accepts no caller
digest as authority. On successful stage, compute ordinary SHA-256 over
`view->bytes[0..view->file_bytes)` from the **same** second retained image
that supplied the eight components, before releasing it. Use bounded,
allocation-free SHA state. Do not hash just the C2 trailer, model segment,
first measure image, or a reopened path. Match the existing stage's
size-race/status behavior; invalid or aliased spans reject **without touching
any caller buffer**. Once valid, disjoint 224-byte result storage is admitted,
later stage or image-release failure retains the existing 192-byte C2 failure
record, including its diagnostic fields at `136..184`, and clears only the
private SHA tail `192..223`. Changed component lengths reject before copying
any component or digest. Digest and component outputs are
private staging scratch until the full C2/K2 load finishes. A stage or image
release failure must not publish a C2 owner or digest receipt; a cleanup defect
after private copy is internal/fail-stop if it cannot be rolled back.

The proposed Eshkol entry is
`(g3r-c2-load-with-sha-internal path policy report) -> receipt-shell`, in a
feature-local owning aggregate. It follows `c2-public-checkpoint-load` in
`native/c2_public_extension.esk`: private stage with SHA, exact K2
`tensor.f32/storage.copy` capability admission, C2 reconstruction, and stage
cleanup. The exact stage shell and its copied digest travel together in a
private registry sidecar through reconstruction; an arbitrary digest vector
must not be attached to a different C2 owner. The returned receipt is a fresh
registered identity whose canonical record owns exactly one live detached C2
owner and a copied 32-byte SHA. No receipt is returned before both are ready.
The caller may borrow that owner and digest only through the owning aggregate's
exact live receipt; release is borrow-aware, idempotent, zeroes digest bytes,
and releases the C2 owner once. If any later admission, model construction or
replay fails, the candidate ledger releases the receipt; an existing caller
receiver remains untouched. Never publish the receipt from measurement alone.
The accepted G3RCV1 decoder receives this authenticated SHA as its expected
digest; its ordinary in-memory bytevector parameter does not confer C2 file
authority by itself.

## Idle G3-T RNG equality witness

`src/eshkol_transformer/g3t_transport.c` has an exact-address registry,
kind-1 generator, four inline i64 RNG words, active-call links and model pins.
The accepted typed import creates a separate G3-T kind-8 owner; the `:rng`
constructor copies its words into the generator. The only current-generator
word reader is `ET_G3T_TESTING`-only, and the production clone reads a
completed output. Neither can establish no-draw replay in production.

Under `ET_G3R_G3T_RNG_EQUAL_PRIVATE`, requiring
`ET_G3T_GENERATOR_RNG_PRIVATE`, propose this status-only native symbol:

```c
int64_t et_g3t_private_idle_rng_words_equal_v1(
    void *generator, int64_t version, int64_t seed,
    int64_t counter_low, int64_t counter_high);
```

Return `0` only for an exact four-word match on a registered, live, kind-1
generator with its authentic live `c->model`, idle frame/call, no active model
link (`o->active == NULL`), no pins and no **pending** output/logits pointers;
otherwise return the existing G3-T category/status with TLS
domain/category/code set. Search the G3-T registry
before dereferencing `generator`; validate `version == 1` and `seed >= 0`.
Treat word mismatch as `G3T_STATE/G3T_INVARIANT`, wrong-kind/foreign as
`G3T_ARGUMENT/G3T_IDENTITY`, and busy/dead as lifecycle rejection. The
predicate allocates, borrows, samples, copies, mutates and publishes nothing;
it returns no pointer, RNG shell, raw vector or output buffer. Both signed
counter words are compared bit-for-bit, including carry and exhaustion.
An idle generator after a real manual commit/call-finish may legitimately have
`prefill_committed`, committed cache/KV, `binding_ready` and `last_logits`, and
the returned live logits owner is detached from `c->pending_logits`; these
states must pass when the other idle checks hold. `o->active == NULL` does
**not** mean `c->model == NULL`. Reject an active frame, held pins or an
actual pending pointer, not a completed manual replay.
The Eshkol restore aggregate calls it after typed `:rng` generator creation
and after each closed manual replay frame, using the four independently
authenticated G3RCV1 words. Same-aggregate exact Eshkol generator admission
precedes the native call. Test-only word readers remain test-only. C4's
different kind-8 result RNG and kind-9 clone are never accepted as G3-T
authority.

## Fresh candidate recipe and publication

The first implementation accepts external resolved-X1 config, registered
canonical raw T1 tokenizer and D2 corpus only as construction inputs for a
**fresh candidate**. Before TR3 restore, compare the loaded C2 borrow's
canonical X1/config fingerprint, tokenizer fingerprint, dataset/cursor
identity, fixed 14-path model schema, dtype/device and limits with those
inputs and G3RCV1's profile/fingerprint. The caller's resource policy can
lower, never widen, C2 bounds. A hard-coded equal-seed model is only an
initial receiver scaffold; it is never proof of restored parameters. Use the
actual `cli3_generate_extension.esk` pattern: create M3T model and optimizer
under one same-registry candidate ledger, create trainer with exact T1/D2,
TR3-restore the authenticated detached C2 owner into **that candidate only**,
release its trainer lease, switch its model to eval, and verify all 14
parameter paths/shapes/tied alias and bitwise loaded values. The candidate
retains independently live model/tokenizer binding when the C2 receipt and
trainer are released. Never restore into the caller's current model/trainer.
The loaded C2 X1/dropout RNG is not G3-S; import G3RCV1's four checked G3-S
words through `g3r-rng-owner-from-checked-words`, then use G3-T's `:rng`
constructor and release the import owner after the generator copied it.

The aggregate owns a copied ordered raw history and its prompt/generated/
manual counts and EOS-terminal derivation. For length one, use real T1 input
and `g3m-prefill-p1!`; for length two, use real T1 `[1,2]` input and
`g3m-prefill-p2!`, or P1 plus admitted manual decode when that route is
specifically witnessed. Never call `generator-generate!` during replay.
Check the idle RNG predicate before and after each real frame; compare
authentic last-row all-256 logit bits, cache K/V/mask/length and original
model/gradient state against an independent uninterrupted owner. Allocate,
register and enroll every candidate child before the next fallible call.
Rollback releases only candidate input/logits/generator/import RNG/trainer/
dataset/C2 owners in dependency order, preserving the first exception and
the existing receiver/outputs. All leases, release preflights and
authentications precede a single assignment-only candidate publication;
after the first closed-tail mutation an unexpected failure exits 134. A
forged receipt, foreign RNG, held I1/A2/model lease or busy frame rejects
without altering the caller and permits a fresh retry.

## Acceptance and limits

Native C2 tests: independent whole-file SHA oracle on exact complete images;
same-size valid A/B replacement between measure and stage must bind B's
owner **and** B's SHA; mutation/replacement after stage cannot change the
receipt; truncated/torn, changed component size, image-release and allocation
faults leave no receipt. Test the 224-byte result's header/interior aliases
with path, measurement and each of eight outputs; check every rejected buffer
and native live-image count. Existing public LOAD bytes/status and feature-off
symbols remain identical. Native G3-T tests: true idle match, each-word
mismatch, high-bit/carry/final/exhausted words, foreign/stale/wrong-kind/C4
pointer, active call, pins, pending output and model lease. Assert no cache,
RNG, registry or output changes on every rejection, feature-off symbol
absence and correct error provenance. Run normal/repeat and
ASan+UBSan+LSan, source/symbol closure and Q0 checks.

An owning Eshkol aggregate must create an actual C2 checkpoint and companion
in separate producer/consumer processes, compare bitwise 14 restored model
parameters and gradients, T1/X1/policy/history, all-256 replay logits,
K/V/masks/lengths and all four RNG words, and exercise greedy/categorical,
singleton/carry/final/exhausted counter cases. Independent G3-S/Philox
reference must confirm any genuine next draw. Missing/orphaned/mismatched
pair, malformed 288-byte record, wrong profile/schema, held leases, provider
and allocation cuts must preserve an existing receiver and permit retry.
The full length-two `max_new=1` replay admits reconstruction but a subsequent
new request rejects capacity before draw. Authentic resumed ID/raw/output
publication still depends on a separately reviewed replayed-prefix
sample/commit API, which is outside this prerequisite. The two-token profile
does not prove repeated decode, later EOS, general G3-R or public resume.
