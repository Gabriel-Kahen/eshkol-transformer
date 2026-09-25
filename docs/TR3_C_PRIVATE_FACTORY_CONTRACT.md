# TR3-C private same-package trainer factory and close contract

Status: **proposed contract for review; no implementation, export, public trainer,
or resume acceptance**. Source base: integration commit `76299fd`. This freezes a
bounded linked-package proof seam, not CLI3 policy or a general training API.

## Accepted source and missing boundary

The one linked [private root](TR3_C_LINKED_PRIVATE_PACKAGE.md) already contains
`x1-private-config-parse/resolve`, `t2-wave2-tokenizer-byte`,
`d2-token-dataset-open/close!`, `m3t-public-initializer-state/model-create`,
`module-parameters`, `o2-public-optimizer-create`, and
`tr3-lease-create-internal/unenroll-internal!`. The fixture in
`tests/tr3_snapshot/runtime_smoke.esk:69-110` calls them in one compiled
identity universe. Only `et_tr3_c_private_initialize_v1` and its version marker
are dynamically exported today. There is **no accepted same-package factory
wrapper, per-call E1 bridge, C trainer handle, or C status ABI**. Those are the
remaining implementation prerequisites; localized source C thunks alone cannot
receive identities from another package.

## Proposed version-1 request and producer policy

Two private C functions would use fixed, naturally aligned LP64 SysV structs:
`et_tr3_c_private_trainer_create_v1(const request *)` and
`et_tr3_c_private_trainer_close_v1(handle *)`, each returning a result by value.
The proposed exact layout is:

```c
typedef struct et_tr3_c_private_trainer_handle_v1 et_tr3_c_handle_v1;
typedef struct {
  uint32_t size, major, minor, flags;  /* 176, 1, 0, 0 */
  const uint8_t *x1_json;
  uint32_t x1_len;
  const uint8_t *directory;
  uint32_t directory_len;
  uint8_t expected_manifest_sha256[32];
  int64_t batch_size, sequence_length, maximum_manifest_bytes;
  int64_t maximum_shard_bytes, maximum_total_tokens, maximum_batch_bytes;
  int64_t shuffle_seed, shuffle_window_rows;
  uint32_t adamw_f32_bits[5]; /* learning rate, beta1, beta2, epsilon, decay */
  uint8_t shuffle_seed_present, packing;
  uint8_t reserved[6];
} et_tr3_c_create_request_v1;
typedef struct {
  uint32_t status, stage, reason, original_category;
  et_tr3_c_handle_v1 *handle;
} et_tr3_c_result_v1; /* 24 bytes */
```

The request is exactly 176 bytes on the pinned ABI. Unknown sizes or versions,
nonzero flags/reserved, and noncanonical absent shuffle seed are rejected,
not partially interpreted. A null request returns `invalid-argument`; the
caller must provide readable request and byte spans. It contains only the
following data:

| Field | Version-1 rule |
| --- | --- |
| X1 text | Pointer plus byte length `1..16384`; exact ASCII X1 schema JSON, no embedded NUL; empty override list. |
| Corpus | Directory UTF-8 pointer plus length `1..4096`, no embedded NUL, and an expected 32-byte D1 manifest SHA-256 trailer. No environment expansion or implicit path. |
| D2 | Exact ten-key values: directory above; positive signed-i64 batch size, sequence length, manifest/shard/batch byte limits, shuffle-window rows; nonnegative signed-i64 maximum total tokens; explicit optional nonnegative signed-i64 shuffle seed; explicit boolean packing. |
| O2 | Five raw binary32 bit patterns for learning rate, beta1, beta2, epsilon and weight decay. They are rendered as eight lowercase hex digits for O2's accepted version `1.0` data envelope. |

The byte tokenizer, M3T `eshkol-diagnostic-byte-decoder-v1`, AdamW dense CPU
f32, no clipping, constant schedule, and **one group covering all 14 canonical
unique model parameter paths** are fixed version-1 choices. Paths come from
the authenticated newly created model's `module-parameters`/M3T handles in
canonical order; callers supply no path or Eshkol object. The tied
`token_embedding/weight` alias is not a fifteenth group entry. X1's resolved
`run.seed` is the *only* M3T initializer seed; there is no independent seed
field. The X1 profile must match `m3t-config-check` before initializer/model
creation, including context 2, hidden size 4, one layer, two query/KV heads,
head size 2, vocabulary 256, CPU f32 and deterministic true. D2 sequence
length is 2 and batch size is 1 in this proof version. The other D2 bounds,
shuffle and packing options remain caller-selected and are passed to D2's
existing validator. The corpus digest is compared with bytes 128..159 of the
accepted `ESHKDCU1` cursor after D2 open and before M3T creation; a mismatch
is `invalid-argument` and closes that newly opened D2 receiver.

All pointer/length pairs are borrowed only until return. C bounds and scans
exactly the supplied byte lengths, rejecting embedded NUL, then copies each
span into an owned `length + 1` buffer and appends a NUL before calling the
runtime's `strlen`-based string copier. It never scans beyond a borrowed span;
the owned buffers are released after the tagged strings are constructed. The
bridge constructs
primitive tagged strings/integers in this library's arena; a new same-package
Eshkol wrapper builds the exact D2/O2 lists and calls the accepted producers.
No C-built Scheme list, raw tagged receiver from another package, executable
input, tokenizer artifact, arbitrary M3T profile, or alternate O2 schedule is
admitted by version 1.

## Admission, publication, and failure cuts

The deterministic order is: (1) verify initializer readiness and acquire one
nonreentrant C operation guard; (2) validate request structure, bytes, scalar
ranges, fixed D2 values and O2 bit domains without creating source operands;
(3) claim the process's **single producer attempt**; (4) X1 parse, resolve
with `()`, validate fixed profile and derive `run.seed`; (5) T2 byte tokenizer;
(6) D2 open, cursor/digest check; (7) M3T initializer then model; (8) obtain
the model parameter tree/14 authenticated paths and create O2; (9) call the
genuine five-operand lease constructor; (10) root the C handle record and
publish it only after successful enrollment. A malformed pre-attempt request
does not consume the attempt. Any source-stage failure consumes it; there is
no second producer run or hidden retry in this proof version. Concurrent or
reentrant calls return busy before the attempt or receiver changes.

Each successful child is immediately entered in a package root before the next
fallible call. If a producer fails before returning a receiver, the factory
has no child to close; it relies only on that producer's accepted internal
failure behavior and claims no generic rollback of arena allocations. Once D2 returns, any subsequent
failure calls `d2-token-dataset-close!` under a cleanup exception scope; the
closed shell remains rooted. M3T initializer/model and O2 have no accepted
trainer-owned destroy, so any successfully returned receiver remains rooted
until process exit even when a later stage fails. The lease constructor's own
failure handler clears its staging root/guard and does **not** own or destroy
the five operands. If D2 close itself fails, the factory retains all children,
reports `internal` with a cleanup-failed reason, and preserves the original
producer category in copied result metadata. No partial C handle is returned.

On success, a package-owned record retains the exact shell and all five child
receivers through close. The accepted lease registry also retains them while
enrolled. Close calls only `tr3-lease-unenroll-internal!` on the exact shell;
it does not call D2 close or destroy M3T/O2. The package root keeps children
and the closed handle tombstone until process exit. Thus close ends exclusive
trainer enrollment, not process-local model/optimizer/dataset storage. No
physical arena reclamation is claimed.

## Opaque handle and results

The C handle is a pointer to an incomplete private type, returned only through
the factory result. It is a process-local identity, neither serializable nor
transferable across package instances. The package retains its record and
tombstone for life, never reuses the address, and compares a candidate against
its registry **before dereferencing it**. An exact copied pointer aliases the
same handle; arbitrary/null/different-package pointers are `invalid-argument`
with `bad-handle`. A first idle close changes `live` to `closed` only after
successful unenrollment. Repeated exact close is `invalid-state` with
`already-closed`. A busy TR3 phase, parent, M3T/O2/D2 dependency, acquisition
or overlapping C call is `invalid-state` with `busy`; it leaves the handle live
and permits retry. Any other failed unenrollment also leaves it live.

Each call returns one caller-owned, fixed-size result: status, stage, reason,
optional original producer category on cleanup failure, and a handle that is
null on every failure. It contains no borrowed E1 pointer, message, tagged
value or arena address other than the opaque handle; scalar fields survive
return and may be copied. The caller never frees the handle. The C bridge
opens its own parallel/E1 exception scope on **each** call, preserves the
raised value across handler cleanup, and lets a same-package E1-aware wrapper
classify authentic `transformer-error?` values. All 12 E1 categories map
one-to-one to C statuses, in this fixed numeric order: `ok=0`,
`invalid-argument=1`, `shape-mismatch=2`, `dtype-mismatch=3`,
`device-mismatch=4`, `noncontiguous=5`, `unsupported=6`,
`invalid-state=7`, `io=8`, `corrupt-data=9`, `version-mismatch=10`,
`determinism-unavailable=11`, `internal=12`. `stage` is `none=0`,
`admission=1`, `x1=2`, `t2=3`, `d2=4`, `corpus-identity=5`,
`m3t-initializer=6`, `m3t-model=7`, `o2=8`, `lease=9`, or `close=10`.
`reason` is `raised-e1=0`, `not-ready=1`, `busy=2`, `attempt-used=3`,
`bad-handle=4`, `already-closed=5`, `digest-mismatch=6`,
`cleanup-failed=7`, or `foreign-exception=8`. `original_category` is zero
except on cleanup failure, when it holds the first failing E1 category. A foreign
or unclassifiable raise maps to `internal` with `foreign-exception`; it never
becomes `ok` or a partial handle. C readiness/busy/attempt exhaustion and
bad/repeat handle outcomes map to `invalid-state` or `invalid-argument` as
specified above, with a distinct reason. The initializer's existing
`READY/BUSY/RUNTIME/ARENA/HANDLER/EXCEPTION` codes remain separate and unchanged.

The caller must invoke the initializer first and keep the `.so` loaded from
that call through process exit. This package's hosted runtime and shared arena
remain process-owned; factory/close never shut them down or `dlclose` the
package. Calling after unload, from another runtime/identity universe, or
after process fork is unsupported. No concurrent use or thread-safety is
claimed beyond immediate busy rejection.

## Acceptance gate and decision

Keep the dynamic manifest initializer-only until one linked-library test
compiles the proposed wrapper and invokes these exact C functions through
`dlopen`/`dlsym`. It must cover malformed bytes/lengths/version, fixed-profile
and seed derivation, digest mismatch, each producer and lease failure cut,
cleanup failure, no partial handle, exact idle unlink, busy/forged/repeated
close, process-root retention, per-call E1 categorization, and hostile-link
negatives for every localized source operation. This proposal permits a
bounded private C construct/close proof; it does not justify public trainer
exports or exact-resume claims.
