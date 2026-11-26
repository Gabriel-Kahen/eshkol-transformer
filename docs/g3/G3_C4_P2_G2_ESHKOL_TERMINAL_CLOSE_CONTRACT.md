# Private Eshkol P2/G2 terminal close and same-call coordinator contract

Status: independently reviewed private close contract with a proposed constructor-bound reservation prerequisite below. The native terminal commit and snapshot are merged, and the native logical I1 clone, copy, and release entries are present on the Wave 3 rehearsal. Reservation review, same-call implementation, and compiled proof are pending. Public G2, installed dispatch, and continuation remain separate acceptance gates. [Native terminal merge](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/d48eda71b63ec54f643b4779ec497a82d61d2c9d).

## Existing authority and distinct private entry

`native/m3_model_extension.esk`'s `m3-call` installs its guard before publishing busy state, returns the body's actual answer, then clears the busy cell. `native/g3c4_p2g2_protected_scope_extension.esk` authenticates original record/anchor, model/generator/tokenizer/policy/call/child edges and carrier allocation identities, but admits only before-prefix and prefix-committed phases and requires abort before returning #t. The prefix wrapper also returns #t. Neither existing wrapper can be used unchanged for terminal success: native terminal success has already drained its call.

Add a separate source-private terminal extension and entry, leaving those existing macros and their tests unchanged. Its admission still requires the authentic feature-gated exact private (P,G,N)=(2,2,1), CPU-f32/V256 model, same-aggregate raw T1 registry, and idle generator. It owns one `m3-call` region for both tokens. Its new scope returns only the original independently live terminal shell, never a call, pending owner, or arbitrary body value. This is a private fixed-profile diagnostic coordinator; public profile/EOS-policy/dispatch admission requires later source review and tests.

## Original allocation anchors before prefill

Idle input preflight captures original generator/model/tokenizer/core/policy and original owned P2 input shell/entry/native handle. Allocate the protected call ledger, pending output shell and capacity-two native owner, distinct first/second raw[1] and staging[8] carriers, pending raw[2]/staging[16], distinct logical-one text[1] and logical-two text[2], and a separate terminal-anchor record plus its independently rooted anchor/list nodes before first native prefill/sampling. Do not allocate a snapshot buffer in this region; snapshot is a later detached read. `g3c4-output-reserve` already enrolls the original pending output entry keyed by its original shell. Terminal close must transition that same entry and shell, never enroll a second same-shell entry that could shadow it in `g3c4-registry-find`. Native pending reservation follows authenticated call acquire; its failure still aborts an entry call before any prefill. Capture original linked-list successors before the eventual no-failure close.

The first/second raw[1] and staging[8], pending raw[2]/staging[16], and logical text[1]/text[2] imply exact header-plus-payload extents 9/16,10/24,9/10; retain those actual allocation-time identities and lengths. A rewritten mutable slot, paired replacement, or equal-length substitute cannot redefine an original allocation. The header-aware native API does not prove backing allocation extent: the Eshkol caller must authenticate the original bytevector and actual allocated extent before every ptr conversion. All carriers and shell/control storage are distinct, with full native alias checks retained.

The terminal anchor is separate from call/prefix anchor stacks and must survive their removal and `m3-call` return. A pending terminal anchor is rooted but cannot authorize reads or publication. It records the original output shell/entry/native handle, both original text carriers and their lengths, logical branch/count/cache expectations after authentication, and its own original record identity. No final RNG is available from the accepted Eshkol carriers before commit; leave it unknown here and obtain authentic final words only from a later native snapshot, not an invented precommit expectation. It does not retain an active-call/model/generator/tokenizer ownership edge after terminal success.

### Proposed constructor-bound reservation

The native commit authenticates the active call and its sole pending native
output. The pending-anchor leaf captures the existing reservation's auxiliary
pair. The constructor leaf attests eight allocations in its private lexical
root and currently retires them together. The historical composer also
allocates its own carriers and uses the existing reservation's fresh auxiliary
pair. The new private reservation must connect these authorities at initial
enrollment. [Merged native terminal](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/d48eda71b63ec54f643b4779ec497a82d61d2c9d),
[pending anchor, #218 head](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/23c544b66406b1e8a5397c3d0d566c56c178e3e3),
[carrier constructor, #226 head](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/c4ffa6ee83de3cde4e2359d8af139d12bfc84137),
[historical composer](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/aa9e661fd21f7779147b564347f066ef084ba09d).

Construct the bundle internally. Install its original pending raw[2] and
staging[16] in the sole output entry's auxiliary pair at first enrollment.
Use its original first and second raw[1]/staging[8] pairs and logical
text[1]/text[2] directly. Before sampling, bind the bundle, all eight original
objects and their extents, the protected and terminal anchors, the output
shell/entry/native owner, and the exact registry and root-list nodes and
successors to the authenticated call. Preserve the existing native reserve
ABI and the generic abort-only wrapper. Caller-supplied bundles, replacement
after enrollment, duplicate output entries, and any identity, extent, or
list substitution must reject before sampling or carrier pointer conversion.

Reservation may establish the private pending call and output edges. It leaves
the entry cache, RNG, and all earlier owners intact and creates no live result,
staged frame, or prefix commit. On a recoverable construction or reservation
failure, unwind only this attempt's installed edges and native pending owner
before propagating the original condition. If authentic abort rejects a held
I1 or A2 lease, preserve the same pending owner, all eight carrier payloads, and the
captured nodes while the exact holder releases its lease and abort is retried
in-region. Scrub and unlink the bundle only after abort succeeds. A release
capability must belong to the original holder and the same call. The historical
holder branch and diagnostic fixes do not yet qualify that path.
[Held-lease audit](https://github.com/Gabriel-Kahen/eshkol-transformer/blob/bffb4ab6f1a5a601f4644ab066376ae1734352d3/docs/g3/G3_C4_P2_G2_HELD_LEASE_PRODUCTION_HOLD.md),
[native holder candidate](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/44ad1362883131c5dbaa6ab3467cf231de5b32b7),
[status diagnostic](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/bffb4ab6f1a5a601f4644ab066376ae1734352d3),
[later-case control](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/14ae8395ed41306c3151281a67a91eae9b999b25).

Before native commit, preflight the one-use retirement and capture every
operand, including the selected original text and the root successor to
restore. Native zero first sets the preallocated success barrier. Then transfer
the selected text to the surviving terminal anchor and original output entry,
scrub the other seven carrier payloads, and revoke the pending bundle root
using the captured successor. The terminal anchor owns the selected text
until successful authenticated terminal release. Standalone retirement
scrubs all eight carriers and may rebuild its root list, so the closed
success tail cannot call it. The tail has only preflighted bounded writes,
with no allocation, registry search, recoverable call, or readback.

Implement after independent reservation contract review and integration of
the pending-anchor and carrier predecessor leaves. Their pinned normal,
repeat, and sanitizer runs passed at `12598d3b`. Preserve those gates and
their feature-off and allocation-cut checks when qualifying the reservation.
Its first executable proof must cover sole-entry identity, zero sampling on
reservation failure, original carrier substitutions, same-owner abort/retry,
and preservation of earlier owners. Terminal success additionally requires
independent review and pinned compiled evidence for any held-lease capability
it uses, plus the close and detached-result gates below. [Qualified
predecessor source](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/12598d3bc01c155eab0eeb89b26c11304ee3fcbd).

Run `python3 -B -m unittest tests.g3c4.test_p2_g2_constructor_reservation_boundary`
to check the predecessor boundary. The generic reserve constructs its own
auxiliary pair before enrollment, the abort-only scopes allocate their own
first and second pairs, and the constructor root authenticates its eight
allocations independently. The check preserves those ownership boundaries,
feature-off loading, and the existing order of pending enrollment and native
failure cleanup. A separately reviewed private entry must connect the
constructor bundle at first enrollment. The source check does not exercise
a constructor-bound reservation at runtime.

### Held-lease EOS diagnostic

The sealed `4774d540` run completed feature-on/off compilation. Its first
runtime case reached `eos.native.return` and failed before
`eos.check.return`. The trace records both the real carrier and wrapper
return markers but omits their numeric values. It leaves the origin of the
nonzero status unresolved. The reviewed `bffb4ab6` successor records the
real carrier status and native domain/category/code before the observer,
then reports the observer branch and result and the final wrapper status.
The separate `14ae8395` line tests a later unmarked case at
`case=2 phase=eos.guard.enter` after successful real EOS return.
[Failed source](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/4774d5406c36f8060cc6094ed42a3d2a631a5c1f),
[status provenance](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/bffb4ab6f1a5a601f4644ab066376ae1734352d3),
[later-case test](https://github.com/Gabriel-Kahen/eshkol-transformer/commit/14ae8395ed41306c3151281a67a91eae9b999b25).

Keep the diagnostic separate from the rehearsal. Combining the 210-path
static seal with the emitted feature-on/off Eshkol dependency lists and the
direct composer C fixture identifies 226 source paths. On `bcff170e` 19
are absent, including the private holder root, the holder native header,
and the three composer sources used by the feature-off build. Ten present
paths have different contents, including the native owner, C1 helper,
and coordinator fixture. The historical gate authenticates the entire
clean Git tree in addition to the explicit source hash inventory.

The next pinned run is the unclaimed historical package at
`/home/gabe/.codex/evidence/eshkol-transformer/p2g2-eos-status-bffb4ab6-compiled-20261014-a/result`.
Its read-only static verifier passed again with 210 source paths, four
dependency hashes, and eight negative controls. The invocation SHA-256 is
`fec2efb77ba51815c75061b0be15802d6b8cc96f5665af79824a4e8e66b19968`.
It pins Eshkol `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9` and image
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`.
On the supported build slot, run the sealed launcher with
`python3 -B run-once.py preflight <preflight-release>` and inspect its
toolchain receipt before `python3 -B run-once.py gate <gate-release>`. The gate invokes
`scripts/test-g3c4-p2g2-held-holder-compiled.sh` with read-only source and
dependencies, network disabled, and outputs under the package's result
directory. In the first normal-on fd-4 trace, a nonzero real-carrier status
identifies a carrier rejection. A zero real-carrier status followed by
`guard-reject` or `observer-reject` identifies the corresponding fixture
rejection. Production held-lease retry remains a separate compiled gate.

## Actual same-region sampling, decode, and rollback

Reserve the sole pending output before first sampling. Use the EOS-capable authentic first carrier on the original P2 input. Authenticate the complete LE i64 ID, then perform genuine same-registry T1 decode into the original first raw carrier; reauthenticate every original identity/edge and payload after each recoverable step. Do not provide logits, sampled IDs, RNG words, or cache candidates from Eshkol.

If the actual first ID equals authenticated policy EOS, take the native first-EOS terminal route and make no prefix commit or second draw. Otherwise perform the accepted non-EOS prefix commit, record the committed-prefix phase, use the original pending owner and retained native logits for the authentic second carrier, and decode that actual ID into the distinct original second raw carrier. Second EOS is emitted and consumes no extra draw. Original input checks for the second path must recognize the accepted post-prefix phase; the old before-prefix-only helper must not be silently called as if it admitted that state.

Before terminal commit, finish all fallible validation, T1 work, full original-anchor/registry/owner/model/policy/core/carrier/lease reauthentication, logical branch/cache expectations available before commit (no final-RNG readback), and bounded copying into unreachable preallocated logical-one/two text. The selected logical text must exactly match authentic T1 bytes; unused text remains unpublished. The native commit performs its own final A2 candidate/full-KV/transaction witness validation and derives logical count. Pass (first raw header,9,NULL,0) for first EOS or two distinct (header,9) carriers for G2.

Any nonzero native pre-tail result is recoverable while the original call remains live. Authenticated abort preserves entry cache if prefill failed, P2/cache2/mask1100 and original RNG after prefill but before prefix commit, or P3/cache3/mask1110 and advanced first RNG after prefix commit. No partial output publishes. After successful authenticated abort, scrub original carriers and pending text, tombstone the original pending output entry and separate pending terminal-anchor record, and restore/remove their captured pending anchor/list nodes without affecting earlier owners. A call-entry failure before native acquire likewise removes its own pending anchor and unpublished allocations. Cleanup must not leave a rooted pending terminal anchor advertising a dead output. Held genuine I1/A2 leases reject abort without mutation: while that call remains live, preserve its pending anchor/record for release and retry in-region; remove them only after abort succeeds. Escaping cleanup with forged owner/ledger/anchor is fail-stop. Do not claim rollback by resetting cache length or reconstructing a foreign cache.

## Zero return is the irreversible boundary

Native terminal zero return means output published, candidate K/V and RNG committed, and native call drained. Before that call, authenticate and capture all exact objects, vector sizes, list successors and fixed tail operands. Install a preallocated lexical terminal-success barrier and exception guard before native invocation. The zero branch first sets that barrier through a preflighted nonallocating write; from then on an impossible exception exits134 and must never call native abort/finish, wrap it as a recoverable error, or return a partial shell.

The tail uses only bounded writes to those captured objects: choose the proven logical count 1 or2 and text[1] or[2], transition the already-enrolled original pending output entry/shell to the distinct terminal-output kind/live state and mark its separate preallocated anchor live, remove child/model/generator active-call edges, tombstone the original call, scrub unused carriers/text, and restore original call/prefix stack successors. No allocation, user/test callback, decode, registry search, native readback, recoverable validator or P2/G1 publication helper follows native zero return. The single top-level coordinator returns the original terminal shell; it must not expose an extensible body that can execute arbitrary code after commit. `m3-call`'s existing final busy-cell clear and answer return are part of the audited tail. Native output already has no parent; the independently rooted terminal anchor remains authoritative after call anchors disappear.

The new Eshkol terminal kind is distinct from legacy P2/G1 output/result kinds, so old output validators/public adapters cannot reinterpret its shell. Release authenticates the exact registered terminal entry and original surviving anchor; same-owner dead release is idempotent. Native output-release rejects a held genuine I1 lease recoverably before mutation; successful release then scrubs only original terminal text/carriers and tombstones that output. Clone owners are independent and remain valid.

## Detached access and full proof gate

Only after the terminal region closes may a separate private `m3-call` allocate/validate an original snapshot destination, call the accepted 72-byte native snapshot, and authenticate ID0/ID1, logical count, cache length3/4 and four RNG words. Match IDs/count/cache with original retained text/branch facts; final RNG is first learned from this authenticated native snapshot and proved against independent Philox references in the gate. Do not claim comparison with an unavailable precommit RNG fact. If a private cached snapshot is retained, publish it only after all original-buffer/source/anchor reauthentication and result checks succeed; snapshot failure must not partially fill it. Snapshot errors leave the original destination/source usable. A later private adapter uses the accepted logical I1 clone/copy/release ABI and separate original allocation anchors to publish a genuine detached [1] or[2] owner; a bytevector alone is not a substitute. Do not cast the opaque clone to a public I1 handle. Additional public tensor/scalar accessors or public G2 facade require their own approved prerequisite contracts; scalar P2/G1 constructors still reject length2/cache4.

Compile the actual same-aggregate Eshkol coordinator normal/repeat/ASan+UBSan+LSan with independent G3-S/Philox and full-prefix P3/T4 references. Cover both prompts, greedy/categorical, first EOS, second EOS/non-EOS, singleton/carry/final/exhausted draws, full selected-ID/T1 equality, exact G1 unused-slot exclusion, snapshots, detached I1 shape/parity, source release before clone read, repeated releases, and original model parameter/gradient/binding invariance. Compile feature-off and unchanged P2/G1/nonterminal predecessors. Exercise every allocation/decode/native preflight cut, original/equal-length/paired substitutions, owner/core/ledger drift, genuine held leases and repair/retry, P2/P3 full rollback, and fail-stop at post-native-success fault cuts. Source and emitted-code checks must show no allocation or recoverable callback crosses the successful native boundary. Seal exact source/compiler/runner/runtime/image and private/public symbol closures. Accepted native-only terminal/clone gates cannot stand in for this Eshkol proof.
