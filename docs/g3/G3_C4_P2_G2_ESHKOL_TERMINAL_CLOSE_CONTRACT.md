# Private Eshkol P2/G2 terminal close and same-call coordinator contract

Status: independently reviewed private contract. The native terminal commit
and snapshot merged through
[PR #184](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/184), and
the native logical I1 clone, copy, and release merged through
[PR #189](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/189).
The pending-anchor and carrier prerequisites, constructor-bound reservation,
and compiled Eshkol same-call proof are separate gates. The accepted private
contract preserves the existing P2/G1 and generic abort-required protected
scope. Public G2, installed dispatch, and continuation require separate
acceptance.

## Existing authority and distinct private entry

`native/m3_model_extension.esk`'s `m3-call` installs its guard before publishing busy state, returns the body's actual answer, then clears the busy cell. `native/g3c4_p2g2_protected_scope_extension.esk` authenticates original record/anchor, model/generator/tokenizer/policy/call/child edges and carrier allocation identities, but admits only before-prefix and prefix-committed phases and requires abort before returning #t. The prefix wrapper also returns #t. Neither existing wrapper can be used unchanged for terminal success: native terminal success has already drained its call.

Add a separate source-private terminal extension and entry, leaving those existing macros and their tests unchanged. Its admission still requires the authentic feature-gated exact private (P,G,N)=(2,2,1), CPU-f32/V256 model, same-aggregate raw T1 registry, and idle generator. It owns one `m3-call` region for both tokens. Its new scope returns only the original independently live terminal shell, never a call, pending owner, or arbitrary body value. This is a private fixed-profile diagnostic coordinator; public profile/EOS-policy/dispatch admission requires later source review and tests.

## Original allocation anchors before prefill

Idle input preflight captures original generator/model/tokenizer/core/policy and original owned P2 input shell/entry/native handle. Allocate the protected call ledger, pending output shell and capacity-two native owner, distinct first/second raw[1] and staging[8] carriers, pending raw[2]/staging[16], distinct logical-one text[1] and logical-two text[2], and a separate terminal-anchor record plus its independently rooted anchor/list nodes before first native prefill/sampling. Do not allocate a snapshot buffer in this region; snapshot is a later detached read. `g3c4-output-reserve` already enrolls the original pending output entry keyed by its original shell. Terminal close must transition that same entry and shell, never enroll a second same-shell entry that could shadow it in `g3c4-registry-find`. Native pending reservation follows authenticated call acquire; its failure still aborts an entry call before any prefill. Capture original linked-list successors before the eventual no-failure close.

The first/second raw[1] and staging[8], pending raw[2]/staging[16], and logical text[1]/text[2] imply exact header-plus-payload extents 9/16,10/24,9/10; retain those actual allocation-time identities and lengths. A rewritten mutable slot, paired replacement, or equal-length substitute cannot redefine an original allocation. The header-aware native API does not prove backing allocation extent: the Eshkol caller must authenticate the original bytevector and actual allocated extent before every ptr conversion. All carriers and shell/control storage are distinct, with full native alias checks retained.

The terminal anchor is separate from call/prefix anchor stacks and must survive their removal and `m3-call` return. A pending terminal anchor is rooted but cannot authorize reads or publication. It records the original output shell/entry/native handle, both original text carriers and their lengths, logical branch/count/cache expectations after authentication, and its own original record identity. No final RNG is available from the accepted Eshkol carriers before commit; leave it unknown here and obtain authentic final words only from a later native snapshot, not an invented precommit expectation. It does not retain an active-call/model/generator/tokenizer ownership edge after terminal success.

### Idle owned-input prerequisite

The source-private idle admission uses the existing fixed-budget native
prefill preflight before creating a call or reserving output. The original
P2 input is an owned `[1,2]` i64 copy created through the admitted
same-aggregate T1 constructor. The generator's tokenizer is authenticated
through the baseline raw V256 registry. Preflight captures the exact current
registered generator, eval model and their native handles, tokenizer shell,
entry and core, policy identity and values, and the owned input shell, entry
and native handle. The returned lexical verifier takes the intended generator
and input again after any unpublished preparation. It rechecks the same
identities and runs native idle preflight before returning a fresh projection
of the captured handles. The later coordinator must call it while idle.

The native preflight authenticates its live idle generator, its internal
sealed model owner, the live registered input and the fixed P2/budget-two
tuple. The borrowed T1 source is copied at input construction, so the
resulting owner has no source T1 shell or tokenizer field. The native
preflight also receives no Scheme model or policy handle. The Scheme
reference checks establish drift from the captured registry state. They
supply no independent pre-construction provenance or call-bound release
authority.

The opt-in `G3C4_P2G2_TERMINAL_IDLE_INPUT_WITNESS=1` mode of
`scripts/test-g3c4-p2g2-t1-prefix-coordinator.sh` reuses the admitted native
aggregate and queues normal, repeat and sanitizer qualification. Its fixture
tests exact positive identities, same-content copies, genuine foreign
registrations, stage drift, released owners, and idle native model, RNG,
binding and cache state. Source checks pass. Aggregate execution remains
pending.

## Abort-only first-enrollment witness

The add-only `g3c4-with-p2g2-first-enrollment-probe` is a development source
macro for the exact private budget-two ownership gate and literal P2 reserve.
It reuses the admitted constructor, protected record authentication and native
reserve and abort. The development body observes the pending owner in one
`m3-call` and the probe returns only `#t` after abort. It accepts no prebuilt
bundle. The original auxiliary pair is present at the only registry insertion.
All list nodes are allocated before native acquire. The pending root and the
protected roots retain exact nodes and successors, and the complete registry
scan detects a second shell entry even if normal lookup still finds the first.

Before reserve failure propagates, authenticated abort drains the acquired call
and restores the pre-attempt Eshkol registry and root heads. Before native
acquire succeeds, the bundle is the only newly rooted allocation. The test gate
asserts earlier-owner preservation, native owner and I1 census, exact I1
active-borrow details, cache and RNG bytes, all eight allocation extents and
payloads, and same-call fixture-owned retry with a genuine foreign control.
The admitted allocation shim measures and challenges pre-acquire Eshkol
allocations. It disarms at the native call boundary so reservation and abort
cleanup have their own tests. The link guards observe external prefill and
token-frame entry points. Source call checks and the native cache, RNG and
token-frame snapshots cover the state of the shared aggregate.

`bash scripts/test-g3c4-p2g2-first-enrollment.sh /path/to/evidence` uses the
pinned normal, repeat and ASan/UBSan/LSan strict AOT recipe on Ubuntu 22.04
x86-64 with Eshkol `fe9dfd5` and Clang 21.1.8 under the shared build lock.
The source gate checks exact accepted native and installed source projection.
The admitted abort-only pending-anchor and carrier scripts remain predecessor
gates. Compiled first-enrollment evidence is pending.

The fixture holder is development-only and proves I1 retry for this
reservation. Integrating the idle owned-input prerequisite into reservation,
production call-bound I1/A2 release authority, terminal close, detached
readers, and dispatch remain later gates.

## Constructor retirement prerequisite

The source-private retirement scope preflights an exact current constructor
head and the original eight bytevectors. It captures the head's original
immutable successor, the six work carriers, the two lexical control cells and
the fixed selected and unused text. Logical count one selects original text[1]
and count two selects original text[2]. The direct retirement leaf consumes its
preallocated phase byte once, verifies the constructor is still the head, and
zeros the six work carriers and unused text before tombstoning the bundle and
restoring its exact successor. Selected identity and bytes survive.
The existing checker and abort retirement retain their original behavior.

The scope installs its guard before the caller can cross the synthetic
boundary. A normal return without activation expires the ticket. A recoverable
exception before activation expires it and re-raises the same object.
Repeated, expired or stale activation and ordinary exceptions after
activation fail-stop through `_Exit(134)`. The eventual coordinator must keep
the ticket internal and use its own fixed zero-return branch.

The three tagged writes compile through the checked region barrier. The
liveness flag and tombstone symbol are fixed before activation. The successor
already lives as long as the exact destination root cell because it is the
cdr of that cell's current head. In Eshkol `fe9dfd5`,
`region_evacuate_batch` uses its no-promotion tagged copy path for these
values. The emitted IR retains polymorphic tensor/error blocks for invalid
operands generated by checked primitives. For authentic preflighted operands,
the tail uses fixed indexed loads, literal byte stores and the three
non-promoting tagged writes. Runtime emergency errors bypass Scheme guards.

`bash scripts/test-g3c4-p2g2-terminal-carrier-retirement.sh /path/to/evidence`
queues the pinned strict-AOT normal, repeat and ASan/UBSan/LSan poisoned gate.
It checks the emitted static leaf, wraps the reachable allocation interfaces and
counts its three tagged writes. The fixture covers both original selected
texts, empty and nonempty successors, authentic predecessor states, rejection
and unchanged bytes before activation, and the four fail-stop cases. A small
pinned harness with aggregate dependencies stubbed passes 339 poisoned checks
and the four fail-stop exits. Full aggregate qualification and real native
terminal commit remain pending.

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
