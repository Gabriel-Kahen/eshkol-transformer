# G3-T readiness and future acceptance witnesses

**Planning only; no runtime gate has run here.** This checklist is anchored to
merged main `ba0e37d06076d0a16c473ff742ab95723cb2cb89`. It operationalizes the
accepted [G3-T private contract](G3_T_PRIVATE_CONTRACT.md) and
[shared call/pin contract](../E3_SHARED_CALL_CONTRACT.md); it changes no signature,
slot, owner kind, numerical rule, public surface or dependency hold. The proposed
witnesses below must use the eventual accepted implementations and exact artifact.

## Dependency disposition

| Dependency | Evidence at this planning baseline | Consequence |
|---|---|---|
| G3-N | [Accepted implementation/provenance](../G3N_PRIMITIVES.md#integration-provenance); PR #97 merge `ca3880f6e5da4850bafbf349a846de7829203e51`, acceptance closeout PR #103 | Numerical prerequisite satisfied within its exact eleven-row provider; not generation proof |
| Shared guard/fixed14 | [Root acceptance](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/102#issuecomment-5771662657), contract PR #102 merge `2ac0b5bc5a385e79700f46bf899bab841f33d1cb`; canonical implementation belongs to E3-CG #101 | Contract available; implementation acceptance/merge and consumer integration proof still required. The three canonical runtime files are absent from this main baseline |
| G3-S | Continuation direction retains PR #96's unresolved restricted-review hold | Remains a hard gate. This work does not inspect, duplicate, reroute or reproduce that review, and proposes no replacement provider, mock or bypass |
| G3-T | Accepted exact design and shared-source refinements; no runtime acceptance | No runtime implementation or test dispatch until required implementations are accepted/merged and root dispatches the bounded work |
| G3-M / G3-G / G3-R / larger context | Separate downstream composition, public operations, continuation and profile decisions | No public generation, save/reload, repeated-decode or larger-profile completion follows from T |

The exact E3 private consumer design also merged in PR #104 at this baseline.
That does not make E3 a G3 runtime dependency. E3 owns mode/cursor restoration;
G3 stays eval-only. If the canonical P1 69-slot prerequisite lands before G3,
consume the reviewed merged P1 lineage and its inheritor regressions; do not add
E3 mode authority to G3 or assume an old manifest still matches. G3 itself adds
no P1 slot. Source/ELF inventories must be measured on the actual eventual base.

After baseline reconciliation, the shared owner reported implementation
[PR #105](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/105) at
`e3d8566eaa33eec2c7fb2a4ef278b1c53b0d3acf`, with
[independent approval](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/105#issuecomment-5772575224)
and [supported CI](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35693360705)
passed. The owner's subsequent
[refresh record](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/105#issuecomment-5777816153)
identifies integration head `9ae35ecbae04031bb655925545d4176681742c3d`, tree
`d5e8a801eeaab3ea666f37dfd8d83daaf473f379`, with independent documentation-delta
approval and unchanged non-Markdown blobs/modes. Its new exact-tree
[CI run](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35736728847)
was running at handoff; root acceptance/merge remained pending. The original CI
is historical candidate evidence. These owner-reported results are not new
execution or acceptance by the G3 task and do not satisfy the merged-dependency
gate.

No larger-profile or persistence contract is opened by this readiness work.
Their unresolved limitations remain explicit in the original G3 proposal.

## Proof ownership and artifact boundaries

E3-CG owns the one shared adapter set and pin mechanics, including their direct
adversarial tests. G3 consumes their accepted artifact and proves context/model
linkage, nested-call discipline and lifetime integration. It must not duplicate
the shared implementation or treat E3 consumer success as G3 proof.

The normal G3-T sibling retains seven facades, 87 boxed exports and 93 globals.
Its 29 private native seams and 95 named Eshkol bindings remain exactly the
accepted inventory. A separate development-only source-composed witness may
invoke the closed transport steps and instrument ownership/failures. Its exact
observer/failpoint closure must be reviewed; this checklist introduces no new
callable test API. Normal archives contain no observers, parity entry points or
failpoints. No production full-model/generation loop belongs in T: the witness
supplies explicit steps; G3-M/G3-G retain later orchestration ownership.

Record the merged dependency SHAs, source/header hashes, compiler identity,
actual object/source/depfile manifests, supported host and instrumentation mode
for each future run. Compatibility-host results remain separately labelled.
Do not relabel an ancestor result as execution on a different tree.

## Minimal accepted C2 case set

These are future private-transport transcripts, not calls to installed generation
operations. All numerical roles use their fixed accepted provider and shape.

| Case | Entry and sequence | Success observation |
|---|---|---|
| P1/G0 | One-token prefill, zero budget | Cache length1; owned empty IDs/text, generated length0, unchanged RNG |
| P2/G0 | Two-token prefill, zero budget | Cache length2; same empty-result rules, unchanged RNG |
| P1/G1 | One-token prefill, then one accepted-provider candidate and complete append frame | Initial prompt commit then joint final cache/ID/RNG/output commit; generated length1, cache length2 |
| Manual | Prefill P1, then explicit token input[1,1] | Independent owned last logits[1,256] from each call, cache1→2, no RNG draw |
| Replacement | Existing committed cache, new P1 or P2 prefill | Old cache/binding preserved until candidate commit; new binding replaces it only on success |
| Rejections | Empty prompt, P>2, P2/G1, decode without cache, decode at length2, wrong input kind or shape | Exact accepted category/operation; no cache/RNG/output/model mutation |

Use distinct deterministic model/input fixtures so old cache, replacement prompt
and appended token are distinguishable. For token/text transport include byte0
and byte255, and explicit EOS equality/inequality. A prompt EOS does not stop;
an emitted EOS is included in ID/text/cache after its model append. This does
not authorize a new sampler numerical campaign: eventual candidate/RNG expectations
come from the independently accepted provider contract and artifact. No pending
provider is substituted to execute these cases early.

## Consumer witness matrix

Every row is **pending implementation evidence**. Native instrumentation checks
private bytes/counters only in the reviewed development artifact; no inspection
operation is added to a normal facade.

| ID | Witness | Oracle / invariant |
|---|---|---|
| T-AUTH | Genuine same-aggregate sealed model/tokenizer; copied, dead, wrong-kind, foreign-owner and shape-compatible non-input shells; pending result passed to another context | Registry admission precedes payload access; exact error precedence; no enrollment or mutation on rejection. Release of the exact dead owner alone is idempotent |
| T-GUARD | All38 checked M3T outer entries and eight M3 entries during an admitted G3 call; trusted original M3 schedule outside G3; manual active frame on same versus different model | Shared outer reentry rejects; trusted internal nesting remains legal. Existing model10/native active block only the occupied model between calls. Finish/abort clear only matching acquired tokens |
| T-PINS | Hold canonical14 through role0..20 and through G1 prefill/append; induce valid partial acquisition failures using shared-owner instrumentation | Tied embedding/head has one pin; exact sentinels/control counts; rollback releases only acquired bits in reverse. No retained ordinary I2 borrow shells or gradient plans; all pin state returns to IDLE |
| T-BIND | Change one parameter value bit after a committed call; restore exact original bits; attempt train mode; fresh prefill after an admissible value change | Decode rejects stale binding before staging; exact restoration permits reuse; fresh prefill can replace binding. G3 never changes modes. Parameter bytes, gradient presence/bytes/count/weight and initializer state remain unchanged by G3 |
| T-ROLE | All21 ordinals with T1/T2; repeat, skip, out-of-order, wrong frame kind; foreign pending result; candidate pointer on generated decode, NULL on manual decode | Only exact next role advances readiness. Separate actual-shape scratch and explicit last-logits copy; sampler token[1] is copied to decode[1,1]. No metadata relabel, reverse role, graph capture, fallback or generic provider selection |
| T-CACHE | Layer0 stage at role10, full-capacity A2 K/V/length/keep view, explicit causal mask/positions, then lease close; fail before/after view acquisition | Exact physical K/V geometry and initialized tails; immutable projected learned-position K, no RoPE; all leases end before stage/commit/abort/destroy. Abort scrubs staged tail and preserves committed prefix |
| T-PARITY | Instrumented-only full-prefix reference for P1 and P2, compared with matching cached explicit-step transcript; normal artifact rejects parity kind0/call kind3 | Independent mathematical logits reference plus cached/uncached comparison with documented primitive-appropriate tolerance. Parameter/input bits and shape provenance fixed; repeatability claims separated from tolerance-based parity |
| T-READY | Numeric preparation, ID staging and raw-text acceptance repeated, skipped or ordered incorrectly; foreign child; mismatched raw byte; short staging | Once-only readiness flags reject before writes. No uninitialized reserved ID is accepted as ready. Complete result envelope, empty-ID/RNG/text and cleanup capacity exist before initial prefill commit |
| T-T1 | Baseline raw tokenizer, G0/G1 output decode, byte boundaries and malformed staged ID; repeat many outputs against one reused tokenizer | All IDs validated before first raw write; no T1 shell enrollment, retained lease or allocating general decoder. Public tokenizer-decode remains unchanged. No implicit Unicode or named-special policy |
| T-LIFE | Release output/generator/source RNG in each applicable order; use independent IDs/lengths/cache-lengths/RNG/text copies afterward; active-borrow release | Accessor results remain independent and immutable; release cannot destroy another owner. Busy/borrowed rejection precedes writes; payload roots clear while authenticated tombstones remain |
| T-ERROR | Native error, nested Eshkol failure and cleanup diagnostic clobbering; wrong outer operation normalization; failure in a nested region | Snapshot bounded first-error fields before cleanup; normalize to exact invoking outer operation; no pointer/lease/caught young graph escapes. Pin-end never clears guard; consumer cleanup precedes outer guard exit |
| T-PACKAGE | Actual normal archive/object/facade/dependency inspection and negative private link/import; separate instrumented witness closure | Exactly seven/87/93 public boundary; one registry lineage and shared replacement TU; six selected provider/cache objects; private symbols localized, no test symbols or unresolved private dependency |

Source-private `plan_pins` is not a universal gradient-read lock. Test preserved
gradient state outside the held production transcript; do not add gradient-borrow
calls to G3. Shared-owner invariant/death tests cover illicit control corruption.
The G3 witness proves its closed no-gradient/no-callback execution and authentic
shared-pin consumption rather than claiming stronger I2 exclusion.

Error-delivery allocation is separate from the nonallocating rollback/publication
tail. Do not infer allocation-free Eshkol `raise`, arbitrary error promotion or
flat failure-loop retention from native cleanup. Use the accepted G3 error/outer
operation policy; E3's particular table/sentinel delivery design is not imported
as an unstated G3 seam amendment. Exact error delivery guarantees or missing
witness operation literals must be dispositioned before implementation, not guessed.

## Failure cuts and publication oracle

Capture entry committed cache K/V/length/keep and binding, logical RNG, independent
prior outputs and model/gradient metadata before the call. Fixture observation
storage is accounted separately and must not alter the measured production path.

| Cut | Recoverable failure must leave | Cleanup / publication requirement |
|---|---|---|
| Admission/reservation/pin acquisition | Entry cache/binding/RNG and prior outputs | No result; reverse only acquired state; no token belonging to another call cleared |
| Candidate prefill roles/stage/view/text/preflight, before initial commit | Entry cache/binding/RNG | Close nested views, destroy candidate cache/pending results; G0 empty packaging also obeys this rule |
| G1 after initial prefill commit, before final append commit | New prompt-only cache/binding and pre-call RNG | No partial output; discard candidate/result, abort and scrub appended tail; do not restore the replaced old cache |
| Manual decode before commit | Existing one-token prefix/binding/RNG | Abort/scrub tail; unpublished logits destroyed |
| Final prepared commit and immediate finish | Successful atomic result: matching cache/ID/RNG/output, or manual cache/logits; pins/tokens idle after exactly one finish | No allocation/callback/recoverable branch or recoverable injected failure inside the tail |
| Duplicate/unprepared commit or finish | Accepted rejection before mutation, outside a valid committed tail | No second publication/drain or reused readiness authority |

Enumerate fallible allocations/admissions and legal failure points from the eventual
implementation; record a reached-stage counter so a passing rejection proves it
reached the intended cut. Preallocation does not mean all provider/view work is
infallible. Never inject a recoverable failure inside validated provider invoke or
an accepted nonfailing commit/drain. Deliberately corrupted self/sentinel/count or
other impossible invariants belong to isolated fail-stop tests; they are not
recoverable failures for which successful rollback may be claimed. Preserve the
original recoverable error only while the cleanup invariants remain valid.

For G0 and final token/manual commits, disable allocators immediately before the
validated tail and prove completion plus subsequent owner usability. For abort,
complete the last valid fallible preflight first, then prove the actual drain
without allocation. Report Eshkol and native allocator coverage separately; no
single native switch proves a region/error path allocation-free.

## Evidence and handoff checklist

The future supported gate must identify each reached matrix row and failure cut,
not only print a total pass count. Required artifacts include private Eshkol AOT,
native ASan/UBSan/LSan, normal versus instrumented manifests and appropriate
predecessor regression results. Preserve failed attempts and exact tested/reviewed/
merged SHA/tree distinctions. Common prerequisite proof may be cited, while G3
consumer integration evidence must execute against its own exact artifact.

For the accepted 1,024/8,192-loop measurements, separate:

- repeated operations on one generator with explicit result/clone release;
- repeated construction/release, with all retained identity controls counted;
- bounded repeated recoverable failures at each meaningful transaction cut.

Measure peak live tensors/cache/views/pins, retained native and Eshkol identity
controls/bytes, arena/runtime/error growth and process RSS. Reuse one T1 input
where the test concerns output decoding; record T1 encode retention separately.
No flat total-memory claim follows from C2 capacity or successful release.

The implementation handoff must state exact APIs/contracts, executed commands
and results, unsupported/measured limits, remaining dependencies, and integration
commit/PR. T acceptance still excludes production G3-M/G3-G orchestration,
installed public generation, larger context and G3-R continuation.
