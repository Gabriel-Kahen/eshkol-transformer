# Development roadmap

Status values: `planned`, `active`, `blocked`, `review`, `complete`.

## Parallel execution strategy

Work proceeds in dependency-aware waves. Within a wave, workstreams may run in
parallel in isolated worktrees. Contracts merge before downstream implementation.

The Wave 3 checked-promotion adoption candidate pins
`Gabriel-Kahen/eshkol@81298b4a9608fb92eb6f351a2eabd8392da7d9ef`, proposed
upstream in [PR #714](https://github.com/tsotchke/eshkol/pull/714). Adoption remains
under review. The exact-pin supported toolchain/package rebuild and focused B0/C2/E3
adoption gates pass; the separate #117/#121 guard-order repairs, their combined
aggregate evidence, full supported CI, and final integration review remain pending.
No workstream status changes solely from the pin update.

The first supported full-CI attempt on integration head `c27aaf5` exposed a
shared I2 compiler prerequisite: pinned `81298` tried to transfer into the
unpublished tail-body entry of the private construction rollback helper. The
mirrored I2 roots now keep rollback and abort helpers in ordinary call position
(`c74c267`); their return remains `#t` after the same revocation and cleanup.
On a clean supported f31/LLVM 21 image, strict I2 aggregate compilation passed
in 5m12s (2,903,900 KiB peak RSS, zero swaps), and the exact `build-i2.sh`
package/manifest gate passed in 7m08s (4,645,056 KiB peak RSS, zero swaps).
Full affected-suite CI and combined-head acceptance remain pending.

The draft integration PR #126 passed supported hosted run `36030979305` on
pushed `979bd52`: all 23 jobs, including topology, canonical build, native
numerics/optimizer, D2 shard-loader, model composition, G3N/G3S and all six
C2 groups, succeeded. The preceding `36020537251` run on `8169e68` was also
fully green; its strict C2 joint peaks were 460,464/461,148 KiB below the
unchanged 524,288 KiB cap. Reviewed installed E3 fixed-profile evaluation and
8,192-reservation report-edge proof, plus authentic G3 P1/P2 prompt-to-prefill
binding, were pushed through `8c13bbf`. Supported focused gates pass; fresh
exact-head run `36041612097` completed with 20 substantive jobs passing; its
model-composition job failed only at the stale M3 source pin after E3 facade
source changed. Test-only `5fa2d4e` repins the exact current hash, and all
9 M3 tests and predecessor hashes pass locally. Subsequent pushed `38d9c35`
run `36051266612` passed model-composition, parameter-state and the other
completed substantive jobs, but four jobs failed at stale G3S/G3N/L3S/E3
source hashes or the O2 public-header tree pin. Local test-only `af35b1a`
and `0fdcb8c` repin those exact identities; the predecessor lists and four
focused O2 package tests pass. Canonical-build was cancelled by the failed
aggregate. A new exact-head supported CI run is required before broader
acceptance.
Pushed `a9edb9f` run `36074614278` passed 20 jobs, including model
composition, native numerics/optimizer and all six C2 groups. Its canonical
clean-build job reached the final G3-C4 provider but was cancelled at the
configured 75-minute job limit before the smoke/benchmark step; the two
aggregate jobs therefore failed. The measured log shows no compiler/test
failure in that job. The canonical job budget is raised to 120 minutes for
the next exact-head run; full combined acceptance remains pending.
Pushed `45a05dc` run `36088444454` passed 20 substantive jobs, including
canonical build and model composition. Native numerics failed at the N2
source-closure check because the private TR3 raw-symbol `.txt` fixture matched
the broad source scan; the aggregate jobs consequently failed. Test-only
`65136dd` restricts N2 and N3K closure scans to C source/header files, and
both expected closure fixtures plus shell syntax pass locally. A fresh
exact-head run is needed for acceptance of the combined branch.
The later pushed `ef045da` completed exact-head hosted run `36115680360`
successfully: all 21 jobs passed, including canonical build, native
numerics/optimizer, model composition, parameter state, G3N/G3S, and all C2
groups. This accepts that pushed tree only. Subsequent local P1 indexing and
profile commits require their own exact-head CI after the full P1 package gate.
The private G3 Step19A output-reservation candidate `06b6fd2` was
cherry-picked locally at `ee7325d`. Independent review found two failure-
atomicity gaps: abort could discard a non-idle token frame before a borrowed
output prevented destruction, and logits could alias pending owner/I1 storage.
Focused repair `6e60361`/tree `c613080` is integrated as identical six blobs
at `8056e84`; the same independent reviewer approved both fixes. Supported
f31 normal/repeat/ASan+UBSan+LSan output is identical at 1,473 checks,
including non-idle borrow failure/retry, owner/I1 alias cuts and an I1-borrow
allocation cut. Root verified
`g3c4-output-reservation-fix-6e60361-20260924T182604Z/SHA256SUMS`
(`6440cd34...`); merged static, Q0 4/4 and CI 107/107 pass. This remains
private output capacity and lifetime ownership, not public generation.
The reviewed Step20A private numeric output preparation `ab7ae0a`/tree
`bca1cb1` is integrated as identical patch `c8affc1`. Its one independent
reviewer approved the corrected cross-context, readiness and live-I1-borrow
witnesses. The supported f31 normal, repeat and ASan/UBSan/LSan runs each
passed the same 647 checks; root verified
`g3c4-output-prepare-ab7ae0a/SHA256SUMS` (`803c7d15...`) and reran the
647-check gate on the merged tree, with Q0 Python isolation 4/4. It records
G0/G1 staged numeric IDs, cache length and RNG into a pending output without
publishing it. The next reviewed private Step21A `0a3bf2e`/tree `1dfc829`
is integrated as identical patch `f29811e`. It authenticates the pending
owner and binding, borrows the exact output I1, validates G0/G1 IDs, then
copies G1 as eight explicit little-endian bytes to a disjoint caller carrier
and advances ID readiness once. The supported f31 normal, repeat and
ASan/UBSan/LSan gates pass identically at 655 checks, including alias,
active-borrow, allocation, malformed-length, ownership and retry cuts;
merged Q0 Python isolation passes 4/4. Root's single independent review
approved the code and verified
`g3c4-step21a-merged-f29811e-20260924/SHA256SUMS` (`0151774f...`).
Reviewed Step21B `f4b00c2`/tree `cada86c` is integrated as identical patch
`38e7ec5`. The real T1 registry/core decodes G0/G1 raw bytes in place, and
native private readiness authenticates pending output, binding, I1 and raw
byte before advancing text readiness once. Supported f31 normal, repeat and
ASan/UBSan/LSan evidence passed; root verified the source seal
`g3c4-step21b-f4b00c2-20260924/SHA256SUMS` (`fb36b8f6...`). The candidate's
precommit Q0 run missed its newly tracked checker; post-integration Q0 found
that exact development-script admission gap. Test-only `ce4419c` admits the
checker, and merged Q0 4/4 plus the Step21B checker pass. The private rooted
output-envelope/coordinator Step22 `021ef61`/tree `d27bdc0` is independently
reviewed and integrated with identical source as `23a9dde`. It roots the
pending output in the active call ledger before native reservation and invokes
the accepted two-argument real T1 decode coordinator. Supported f31 normal,
repeat and ASan/UBSan/LSan evidence is byte-identical: 51 real G0/G1 route
checks and 21 allocation-cut checks pass, with inherited Step21B and call-entry
regressions. Root verified the source closure and merged Q0 4/4 plus the
Step22 structural checker. Call finalization deliberately rejects after text
readiness until frame prepare/commit and publication are implemented. Public
result, generation loop and save/reload remain pending.
The reviewed private native manual logits reservation Step23A source
`9e9e628`/tree `f3183fd` is integrated byte-identically as `b3295e1`.
It adds only the documented `logits_reserve` entry: active manual calls own
one pending I2 f32 `[1,256]` result, with typed release, abort tombstone,
borrow rejection and allocation-failure cleanup. Pinned f31 normal, repeat
and ASan/UBSan/LSan passed 1,196 checks with 4 I2 and 1 owner allocation
cuts plus inherited Step22/21B/call-entry regressions. Root verified the
`g3c4-logits-reservation-9e9e628-20260924` source closure
(`a7c54e4...`), byte-identical merged source, Q0 4/4 and structural checker.
The reviewed native manual frame-begin Step23B source `0ecac398`/tree
`d4ec9a9` is integrated byte-identically as `865c1ff`. It admits an exact
prefill T1/T2 or decode-after-one-prefix call only with authentic owned input,
pending logits and idle cache, then copies the IDs into an ordinal-zero
private transcript. Failure preserves caller state; abort releases the frame;
prepare-end and finish reject incomplete work. Pinned f31 normal, repeat and
ASan/UBSan/LSan passed 745 checks with allocation and borrow cuts, inherited
Step23A 1,196 checks and Step22 affected gates. Root verified
`g3c4-step23b-b3295e1/seal.txt`, the exact source closure, merged Q0 4/4
and structural checker. The reviewed Step23C source-private ordinal-zero
token-embedding precursor `af9626a`/tree `31f8d0f` is integrated byte-identically
as `9f614fc`. It uses frame-owned `[1,T,4]` scratch, routes T1/decode through
G3-N and T2 through N3K, advances only after successful dispatch, and preserves
the committed cache, pending logits, binding and RNG on a provider failure.
Pinned f31 normal, repeat and ASan/UBSan/LSan each passed 246 checks, with
inherited Step23B (745) and Step23A (1,196) regressions. Root verified
`g3c4-step23c-cf97072/seal.txt`, its exact 170-source closure, structural
checker and Q0 4/4. The reviewed Step23D source-private pre-A2 block
`c7e32b1`/tree `e84b406` is integrated byte-identically as `016abf4`.
Frame-owned actual-T scratch and the accepted G3-N/N3K/N2 providers execute
ordinals 1–9 individually for T1/T2 prefill and decode, with one dispatch per
call, exact next-ordinal checks, candidate-first output and failure retry.
Pinned f31 normal, repeat and ASan/UBSan/LSan each passed 23,040 checks across
three routes and 27 provider cuts; T1/T2 slots 0–9 match the existing whole
forward bitwise. Root verified `g3c4-step23d-67c579e/seal.txt`, its exact
175-source closure, inherited Step23C/B/A gates, merged structural/Q0 4/4 and
CI topology. Ordinal 10 still needs a capacity-2 candidate A2 transaction;
the remaining 11 roles, manual `role_step`, frame prepare/commit and public
result remain downstream.

The isolated Eshkol F32 runtime line now has a reviewed public `float32?` leaf
at `d960add`/tree `7e0e0fd`: supported f31/LLVM21 focused 11/11 twice, F32
53/53, system O0/O2 23/23, and native/AOT/VM sanitizer gates pass; sealed
evidence is `f32-public-predicate-20260924/SHA256SUMS` (`44b28253...`). A
separate sealed `type-of` parity audit on that exact tree found an accepted
contract ambiguity: public native Scheme reports numeric type tags, the C API
reports interned symbols, and the VM reports strings. Its evidence is
`f32-typeof-parity-blocker-d960add-20260924/SHA256SUMS` (`41b92f67...`). A
sealed read-only decision audit, `f32-typeof-contract-decision-d960add-20260924`
(`SHA256SUMS` `c60eef12...`), maps the complete current value surface and
recommends canonical interned symbols. The orchestrator accepted that public
contract in [TR3_F32_SCALAR_RUNTIME_CONTRACT.md](TR3_F32_SCALAR_RUNTIME_CONTRACT.md);
cross-substrate implementation, migration and parity gates remain pending. An
isolated VM F32 hash-key leaf is reviewed below. Neither leaf establishes full
F32 acceptance or changes the transformer runtime pin.
The first runtime-eligible type-symbol prototype passed exhaustive pointer-based
classification but exposed a C ABI boundary: the legacy by-value
`eshkol_type_of` can erase implicit f32 padding bytes before validation. Its
supported reproducer and independent review are sealed at
`f32-type-symbol-runtime-blocker-d960add-20260924/SHA256SUMS` (`90c6708f...`).
The accepted contract now makes versioned pointer-taking
`eshkol_type_of_ref_v1` authoritative for full-carrier canonicality and records
the legacy wrapper's padding-only limitation. The upstream C implementation is
reviewed below; Scheme/VM parity gates remain pending.
A sealed first-pass successor classifier audit on exact `d960add` indexes
4,628 source sites across 227 files and semantically reviews 36 F32-reachable
sites; `f32-successor-classifier-audit-d960add-20260924/SHA256SUMS`
(`80255c0f...`) has a reproducible source-hash verifier. It found a sanitizer-
reproduced P0 16-byte tagged-cons copy defect, native/VM `type-of` mismatch,
and AOT/VM `conjugate` mismatch; VM hash, native/VM `region-open`, and VM scalar
activation gaps also remain. The reviewed tagged-cons repair is recorded below.
The raw inventory is not an exhaustive semantic classification, so it cannot
support full F32 acceptance.
The isolated VM F32 hash-key leaf `ae12e509`/tree `971a4eaa` now passes
independent focused re-review after adding a 16-entry rehash and nested-region
raw-bit witness. Supported f31 Release poison gates 4/4, real ESKB VM suite
666/666, standalone source suite 77/77 and ASan+UBSan+poison VM gates 2/2 pass;
root verified `f32-vm-hash-keys-c27f85eb-20260924/SHA256SUMS` (`c4cc4c38...`).
It remains isolated on predecessor `c27f85eb`. The reviewed predicate/hash
union `c2df68d`/tree `e043f575` on `d960add` preserves both source patches and
passes supported f31 Release 11/11, ASan+UBSan+poison 5/5, C API 675/675 and
standalone VM 77/77; independent review found no blockers. Root verified
`f32-predicate-hash-union-d960add-20260924/SHA256SUMS` (`c1c4ef59...`). The
broader sanitizer VM smoke hit an inherited signed-shift fault in
`vm_compiler.c:3054`; a separate reviewed repair is described below.
Whole-F32 acceptance remains pending.
The reviewed upstream runtime type-symbol leaf `d251cb0`/tree `07882ae`
implements the accepted public pointer API `eshkol_type_of_ref_v1` in the
runtime archive, covering all declared direct/heap/callable/legacy tags,
canonical and malformed f32 bytes, null/unknown controls and interned-symbol
identity. Supported f31/LLVM21 focused CTests 5/5 and ASan+UBSan+LSan
runtime-only tests 2/2 pass; root verified
`f32-type-symbol-ref-v1-20260924/SHA256SUMS` (`17792a7a...`) and independent
review PASS. Legacy by-value `eshkol_type_of` cannot certify padding bytes 4..7;
first-intern OOM was not injected. Native Scheme/VM `type-of` still return the
old types on the `d251cb0` leaf, and the pre-existing source-only
exhaustive-dispatch gate remains red while its compiled test passes. The
reviewed native Scheme AOT/JIT successor `069efba`/tree `418cf89e` adds a
pointer-result adapter around `eshkol_type_of_ref_v1`, stores the original
16-byte carrier, and returns a typed canonical interned Symbol through direct,
first-class, `apply` and `map` routes. Its single independent review PASS and
pinned f31 Release focused 10/10, type suites 21/21/9/9/70/70, native
sanitizer 6/6 and JIT sanitizer 4/4 pass; root verified
`f32-type-symbol-native-20260924/SHA256SUMS` (`398e3229...`). VM `type-of`
and cross-substrate union gates were pending at that leaf. The reviewed VM
successor `95d86f3`/tree `65bf2df` (following `3cc5a5a` on `069efba`)
returns a per-VM cached, evacuation-rooted canonical Symbol for all declared
VM tags 0–34 plus unknown; undifferentiated source/builtin closures report
`procedure`. Root blocked the first candidate because spelling-allocation
failure could return NIL without a VM error. The same independent reviewer
approved the corrected fatal-error path and deterministic injected failure
witness. Supported f31 Release and ASan+UBSan focused suites pass 4/4 each,
standalone source 77/77 and VM C API 685/685; root verified
`f32-type-symbol-vm-20260924-r2/SHA256SUMS` (`d16d3d83...`). Both type-symbol
leaves are still awaiting cross-substrate union gates; no transformer runtime
pin changed.
The independently reviewed VM callable-kind follow-on `8295121d`/tree
`5f73500` records compiler-authored `lambda-sexpr`, captured `closure`, and
explicit builtin `primitive` kinds, preserving unmarked/legacy callables as
`procedure`. The kind survives ESKB, region evacuation and parallel clone;
source lambdas and captured closures now match native type-of semantics in
focused tests. Supported f31 Release focused 6/6, sanitizer focused 5/5,
and standalone source 77/77 pass; root verified
`f32-type-callable-vm-20260924/SHA256SUMS` (`f1b5a8e1...`) and the one
independent reviewer PASS. Broader `mixed_types_stress` reaches a pre-existing
undefined `compose` lookup after its callable checks; the broader sanitized
introspection path hits an unchanged VM compiler signed shift. The leaf is
isolated pending successor-union composition and does not clear whole-F32.
The separate reviewed tagged-cons transport repair `2904b94`/tree `01b89bd6`
copies all 16 tagged-value bytes through construction, setters/getters, batch
and region escape paths, with overlap-safe self-store. Supported f31 Debug
ASan+UBSan and Release focused CTests each pass 4/4; an immutable baseline with
test-only poisoned arena reuse fails as expected. Independent re-review PASS;
root verified `f32-tagged-cons-transport-20260924/manifest.json`
(`8dadd5c7...`). Its by-value getter byte proof is specific to the pinned ABI.
The three reviewed predicate/hash, C type-symbol, and tagged-cons leaves are
composed in clean isolated Eshkol commit `602876a`/tree `24cc0a1`. Both imported
leaf patch IDs match exactly; the sole independent union review approved the
combined changes. Supported f31/LLVM21 poisoned Release matrix 19/19 and
ASan+UBSan matrix 13/13 pass; direct VM C API 675/675, standalone 77/77 plus
hash-region witness, imported leaf tests and compiled exhaustive dispatch pass
in both builds. Root verified
`f32-runtime-union-c2df68d-20260924/SHA256SUMS` (`8965e26e...`). The
pre-existing source-pattern exhaustive gate and the inherited VM compiler
UBSan signed shift remain separate at this exact union tree; this union is not
full F32 acceptance and does not
change the transformer runtime pin.
The separate reviewed native/VM `region-open` F32 size leaf `4cf63f4`/tree
`774ba8e` promotes canonical F32 like representable DOUBLE in lone and second
size positions, rejects malformed/folded native carriers before handle
mutation, and rejects non-finite or `>=2^64` float sizes on both substrates
before the previously undefined integer conversion. Supported f31 Release and
ASan+UBSan focused gates pass 2/2 each, with VM source 77/77; root verified
`f32-region-open-parity-20260924/SHA256SUMS` (`28f14c97...`) and approved the
single focused review. VM bookkeeping handles do not retain accepted size
hints, so helper and successful dispatch are separate witnesses. This remains
isolated and does not establish full F32 acceptance.
The separate reviewed VM scalar activation leaf `e6eba88`/tree `45f8a8f`
accepts canonical F32 at scalar FIDs 462–468, returns the same result kind as
binary64, rejects folded unknown tags, and canonicalizes promoted NaNs to the
accepted binary64 quiet-NaN bits. Supported f31 Release focused gates pass
3/3, ASan+UBSan 2/2, direct VM C API 734/734 and standalone source 77/77 in
both builds; root verified
`f32-vm-scalar-activations-d960add-20260924/SHA256SUMS` (`490c9feb...`)
and approved the single focused review. The existing scalar fallbacks in
FIDs 463 and 466–468 remain; this isolated leaf is not full F32 acceptance.
The reviewed isolated unity-VM literal-pack repair `efb6ba0`/tree `146402c`
uses unsigned 64-bit shifts and byte-preserving constant storage for the
string, record-name, helper-symbol and ordinary quoted-symbol pack paths;
FID 100/101 unpack uses logical unsigned shifts. The single independent
reviewer blocked the first candidate for an omitted quoted-symbol path and
approved the amended commit after replaying its high-eighth-byte UBSan
reproducer with exact output and no finding. Supported f31 Release and
ASan+UBSan+LSan standalone source gates each pass 80/80 plus CTest 1/1;
root verified `f32-vm-string-pack-ubsan-efb6ba0d-20260924/SHA256SUMS`
(`1b8cc21d...`). The excluded legacy standalone compiler is outside this
unity-VM repair, and the leaf has not yet joined the runtime union.
The bounded read-only classifier audit on immutable `d960add` exhaustively
disposes 227 raw indexed sites in `runtime_regions.cpp` and
`runtime_arena_core.cpp` as 224 exact groups: two accepted, 200 unreachable
and 22 defect rows in four groups. Supported f31 sanitizer reproduction proves
that mixed-root evacuation at `runtime_regions.cpp:1960-1961` loses canonical
and malformed F32 carrier bytes while a regional pointer promotes; lone
controls pass. The same by-value `evac_value`/`evac_assign` chain reaches 16
nested tagged-slot sites, so they too require full-byte staging and witnesses.
`region-close` direct struct assignments lack a portable
full-carrier guarantee even though the pinned poisoned canonical witness
passes. The baseline `region-open` defect is separately repaired by reviewed
`4cf63f4`. Root verified
`f32-region-classifier-audit-d960add-20260924/SHA256SUMS` (`81252613...`);
a separate byte-preservation repair is active. This covers only the two
indexed source files, not the whole-runtime exhaustive classifier requirement.
The separate reviewed full-carrier region repair `da9ce5e`/tree `ac497df`
replaces by-value evacuation with in-place tagged roots and bytewise nested
slot/publication copies, including one/many-value `region-close`. Supported
f31 Release and ASan+UBSan focused region/promotion gates pass 3/3 each;
canonical and malformed mixed-root, nested-cons and close witnesses pass.
Root independently verified
`f32-region-full-carrier-20260924/SHA256SUMS` (`d1e6ef67...`) and approved
the single focused review. A broader Release emergency stable-identity test
fails identically on exact parent and candidate outside the changed region
path; the baseline and candidate logs are sealed. Complete nested-container
coverage, union gates and whole-F32 acceptance remain pending.
An isolated provisional successor union `8b872aff`/tree `68ebef7` composes
seven reviewed leaves onto `602876a`, including type symbols, region-open,
scalar activations, conjugate, VM signed-shift, and full-carrier evacuation.
Root independently checked the clean tree, source/evidence seals, conflict
resolutions and integration-only test changes. Supported f31 Release passes
F32 59/59, focused 7/7 and standalone VM 80/80; sanitizer passes F32 59/59,
LSan-on focused 11/11 and standalone VM 80/80. The sealed evidence is
`f32-runtime-successor-union-8b872aff-20260924/SHA256SUMS` (`fd71d433...`).
This is a composition acceptance, not a runtime repin or whole-F32 acceptance:
VM callable subtypes still collapse to `procedure` where native reports
`lambda-sexpr`/`closure`; exhaustive nested-container and operation matrices
also remain open.
The reviewed pair/vector transport witness was composed onto this successor as
`9999087d`/tree `035abe7`; the reviewed callable-kind leaf was then composed
as `2cab6a0c`/tree `85c723b`, so compiler-authored VM closures now retain
their native callable subtype in this provisional union. Root verified both
sealed compositions, clean source and supported f31 Release and
ASan+UBSan+LSan gates. On the callable composition, direct VM C API passes
815/815, focused gates 7/7 and standalone source tests 80/80 in both modes,
without sanitizer diagnostics. Evidence:
`f32-vm-transport-union-compose-9999087d-20260924/SHA256SUMS`
(`bb994fd...`) and
`f32-vm-callable-union-compose-2cab6a0c-20260924/SHA256SUMS`
(`8a8b899c...`). Unary operation composition and whole-F32 acceptance remain
pending; the transformer runtime pin is unchanged.
The separately reviewed test-only direct VM pair/vector transport witness
`27c2ed84`/tree `8fe4a27` preserves eight exact F32 bit patterns through
real container storage, regional evacuation and the public raw-bit inspector.
Wrong-tag, wrong-container and invalid-index controls pass. Supported f31
Release and ASan+UBSan+LSan focused runs each pass 733/733; root verified
`f32-vm-transport-witness-27c2ed84-20260924/SHA256SUMS` (`b81ba2e8...`).
It changes no production API; its reviewed composition is recorded above.
Closure/upvalue, continuation, exception and hash transport are still
unwitnessed.
The sealed read-only operation inventory on reviewed union `602876a` lists
every source-visible DOUBLE spelling, AOT/JIT and VM routes, result-kind and
missing witness. It identifies concrete gaps: unary `+`/`*`/min/max can leak
F32, direct VM unary `/` returns its input, VM `sign` and `numerator` silently
interpret F32 as zero, native/VM rational operations differ, and some native
elementary spellings lack VM routes. Root verified
`f32-operation-matrix-602876af-20260924/SHA256SUMS` (`fea57bf3...`) and
reran its exact-source query. A bounded unary route repair is active; the
inventory is not operation acceptance evidence. The reviewed scalar
activation leaf exposes an existing VM scalar path but lacks native public
parity, so it does not close this matrix.
Root independently approved the isolated unary route leaf `be721d90`/tree
`3c27466`: native direct/first-class `+`, `*`, `/`, direct unary min/max, and
VM direct unary `+`, `*`, `/` now promote canonical F32 to the established
DOUBLE result, with malformed/folded native rejects and non-F32 controls.
Supported f31 focused Release and ASan+UBSan+LSan gates pass 6/6 each;
`first_class_variadics_vm_smoke` passes in Release. Root verified
`f32-unary-routes-be721d90-20260924/SHA256SUMS` (`fb2edd52...`).
The reviewed unary leaf is composed onto the provisional F32 successor as
`1240e906`/tree `4a58de1`. The first composed test run found three DOUBLE
baseline controls supplying 2.5 while expecting unary results for 2.0; the
narrow correction changes only those selectors, not production code. Root
verified `f32-unary-union-1240e906-final-20260924/SHA256SUMS`
(`0156a261...`) and pinned f31 Release and ASan+UBSan+LSan: direct VM C API
916/916, standalone source 80/80 and focused AOT/JIT/VM 6/6 in both modes.
Build-time AOT generation initially exposed pre-existing frontend AST leaks;
final test execution used leak detection with the repository's scoped
suppression file and produced no sanitizer finding. Public VM unary min/max
still leak through generic CALL arity handling, and the known VM compiler
signed-shift repair remains a separate reviewed leaf. This composition is
not gap-5 or whole-F32 acceptance and does not change the transformer pin.
The reviewed VM closure-arity leaf `e496bec5` checks fixed and dotted
minimums at CALL, TAIL_CALL and the native callback. Its first composition
`2f145974` onto `1240e906` regressed legal fixed 256-parameter calls and
unary DOUBLE `min`/`max`. The isolated successors `c4769a96` and `4abe64a1`
repair both with source/ESKB 255/256/512 and unary controls; their individual
seals are `f32-unary-arity-width-20260924/SEAL.sha256` (`3866e3c...`
manifest) and `f32-unary-minmax-20260924/SEAL.sha256` (`c6960ef...`
manifest). The separately reviewed VM F32 unary min/max leaf `af2f4e79`
returns DOUBLE from direct and stored first-class calls with signed-zero and
canonical NaN controls; the VM `sign`/`numerator` leaf `02b02539` promotes
F32 for `sign` and explicitly rejects F32 `numerator` instead of silently
returning zero. Root composed them onto the compatible arity successor as
`4b0fd02a`/tree `01f3ee0`: pinned f31/LLVM 21 Release and
ASan+UBSan+LSan affected suites each pass 14/14, including native unary
AOT/JIT, VM source/ESKB arity, type, first-class calls and prelude freshness.
Root verified the sealed composition at
`f32-unary-arity-minmax-union-a55f58c8/SHA256SUMS` (`0c14296...`). Native
and VM DOUBLE `numerator` result kinds disagree, native `sign` is absent,
generated record callable metadata and ESKB v1 named high-arity export remain
unresolved. The switch fallback compiles but lacks supported dynamic proof.
This is a provisional runtime successor, not whole-F32 acceptance or a
transformer pin change.
The separate reviewed native AOT/JIT and VM `conjugate` leaf
`e1394ee`/tree `66e4fa7` promotes canonical F32 to the existing DOUBLE
result kind, rejects malformed/folded native carriers, preserves VM INT/FLOAT
scalar identity and retains complex conjugation. Root's single independent
review blocked an initial VM scalar-kind mismatch, then approved the revised
leaf after DOUBLE/INT baseline witnesses. Supported f31 Release and
ASan+UBSan+LSan focused gates each pass 6/6; root verified
`f32-conjugate-e1394ee5-final-20260924/SHA256SUMS` (`e0507204...`). The leaf
has not yet been composed with the runtime union or accepted as full F32.
The sealed generic E3 dependency audit on exact `9bbb3ff`,
`e3-generic-dependency-audit-9bbb3ff-20260924T184109Z/evidence-files.sha256`
(`49195ba6...`), confirms the installed diagnostic facade and private E3
selected-bit seam do not supply A0 `trainer-evaluate!` or `metrics-ref`.
True-f32 runtime acceptance/repin is the first hard dependency; then TR3 metrics
authority, public trainer creation, evaluation projection and checkpoint
admission follow. The [TR3 metrics proposal](TR3_PUBLIC_METRICS_PROPOSAL.md)
has been reconciled locally with the integrated 22-slot lease and implemented
E3 seam at `8651c94`; independent review approved the exact source/doc facts,
sealed at `tr3-metrics-doc-review-692b0a9-20260924T184736Z/evidence-files.sha256`
(`5ef8df83...`). This remains a design and dependency correction, not a public
metrics implementation.

## Wave 0 — contracts and verification foundation

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| F0 | Package layout, build/test entry points, CI matrix | — | [Accepted CI-E2 full-coverage engine](CI_EFFICIENCY.md): all 16 suites and 23 commands, separate native optimizer, six disjoint C2 groups, canonical build/smoke/benchmark, audited prerequisite plans, strict original-evidence reuse, supported exact-tree CI 35407378830, independent approval, PR #82 merge, and successful 18-second main reuse | complete |
| A0 | Public API, shapes, dtype/device, error and ownership contracts | — | Reviewed specification and compile-only API fixtures | complete |
| R0 | Audit Eshkol tensor/autodiff/runtime capabilities | — | Executable capability probe and gap report with no inferred support | complete |
| Q0 | Test harness and frozen reference-oracle format | — | [Deterministic harness, frozen fixture, and passing compiled parity](Q0_VALIDATION.md) | complete |
| B0 | Benchmark and memory-measurement harness | F0 | [Versioned/checksummed definition, report schema, and smoke benchmark](BENCHMARK_FORMAT.md) | complete |

## Wave 1 — independent foundations

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| E1 | [Shared structured-error construction and accessors](ERROR_CONTRACT.md) | A0, Q0 | 112 construction/accessor/ownership/forgery/negative checks, repeated AOT/JIT parity, and declared-public-boundary gates | complete |
| E1B | [Separately compiled fixed-arity raise-only consumer boundary](E1B_CONSUMER_BOUNDARY.md) | E1 | Fresh-cache mixed-facade AOT identity plus closure, arity, localization, exact defined/undefined-symbol allowlists, determinism, and artifact evidence | complete |
| P1 | [Named parameter tree, buffers, modules, state dictionaries](P1_MODULE_STATE.md) | A0, R0, Q0, E1, E1B | One 24-definition/18-export registry-owning package, 419 structural/ownership/atomicity checks, 405 native identity/failpoint/cross-role checks, 169 registry/error-mapping/topology/name-bound checks, exact manifests, import-both/fresh-cache negatives, sanitizers, deterministic AOT, and supported CI | complete |
| T1 | [Byte tokenizer, special tokens, fingerprints and format](TOKENIZER_FORMAT.md) | A0, Q0, E1/E1B, I1, X1, P1, D1, C1 | Exhaustive byte/UTF-8 and special-token semantics, canonical-format/fingerprint negatives, all-registry retention/complexity evidence, exact-max and exact/one-over optional-header public-AOT RSS/time checks, exact-i64 shell lifetime and sanitizers, C1 policy/atomic-I/O checks, exact 47-global aggregate boundary, deterministic fresh-AOT evidence, production Python isolation, independent approval, supported CI, and merge-commit retest | complete |
| D1 | [Versioned token-shard format and corpus writer](TOKEN_SHARD_FORMAT.md) | A0, Q0, E1 | Deterministic/reference/corruption/boundary groups; checked partial-write/ENOSPC/EIO/close-failure cleanup with no visible manifest; summary mutation/vector-copy/constructor/receiver-forgery; fresh-cache source/object/AOT plus symbol/depfile/crafted-link public-boundary gates; and supported CI | complete |
| K1 | Native kernel ABI/capability layer | A0, R0, Q0 | [Versioned ABI, canonical unverified baseline, and 596 conformance/unsupported/malformed-call checks](K1_KERNEL_ABI.md) | complete |
| I1 | Exact signed-i64 dense CPU tensor container and bounded K1 storage-copy provider | A0, R0, Q0, E1, K1 | [ABI 1.0 ownership/layout contract, exact boundary round trips, malformed/failure-atomic checks, sanitizers, and canonical-pin AOT interop](I1_I64_TENSOR.md) | complete |
| C1 | [Versioned checkpoint container and atomic I/O](CHECKPOINT_FORMAT.md) | A0, P1, Q0, E1/E1B | 245 logical-state/ownership/error checks, 1012 bounded adversarial parser/validator cases, deterministic repeated AOT bytes, native failpoint/ABI/sanitizer gates, atomic old-or-new publication evidence, independent approval, supported CI, and merge-commit retest | complete |
| X1 | [Declarative configuration and resolved-run manifests](CONFIG_FORMAT.md) | A0, Q0, E1, E1B | 111 native semantics, 11 reference/isolation checks, source-before-overlay admission, deterministic fresh object/AOT builds, both E1 import orders, exact 12-export/95-undefined artifact admission, private-source/symbol leakage negatives, and supported CI | complete |

## Wave 2 — model and training primitives

The accepted [CI-E2 follow-up](CI_EFFICIENCY.md#ci-e2--native-critical-path-and-main-push-reuse)
is tracked separately in issue #81. It changes CI scheduling/evidence selection,
not the accepted Wave 2 runtime scope, and starts no Wave 3 task.

The original Wave 2 component contracts below are accepted. The bounded
[C2/P1 handler-order correction #121](C2_HANDLER_ORDER.md) is active. Its final
runtime allocation witness and focused owner/LOAD gates pass; transformer commit
`beb5821` is frozen and independently approved. Pin adoption, the full affected-gate
union and supported integration remain pending.
Wave 3 resumed on September 19 at
04:30 EDT by user direction; M3T and bounded M3 model composition
are accepted. Downstream evaluation, generation and training require the remaining
composition contracts. C2 completion does not prove a live trainer trajectory or
generation, and Wave 3 remains incomplete.

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| P1L | [Release-capable provider/state ownership correction](P1_MODULE_STATE.md) | P1, E1B, C1, T1 | Provider 2.0 exact-once clone ownership, explicit idempotent state release, scoped read-only state-backed handles, callback-defect/failure cleanup, exact P1/C1/T1 aggregate manifests, sanitizers, deterministic fresh-AOT negatives, and supported CI | complete |
| I2 | Shared dense CPU-f32 tensor, P1 value/gradient, and atomic-mutation substrate | P1L, K1, Q0, E1/E1B, R0 | [ABI 1.0 exact-bit storage, borrowed K1 views, P1-bound accumulated gradients, whole-batch preflight/commit, sanitizers, packaging negatives, and deterministic AOT evidence](I2_F32_TENSOR.md) | complete |
| K2 | [Process-local A0 capability facade over exact K1/I2 discovery](K2_CAPABILITY_FACADE.md) | A0, K1, E1/E1B, I2 | Twelve unchanged A0 operations; exact 11-row audited report; 59-global/53-export aggregate; independent approval, supported blocking/exhaustive CI, identical-tree merge and focused merged-main retest | complete |
| T2 | [Deterministic BPE training and streaming encode/decode](BPE_TOKENIZER_FORMAT.md) | T1, D1 | Canonical repeated artifacts and order/partition-invariant merges; all-byte/raw/strict/F0-F4/special whole-stream parity; exact 65,536-byte/73,728-ID ceilings including 73,728 one-ID chunks and one-over negatives; compiled parser/D1 corrupt-data negatives; deterministic localized object/archive/evidence/AOT boundary suite; exact aggregate manifests; production Python isolation; independent review; supported CI; and merged-main retest | complete |
| D2 | [Memory-bounded shard loader, batching, packing and cursor state](D2_SHARD_LOADER.md) | D1, T1, Q0 | Accepted ten-key config and 11-operation surface; exact CPU `17*N*T` carrier/lifetime; packed/unpacked shift and bool masks; unbiased bounded-window shuffle; `ESHKDCU1` exact resume; corrupt/resource/sanitizer/AOT/58-global/52-export gates; independent D2-R, supported CI, identical-tree merge, and merged-main retest. After hosted PR #126 run `36009739388` passed the semantic/resource gates but tripped the non-diagnostic raw-byte oracle scan, the test-only isolation checker now bounds raw dependency markers, preserves source/symbol/link checks, and reports the artifact, channel, and match. Four focused mutation checks and all exact rebuilt delivered artifacts pass; hosted PR #126 run `36020537251` on pushed `8169e68` passes the full shard-loader suite with the hardened isolation check | complete |
| N2 | [Embedding, linear, normalization, activations, dropout, residuals](N2_PRIMITIVES.md) | P1L, K1, Q0, I1, I2 | Exact-row carrier-backed forward/VJP parity, scaled numerical gradients, Philox bit determinism, failure atomicity, native lifetime, private AOT, ABI/isolation manifests, sanitizers, independent review, and supported CI | complete |
| A2 | Causal attention, masks, RoPE and KV-cache primitives | P1, K1, Q0 | Masking, forward/backward and cache parity tests | complete |
| L2 | [Fused indexed token cross-entropy](L2_INDEXED_CROSS_ENTROPY.md) | K1, Q0 | Explicit carrier-neutral K1 provider, stable per-token f32 loss, direct-backward/oracle/finite-difference parity, adversarial failure atomicity, deterministic Eshkol AOT, sanitizer and isolation gates | complete |
| O2 | [AdamW, parameter groups, clipping, accumulation and schedules](O2_OPTIMIZER.md) | P1L, Q0, I2 | Exact 1,365-parameter/group boundary; accumulated-gradient, clipping, AdamW and schedule parity; atomic step/load; releasable logical-state continuation; exact 53-global/six-wrapper aggregate; 6,201 adversarial checks, sanitizers, independent review, supported blocking/exhaustive CI, identical-tree merge and bounded merged-main retest. Reviewed test-only repair `2a9f368` makes the TR3-O defined-symbol query portable to supported Ubuntu 22.04/GNU nm 2.38 without changing assertions or runtime scope; four focused package tests and the full TR3-O gate pass on the supported host. After accepted private A2 Step12A changed the header tree, hosted run `35999574298` reached and passed the long O2 build but failed only the stale header-tree test pin; test-only `ecbd0a6` repins the exact new tree and all four focused O2 package tests pass locally. Fresh hosted O2 rerun is pending | complete |
| C2 | [Detached full training-state checkpoint schema](C2_TRAINING_STATE.md) | C1, D2, O2, X1, K2 | Accepted C2 1.0 component continuation and atomic-file publication: exact 81-global/75-export/81-string boundary, authenticated carriers, exact-bit LOAD/SAVE ownership, failure atomicity, deterministic packaging, flat 1,024/8,192 root retention, and final joint 64-tensor runs at 521,500/521,576 KiB below 524,288 KiB. PR #79 merged as `cbd0929`; PR #77 merged to main as `913cdf4097db09d6c33769e9b0968c01ce0e1f55`; reviewed, tested, and merged states share tree `512a3355cea79583d73b49f690a46478fb1c772c`. Supported run 35393213200 passed all 15 suites/23 commands plus smoke/benchmark, and acceptance 35401088338 reused the verified exact tree successfully. TR3 retains joint live restore/full trajectory and G3 generation proof; [issue #121 handler-order correction](C2_HANDLER_ORDER.md) is in progress: source-order/mutation tests, the exact 5/6/11 final-runtime allocation witness, and focused owner/LOAD gates pass; transformer commit `beb5821` is frozen and independently approved. The reviewed handler-order source is present in the Wave 3 integration branch. On exact clean `7ae5c37` with pinned `81298`/f31, fresh supported P1 handler/LSan, C1 native/sanitizer/AOT, four-image C2 SAVE and canonical C2 public/package/retention gates passed; public retention was 6,356,992 bytes at both 1,024 and 8,192 iterations, slope zero. Sealed public evidence is `c2-121-public-retry-root-7ae5c37-20260924T074300Z` (`SHA256SUMS` `875c3717...`). The scoped exact-7ae5c37 operational gate failed its strict <524,288 KiB peak target at 527,828 KiB; hosted PR #126 corroborated an old-head joint peak of 525,980 KiB. The 1,024/8,192 public-retention slope and production C2 bytes stayed unchanged. A test-only dedicated joint AOT witness preserved the same 26 assertions and 33-file source closure but still peaked at 524,952 KiB, 664 KiB over the unchanged cap; it was not integrated. An isolated byte-preserving C1 SHA-256 streaming repair `daeac8b` avoids copying each complete checkpoint input block and was integrated as the identical source blob at `cfb878e`. Fourteen supported known-answer/boundary cases, canonical four-image SAVE, fresh public/package/retention, and the strict joint operational gate pass. The two joint AOT peaks were 463,836 and 463,692 KiB, leaving 60,452 and 60,596 KiB under the unchanged 524,288 KiB ceiling; both 26-check runs preserved exact bytes and ownership counters. The isolated joint evidence is sealed at `c2-sha-streaming-joint-daeac8b-20260924T100039Z` (`SHA256SUMS` `3974ae27...`). Hosted PR #126 run `35989961605` on pushed `dbc4112` now passes the full C2 operational suite after the shared D2 repair: two joint runs peaked 460,520/460,576 KiB, 63,768/63,712 KiB under the unchanged 524,288 KiB cap, and file/metadata/tensor/count/corrupt, exact/one-over, admission-failure, strict Clang/GCC/C++, sanitizer and byte-lifecycle gates passed. C2 public, SAVE, LOAD, format and state suites in the same run are green. Hosted PR #126 run `35999574298` on pushed `a914540` also passes all six C2 matrix groups, including the strict operational suite; its aggregate still fails only separately scoped A2/O2 exact test pins now repaired locally. Hosted PR #126 run `36009739388` on pushed `46d5c31` also passes C2 SAVE/LOAD/format/state and the strict operational suite; its two joint peaks are 460,776/460,800 KiB, below the unchanged 524,288 KiB cap. Fresh hosted PR #126 run `36020537251` on pushed `8169e68` passes all six C2 matrix groups again, including strict operational joint peaks 460,464/461,148 KiB below 524,288 KiB; D2 shard-loader and the prior symbol-manifest failure paths also pass. All 23 jobs in that supported run completed successfully. Newer supported run `36030979305` on `979bd52` passed all 23 jobs, including all six C2 groups; the pushed `8c13bbf` E3 facade/G3 binding union run `36041612097` is queued, so final #121 acceptance remains pending its exact-head result | active |

## Wave 3 — first complete language model

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| N3K | [Bounded diagnostic numerical primitives and initializer](N3K_PRIMITIVES.md) | N2, A2, K1, I1, I2, Q0 | 31 exact operation/row pairs; independent VJP/bit/negative/carrier tests, private AOT and sanitizers; independent N3K-R approval, supported CI 34222643674, PR #67 merge and focused merged-main retest | complete |
| M3 | [Fixed-profile decoder-only model composition](M3_MODEL.md) | M3T, N2, A2, P1 | #85; bounded immutable-output/graph and 14-parameter contribution contract; independent M3-R approval 5747957422, supported full CI 35465971406 (18 suites / 25 commands), PR #88 merge `aa9e78f` at reviewed tree `7d81855be5ad35a44e74a708642668f8d789ebff`, verified main evidence reuse and focused merged-main native/oracle/package/public-AOT/lifetime checks. [Issue #117 guard-order follow-up](M3_CALL_GUARD_ORDER.md) is frozen at `dda768b` / tree `b8accc6`; its final runtime-`81298b4a` manifest-bound allocation-failure witness passed and `/root/c4_final_review` approved the implementation/test/docs aggregates. Runtime-pin adoption and full integration CI remain pending. [Retention evidence](M3_RETENTION_GATE.md) preserves the original timeout and cumulative costs; no full-training or flat-memory claim | active |
| M3T | [Bounded production Eshkol transport/lifetime contract](M3T_TRANSPORT_PROPOSAL.md) | N3K, N2, A2, I1, I2, P1L, E1B | #68; [accepted implementation and evidence](M3T_TRANSPORT.md); restricted 85-global sibling; independent M3T-R approval, supported full CI 35441033357, identical-tree PR #84 merge `f501609`, and focused merged-main retest; no model-completion claim | complete |
| SHARED-R2 | [Bounded I2/K2/C2 persistence admission](SHARED_R2_PERSISTENCE.md) | I2, K2, C2, M3 | #114; source-approved candidate `8047cec` keeps K1 ABI 1.0 and C2 bytes, adds only the five exact rank-two storage-copy shapes, exact K2 live descriptor/report audits, installed-facade M3 public load/save/release, and a C4 schema witness. Independent source/gate review approved tree `e8093079`; full I2/K2/C2 owner gates pass on the unsupported LLVM 22 compatibility host, including flat 5,832,704-byte public C2 retention at 1,024/8,192 iterations. Accepted through [PR #124](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/124), merge `f74fede`, after supported full CI `35925066612` passed all 19 suites/34 commands; the merge preserves the exact tested tree and main run `35933002462` verified evidence reuse | complete |
| L3S | [Bounded masked-objective reduction and explicit seeds](L3S_MASKED_OBJECTIVE.md) | L2, K1, Q0; I1/I2 for evidence | #87; accepted six-operation CPU-f32 `[1,2]` contract; independent L3S-R approval 5770338523, supported full CI 35531132634 (18 suites / 26 commands), PR #91 merge `fe249316`; only intervening accepted G3 Markdown differs from the tested candidate; focused merged-main gate and leak checks passed; fresh combined-main CI 35680925863 passed all 18 suites / 26 commands at exact merge/tree `fe249316`/`902d4645`. Reviewed repair `a6d8851` admits the exact TR3-B consumer after supported integration run 35916891673 exposed only the missing inventory row; the repaired package audit passed under Clang 21.1.8 and the pinned full L3S gate passed on the unsupported LLVM 22 compatibility host | complete |
| E3 | [Loss, perplexity, token accuracy and validation runner](E3_EVALUATION_PROPOSAL.md) | M3/M3T, D2, L2/L3S; shared guard and bool metrics accepted; private P1/D2 source staged, root union pending | #99; the shared-call implementation, E3-METRICS and E3-D2 prerequisite union is accepted at merged `33a54ef` after supported 19-suite/29-command CI and focused merged-head checks. The exact nine-file deterministic reference/oracle toolkit from `37a8078` remains development-only. The exact five-file bounded native-frame leaf from `38fb989` is independently source-approved; root authenticated its supported normal and ASan/UBSan/LSan runs, each reporting 4,381 checks. The independently approved six-file parity leaf from `e086a61` has supported five-case normal/sanitizer/repeat evidence across all 21 roles and strict metrics. The exact native frame/parity integration is accepted and merged through PR #124 as `f74fede`, tree `64ac6cdc`, after independent review and supported CI `35925066612` passed all 19 suites/34 commands. Main run `35933002462` verified reuse of that exact-tree evidence. The reviewed closed private transaction `6b12422` plus authority fix `aa0adb`, private package `d876369`, and E3-owned selected-metric-bit accessor `38e43d0` are source-staged in the integration branch. The package leaf passed authenticated f31/81298 private package integrity, five genuine-corpus packed/unpacked poisoned-runtime cases with D2 cursor restoration, and native frame normal/sanitizer 4,381-check gates. The accessor leaf and root-composed native normal/sanitizer rerun each passed 4,434 checks. The accessor package relink and root combined E3/CLI3 package and CI gates, public evaluator API, runtime-pin adoption, guard integration and full evaluator acceptance remain pending. An isolated P1 active-enrollment-chain candidate `5baedaf` passed supported private build/package and 1,024 successes in 387.37 seconds, but functional 2,048 successes took 1,791.55 seconds (2.312x normalized rate), exceeding the predeclared 1.25x near-linear guard; 4,096/8,192 were not run. Sealed evidence is `e3-diagnostic-scaling-5baedaf-f31-d5c23f-20260924T085811Z` (`index.sha256` `c84451f0...`). A sealed 60-sample read-only profile found 50 `m3_call_foreign_storage` stacks, 49 `ranges_overlap`, 40 M3 pin checks, and zero P1 E3 mode stacks (`e3-diagnostic-profile-5baedaf-20260924T095328Z`, `index.sha256` `5246b29d...`). The independently reviewed shared retired-control AVL plus malformed-index guard and exact 73-slot P1 active-enrollment composition are preserved on clean isolated `375d6f3`/tree `deefd7b`. Its supported full P1 gate passed under the exact 900-second per-compile bound (52m48.78s, 3,900,048 KiB peak RSS); sealed `e3-p1-compose-375d6f3-p1-900-20260924T105958Z/SHA256SUMS` is root-verified. The fresh private E3 package passed single success/EOS, success horizons 1,024/2,048/4,096/8,192, failure horizons 1,024/8,192 and the 8,193 cap. Observed normalized success-rate maximum was 1.019469 under the predeclared 1.25 guard; 8,192 success took 190.60 seconds and 504,908 KiB RSS. Sealed `e3-diagnostic-scaling-375d6f3-f31-d5c23f-20260924T115354Z/evidence-files.sha256` is root-verified. Arena deltas are total only, with attribution among reachable reports, unreachable staging and inherited tombstones unmeasured. The selective 3-commit current-root replay `cc78efd` → `8a85118e` → `4b7f9c4` was independently source-approved and passed supported f31/LLVM21 network-none G3N/G3S plus complete E3 base/package/runtime on that exact tree: success 1,024/2,048/4,096/8,192, failure 1,024/8,192, EOS, cursor/key/foreign negatives and the 8,193rd cap. The 242-file seal `e3-current-root-supported-4b7f9c4-20260924T131728Z/evidence-files.sha256` (`b694c670...`) is root-verified. Exact commits are integrated onto newer private G3 Step15 root as `ca21861` → `d9e4d56` → `d8632ae`; local 15 G3 static, 11 E3 source/package, 8 O2/Q0, 107 CI/topology and G3N/G3S/N3K hash manifests pass. The exact Step15 affected union on clean detached `d8632ae2`/tree `b7f8ea97` passed supported f31/LLVM21 network-none K1/I1/I2/N2/N3K/A2/G3N/G3S and full E3 base/diagnostic package/runtime. G3N passed in 17.73 seconds/137,648 KiB, G3S in 17.88 seconds/141,744 KiB, and the E3 union in 1,432.24 seconds/5,210,992 KiB. Actual E3 success/EOS, cursor/key/foreign negatives, success horizons 1,024/2,048/4,096/8,192, failure horizons 1,024/8,192 and embedded 8,193 cap all passed. Success arena deltas were 48,807,032/88,931,448/169,180,280/329,677,944 bytes; failure deltas 48,975,408/331,681,328 bytes. Root verified all 233 files under `e3-step15-supported-d8632ae-20260924T135605Z/evidence-files.sha256` (index SHA-256 `461869670fc5dc662edefd8af267d717329a97b6d424351393ec96d28d0a0973`). The independently reviewed three-commit retention-attribution leaf `8c6c628` → `b88387a` → `6919dd1` is integrated as identical source/test blobs at `2b9f961` → `150ed1b` → `1a90eb8`. Its exact f31/LLVM21 network-none supported gate passed static 15/15, base/diagnostic package and runtime, EOS/key negatives, success horizons through 8,192, failure 1,024/8,192, poisoned runtime and 8,193 cap in 1,373.46 seconds/5,222,484 KiB. Per call, success retains 224 bytes of reachable report authority and 64 bytes of unreachable envelope; failure leaves the full 288-byte promoted graph unreachable. The inherited post-reservation transaction bucket is 2,240 bytes per call. At 8,192, success measured reachable 1,835,008, unreachable staging 524,288 and inherited transaction 18,350,080 bytes; failure measured reachable 0, unreachable staging 2,359,296 and inherited transaction 18,350,080 bytes. The outer witness remainder is measured separately. Root verified all 171 files in `e3-retention-attribution-6919dd1-20260924T151831Z/evidence-files.sha256` (`ed53012d...`). The inherited bucket includes E3/P1/I2 controls and failure-side error retention, without object-level tombstone tracing. This remains test-only/source-private. The independently reviewed bounded installed facade `2ec331c`/tree `cdbd806` is integrated as identical source/package/test blobs at `22c35f2`. Its supported f31/LLVM21 network-none base, private-diagnostic and installed-public packages pass, with exact 101 exports/107 globals/168 undefined symbols, 30 Eshkol/57 native sources and nine facades. Public `transformer.evaluation` exposes only `diagnostic-evaluate-fixed!`, `diagnostic-evaluation-f32-bits` and `diagnostic-evaluation-count`; a genuine model/tokenizer/dataset caller measured loss/mask/perplexity/accuracy bits 1085366897/1084227584/1132430055/0 and token/batch counts 5/3. Success/EOS cursor bytes, structured EOS error, foreign report/key negatives, prior report validity after failure, depfiles, localization and predecessor code/data/symbol identity pass. Root verified all 232 files in `e3-public-facade-2ec331c-final-20260924T171734Z/evidence-files.sha256` (`7ad76adc...`) and merged private/public static 20/20, Q0 4/4 and CI 107/107. This is the fixed diagnostic model/profile/canonical byte tokenizer only, with no rank-zero report, generic checkpoint evaluation, trainer-evaluate! or installed retention statistics. The test-only installed report-edge source `d9a3779` is integrated byte-identically at `b47f769`: one poisoned process accepts exactly 8,191 public reports and one charged EOS failure, then rejects the 8,193rd reservation before cursor mutation or public result. Root reviewed its public-only source, passed merged static 4/4, and verified all 115 files in the pinned f31/LLVM21 supported gate `e3-public-quota-d9a3779-20260924T175043Z/evidence-files.sha256` (`b1e400b9...`); the installed quota runtime passed in 616.63 seconds/5,200,280 KiB peak RSS. This is fixed-profile diagnostic proof only. Generic `trainer-evaluate!` and `metrics-ref` remain blocked on accepted TR3/A0 construction. Full E3 acceptance remains pending | active |
| E3-D2 | [Canonical-byte identity and native idle prerequisite](e3/E3_D2_PREREQUISITE.md) | T1, D1, D2; merged PR #104 contract | #106; exact one-form hash-pinned source variant, authentic T1 admission and native idle 0/1/2; private compiled closure, lifecycle, provenance and sanitizer gates; byte-preserving prerequisite union passed supported CI 35744832879 and merged as `33a54ef7`; focused merged-head component, closure, topology and isolation gates passed. Frame-bound restoration, staging, rollback and full E3 composition remain deferred | complete |
| E3-P1 | [Canonical fixed-model mode restoration](E3_P1_MODES.md) | Accepted E3/P1 contract; checked-promotion upstream prerequisite | #108; lexical64..68 preserves first64/public surface. Frozen runtime81298 focused Ubuntu22/LLVM21 evidence passes unchanged O0 ordinary749 checks, O2 poisoned short-region lifetime and genuine comparator reentry, allocation-disabled tails, exact flat1,024/8,192 reuse counters, and all six constructor plus six checked-promotion bind failpoints with retry. Independent final review approved `95dcec4`; root verified the frozen compiled-input hashes and captured outputs. Inherited-package fixed costs/isolation, complete suite union and full E3 composition remain pending | active |
| G3 | Greedy, temperature, top-k/top-p generation with KV cache | M3, T1, A2 | #89 [bounded design and compiled reachability](G3_GENERATION_PROPOSAL.md), decision 5748532295; G3-N, G3-S and shared guards are accepted in the Wave 3 integration branch. [Readiness checklist](g3/G3_T_READINESS.md) supplies pending consumer witnesses only. The bounded G3-C4-N provider is accepted separately below. The independently approved source-private Step16A P1 numerical prefill candidate `8e14106`/tree `051fb5e` is integrated as identical eight blobs at `03e25a0`; root verified `g3c4-prefill1-step16a-8e14106-20260924T225000Z/seal.sha256` and all 16 merged structural contracts. Supported f31/LLVM21 normal/repeat/sanitizer evidence reports 19,295 checks, authentic T1 21-role forward, bit-exact P1 row against accepted T4 row zero, actual K/V position one with keep mask {1,0,0,0}, 21 dispatch, 11 A2 allocation and two runtime-discovery allocation cuts, with old authority and caller output preserved on failure. The independently approved source-private Step17A P2 numerical prefill candidate `9c5b4b6`/tree `a5616f6` is integrated as identical eight blobs at `2b4dabf`, with the exact development-only checker admission at `d9105d6`. Root verified `g3c4-prefill2-step17a-9c5b4b6-20260924T235000Z/seal.sha256`, all 17 merged structural contracts, Q0 4/4 and CI 107/107. Supported f31/LLVM21 normal/repeat/sanitizer evidence reports 19,748 checks, authentic T2 21-role forward, bit-exact P2 last row against accepted T4 row one, two real K/V positions with keep mask {1,1,0,0}, 21 dispatch, 11 A2 allocation and two runtime-discovery cuts, with old authority and caller output preserved on failure. The independently reviewed private Step18A prompt-prefill adapter `90b64db`/tree `37546a3` is integrated as identical eight blobs at `f29ba00`. It authenticates the Step15 owner and rejects invalid P/G or P+G>2 before acquisition/pinning; scoped I1 borrows route exact P1/P2 IDs only to the genuine numerical leaves. Supported f31/LLVM21 normal/repeat/sanitizer runs are byte-identical with 18,373 checks, two routes, five pre-acquire cuts, 42 numerical cuts and three borrow cuts, preserving cache/binding/RNG/output/call state on failure. Root verified `g3c4-prompt-prefill-step18a-90b64db-20260925T001500Z/SEAL.sha256` (`c8364976...`) and merged structural/Q0 4/4/CI 107/107 gates. This remains private: public generation/result/text/EOS, package exports and end-to-end generation/save-reload acceptance remain pending | active |

| G3-N | [Exact T1 forward numerical provider](G3N_PRIMITIVES.md) | K1, Q0, I1, I2, N2, N3K, A2, PR #92 | #93; ABI 1.0 six capabilities/seven operations/eleven pairs; independent G3-N-R approval 5771670331, supported full CI 35681252651 attempt 2 (19 suites / 27 commands), PR #97 merge `ca3880f` and root focused merged-head retest 5771719405. [Provenance](G3N_PRIMITIVES.md#integration-provenance) retains the initial P1 timeout/sole retry and distinct candidate/merged trees; no model/generation claim | complete |
| G3-S | [Bounded deterministic native sampler](G3S_SAMPLER.md) | K1, Q0, I1, I2, N2, N3K, A2; accepted PR #92 | #94; exact greedy/categorical ABI1.0 from PR #96 head `4a7ecdc` accepted into Wave 3 integration as reviewed merge `37d3b05`, tree `551c192`. Provider/header blobs are unchanged; only the accepted predecessor source hash was repinned. Supported exact-head CI and fresh pinned LLVM21 union gate passed: 59,028 numerical and 37,796 adversarial checks, 23/23 mutation rejects, package/private AOT/stack/ABI checks, and ASan/UBSan with leak detection; sealed evidence `g3s-merge-37d3b05-20260924T074900Z` (SHA256SUMS `f95d0106...`). Twenty-suite/35-command topology and 73 static CI checks passed. No G3-T/public generation claim | complete |
| W3-GUARD | Checked-allocation cleanup ordering | M3 shared calls, C2 state borrows, runtime repair #108 | [M3 call guard #117](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/117) has independently reviewed source plus passing ordinary and final-runtime handler-failure evidence. [C2 cleanup ordering #121](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/121) covers eight audited C2 handler-setup windows plus the concrete P1 owned-release setup prerequisite; its exact 5/6/11 final-runtime witness and focused owner/LOAD gates pass. The isolated union adds the exact reserve dependency to all ten P1-containing aggregate manifests. Full inherited package/retention gates, supported union CI, and integration acceptance remain pending; no new public API or checkpoint format | active |
| G3-C4-N | [Four-position generation numerical provider](G3C4_PRIMITIVES.md) | K1, accepted N2/N3K/A2/G3-N numerical contracts | 28 bounded operation/shape pairs implemented at `b04edf0`; independent source and exact integration reviews approved. [PR #122](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/122) merged as `0342932`, tree `df022486337c92a560147bf1f0d3a9f7a901b5b8`, identical to the reviewed candidate and authenticated supported CI 35799381054 attempt 2: all 19 suites/30 commands, evidence and final aggregation passed. C4 numerical/ABI/two-fresh-AOT/sanitizer gates passed with leak detection. This completes the bounded numerical provider; full C4 model, continuation and generation persistence remain separate dependencies | complete |
| G3-C4-NATIVE | [Seeded native owner and fixed-14 pin integration](WAVE3_G3_I2_NATIVE_INTEGRATION.md) | G3-C4-N, N3K, I2, M3/M3T | Root accepted the bounded native-only parent `25a700b` / tree `f0f6c150` after exact blob, supported owner/pin/C2 closure, and independent review. This integration-preparation candidate composes the independently approved and supported I2 restore sibling `e20677f`, preserving SHARED-R2 and one inherited I2 registry; the pin ordinary-object baseline advances to that sibling. Both exact combined native gates pass on supported Clang 21/GCC 11, including normal, repeat, sanitizer, symbol, object, restore, owner, pin, and C2 checks. Root independently audited and accepted the bounded combined native source `3928fc0` / tree `66ff3019`; package/AOT/full CI remain pending. No active model seam, construction bridge, Eshkol/P1 construction, O2 restore, transport, cache, sampler, public API, package manifest or CI registration is added | review |
| G3-C4 I2 construction bridge | [Private construction-admission bridge contract](g3/G3_C4_I2_CONSTRUCTION_BRIDGE_LEAF_CONTRACT.md) | Accepted G3-C4 native union; existing I2 package bridge | The accepted two-function conditional I2 wrapper preserves the I2 restore/R2 lineage. The independently reviewed successor lineage implements the P1 prepared split, sole 12-slot C4 model authority, atomic seeded construction, fixed C4 A2 cache context, authenticated active-call tuple, native generator/RNG ownership, and private one-registry Step 6A call-entry/empty-commit witness. Root verified the 5,077-file sealed supported LLVM 21 preflight for exact isolated candidate `780a64d` (including model-authority 25, call-entry 66, allocation 135, publication 17, retention 1,024/8,192, and native generator checks) and integrated it at `98a2ab8`, tree `f171e67`. Combined-head static contracts, 107 CI checks, four Q0 checks, and 20-suite/35-command topology pass. The independently reviewed [Step 7A provider-route leaf](g3/G3_C4_PROVIDER_ROUTES_STEP7A_CONTRACT.md) at `730e5b7`/tree `bebc884` is integrated at `da79636`/tree `78325a7`: it stages and atomically admits the exact G3-C4 full-prefix rows and both actual G3-S sampler rows during private generator construction without invocation or exported API. Supported f31/LLVM 21 normal, repeat, ASan/UBSan/LSan, 79 focused checks, symbol/macro/closure audits and four Q0 checks pass; sealed evidence is `g3c4-step7a-provider-routes/final-730e5b7` (`SHA256SUMS` `8878b39f...`), root verified. Merged-head 107 CI, four Q0, six G3 source contracts, and 20-suite/35-command topology pass. The independently reviewed [Step 8A fixed full-prefix leaf](g3/G3_C4_FULL_PREFIX_FORWARD_STEP8A_CONTRACT.md) at `c96714b`/tree `8113d10` is integrated at `7ce6dd6`/tree `c3e0443`: it consumes the admitted C4 runtime through the exact 21-role T=4 schedule and candidate A2 transaction, aborts that transaction on success, and publishes logits only after complete success. Supported f31/LLVM 21 normal, repeat and ASan/UBSan/LSan gates passed 2,475 checks, six A2 allocation cuts, token sensitivity, 21-site supplemental first-error/atomicity review, exact closure/symbol audits and four Q0 checks; sealed evidence is `g3c4-step8a-full-prefix/final-c96714b` (`SHA256SUMS` `b376b44e...`), root verified. Merged-head 107 CI, four Q0, G3 source contracts and 20-suite/35-command topology pass. The independently reviewed [Step 9A private sampler transport](g3/G3_C4_SAMPLER_TRANSPORT_STEP9A_CONTRACT.md) at `209a335`/tree `be43c84` is integrated at `c70e09b`/tree `7be4016`: it extracts the owned last logits row, dispatches the exact G3-S greedy/categorical schema under an authenticated one-token generation call, and returns speculative token/RNG without mutating generator RNG, cache, policy, call, or results. Supported f31/LLVM 21 static, normal, repeat, ASan/UBSan/LSan, 371 checks, both modes, exact closure/symbol/macro/Q0 gates pass; sealed evidence is `g3c4-step9a-sampler/final-209a335-r3` (`SHA256SUMS` `580a6bba...`), root verified. Merged-head 107 CI, four Q0, G3 static and 20-suite/35-command topology pass. The independently reviewed [Step 10A private token frame](g3/G3_C4_TOKEN_FRAME_STEP10A_CONTRACT.md) at `2aea957`/tree `c3f7cc8` is integrated at `23637ae`: it owns one speculative sampled token and pending A2 append, stages matching finite `[1,2,1,2]` K/V, and commits cache, token, RNG and budget through an infallible tail. Supported f31/LLVM21 static, normal, repeat, ASan/UBSan/LSan, 400 checks, six sampler dispatches, exact closure/symbol/macro and Q0 gates pass; sealed evidence is `g3c4-step10a-token-frame/final-2aea957` (`SHA256SUMS` `7c2ef717...`), root verified. The independently reviewed [Step 11A private one-token numerical forward](g3/G3_C4_TOKEN_FORWARD_STEP11A_CONTRACT.md) at `4e2d092`/tree `25b72a8` is integrated at `8e99d67`: it composes the accepted G3-N, N2 and G3-C4 T=1 rows, captures committed A2 position before append, stages actual candidate-derived K/V, attends over the pending cache, and leaves the Step 10A frame ready for joint publication. Supported f31/LLVM21 normal, repeat, ASan/UBSan/LSan gates passed 10,388 checks, exact 21-role routing, 31 failure cuts with first-error and caller-state atomicity, four-row bit-exact parity against the accepted T=4 provider, exact closure/symbol audits and four Q0 checks; sealed evidence is `g3c4-step11a-token-forward/final-4e2d092` (`SHA256SUMS` `f45bf480...`), root verified. The independently reviewed [Step 12A private P3 prefill](g3/G3_C4_PREFILL3_STEP12A_CONTRACT.md) at `492a4b4`/tree `070c597` is integrated at `3a0b3f6`: it runs the real 21-role T=3 schedule into an unpublished A2 cache, atomically replaces cache and a 14-identity/4,768-byte parameter snapshot, rejects stale continuation and complete cache/parameter storage overlap. Supported f31/LLVM21 network-none normal, repeat and ASan/UBSan/LSan passed 19,148 checks, including row-two and later fourth-token parity with accepted T=4, 21 dispatch cuts, 11 A2 cuts, discovery, stale binding and alias failures; four Q0 and exact source checks pass. Sealed `g3c4-step12a-prefill3/final-492a4b4` (`SHA256SUMS` `a7dae791...`) is root-verified. Local integrated source blobs match the reviewed candidate; 107 CI, 12 G3 source checkers, focused Q0 isolation and topology pass. The independently reviewed [Step 13A private last-logit frame](g3/G3_C4_LAST_LOGIT_FRAME_STEP13A_CONTRACT.md) at `58337c1`/tree `4775dbe` is integrated at `9ae1eaf`: it dispatches accepted G3-S greedy/categorical `[1,256]` schema from a real last-logit row, releases the A2 borrow, and opens the accepted Step10A pending frame without inventing a `[1024]` carrier or publishing state. Supported f31/LLVM21 normal, repeat and ASan/UBSan/LSan gates pass 4,348 checks, exact P3→sample→Step11→publish parity with T4, dispatch/allocator/alias/stale/nonfinite/exhausted cuts, exact object-symbol delta and four Q0 checks; sealed `/tmp/g3c4-last-logit-frame-evidence-58337c1-final/SHA256SUMS` (`a569bd8a...`) is root-verified. Local integrated blobs match the candidate; 107 CI, 13 G3 source checkers, four focused Q0 isolation tests and topology pass. The independently reviewed [Step 14A private T1/eval admission](g3/G3_C4_T1_EVAL_ADMISSION_STEP14A_CONTRACT.md) at `cad7cf5`/tree `7343209` is integrated at `a13d87f`: it retains an exact same-aggregate baseline T1 shell in generator slot 6, authenticates the V256 profile/fingerprint and P1 eval mode before construction and before/after private native call admission, and preserves rollback/close on drift. No raw T1 pointer crosses into C. Supported f31/LLVM21 inherited call-entry gate passed 66 semantic, 135 allocation/handler, five publication cuts, 1,024/8,192 retention and native regression; Step14A passed 29 identity/stale/alias/rollback checks normal/repeat/ASan/UBSan/LSan with identical output. Sealed `g3c4-t1-eval-admission-cad7cf5-20260924/SHA256SUMS` is root-verified. Local integrated blobs match reviewed candidate; 107 CI, 14 G3 source checkers, four focused Q0 isolation and topology pass. The independently reviewed [Step 15A private authenticated prompt owner](g3/G3_C4_PROMPT_T1_BORROW_STEP15A_CONTRACT.md) at `e7c7921`/tree `d68cc92` authenticates exact same-aggregate sealed T1 inputs and synchronously copies P1/P2 byte IDs into independently owned I1 `[1,P]` storage, with typed release and no retained T1 lease or invented raw-borrow API. Supported f31/LLVM21 network-none normal, repeat and ASan/UBSan/LSan gates pass 72 focused checks, 3/1 Eshkol allocation cuts and 1,024/8,192 bounded retention using one reused P2 shell; the unskipped Step14/Step6 regressions, 15 G3 checkers and four Q0 checks pass. Sealed evidence is `g3c4-prompt-t1-borrow-e7c7921-20260924/SHA256SUMS` (`14ae1fd9...`); independent review approved with no remaining findings. A read-only source-grounded audit on clean `1cd4d74` found the accepted Step15 input owner admits only P1/P2 while Step12 prefill is fixed P3 across its 21-role schedule, cache length, binding and last-logit row; padding or a direct adapter would fabricate token/K/V state. Sealed `g3c4-prompt-prefill-bind-blocked-1cd4d74-20260924/manifest.sha256` is root-verified. A genuine private P1/P2 numerical prefill and pre-acquire P+G admission are required before binding the owned prompt; public result/text/EOS, persistence, packaging and full combined CI remain pending | active |
| TR3-F32 | [True-binary32 scalar runtime prerequisite](TR3_F32_SCALAR_RUNTIME_CONTRACT.md) | A0, R0 | Accepted bounded implementation contract for canonical native/VM/FFI scalar bits, explicit f64 arithmetic promotion, regions and parity. Isolated Eshkol native/FFI representation source `f52bbed7` plus shared-export fix `db0e83b5` / tree `aca2ad0` is independently reviewed; supported immutable f31/LLVM21 Release focused CTest 5/5 and fresh ASan+UBSan native 4/4 passed. The subsequent raw LLVM/HoTT/VM transport slice `ceb1f746` / tree `ec33f62` passed two targeted source reviews. Its separate supported Release gate passed 8/8 plus HoTT 15/15; strict ASan+UBSan passed 6/8 plus HoTT 15/15, with preserved unrelated VM misaligned-access and parser `strdup` leak failures in two broad binaries. Root verified source, log and artifact manifests. The separate upstream Eshkol runtime lineage has independently reviewed allocator/LLVM extern/equality/hash/formatting leaves, then negative persistence `d0fa393a` (native/VM KB and ESKB v1 reject unsupported f32) and negative generic JSON `5fa58a4` (native/VM direct and nested f32 reject before output/file publication). Supported f31/LLVM21 JSON gates passed focused Release 7/7, existing JSON 3/3, f32 label 32/32, strict native/AOT/VM sanitizer 5/5 and JIT sanitizer 2/2 with the known frontend leak-detector limitation; evidence `f32-json-20260924/logs.sha256` (`7076dac9...`). The independently reviewed upstream normalization repair `e6aa1db`/tree `b64bc16` uses one canonical f32-to-f64 lowering for raw/tagged values, admits exact f32 batch/layer normalization scalars, and rejects active AD f32 before node construction. Its sealed supported f31/LLVM21 dependency-clean 1/1, Release 7/7, f32 label 38/38, cache-disabled O0/O2 JIT, and sanitizer gates pass (`f32-normalization-20260924`, `SHA256SUMS` `9050ad65...`), root verified. The independently reviewed bounded workspace repair `9b1da2a`/tree `9b8d67e` promotes exact f32 salience before softmax and rejects malformed tag 11 before mutation. Supported f31/LLVM21 Release focused 5/5, full f32 label 43/43, existing O0/O2 workspace regressions, and native/AOT/JIT sanitizer gates pass; sealed `f32-workspace-20260924/SHA256SUMS` (`ed0fae29...`) is root-verified. The independently reviewed bounded system-int repair `2018aef1`/tree `8c3bd93` rejects exact tag 11 before integer/resource lookup or mutation, while `format-relative` alone canonically promotes f32 then uses the existing f64-to-int64 truncation. Supported f31/LLVM21 focused Release 5/5, full f32 label 48/48, event-loop regression, native direct sanitizer, AOT O0/O2, and cache-disabled JIT O0/O2 gates pass; sealed `f32-system-int-20260924/SHA256SUMS` is root-verified. Live WebSocket service was not fabricated; nonfinite format-relative remains an existing DOUBLE-cast limitation. The independently reviewed bounded ISO time repair `218886ce`/tree `744e870` validates canonical finite in-range f32 before `gmtime` or publication and then uses the historical f64 truncation path. Supported f31/LLVM21 focused Release 5/5, full f32 label 53/53, VM date/time 7, native time API 15/15, AOT O0/O2 sanitizer 3/3, and cache-disabled JIT O0/O2 2/2 gates pass; sealed `f32-time-format-20260924/SHA256SUMS` is root-verified. Historical DOUBLE nonfinite/out-of-int64-range behavior and platform-limited `gmtime` remain outside this leaf. The independently reviewed bounded `allow-sleep` repair `bee408ff`/tree `18855f8` rejects exact f32 before inhibitor-handle lookup or mutation; a same-process witness preserves the live handle for exactly one release. Supported f31/LLVM21 Release focused 5/5, full f32 label 53/53, VM system-info T1-T6, and AOT/JIT O0/O2 sanitizer gates pass; sealed `f32-allow-sleep-20260924/SHA256SUMS` is root-verified. Windows ordering is source-verified only. The independently reviewed bounded `process-wait` repair `20e3c597`/tree `7cfa56e` rejects exact f32 before PID extraction or process wait; a real-child public witness proves the INT64 wait still receives status 7 after the f32 rejection. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, VM process/system regressions, AOT and cache-disabled JIT O0/O2 sanitizer gates pass; sealed `f32-process-wait-20260924/SHA256SUMS` is root-verified. Windows ordering is source-verified only. The independently reviewed bounded `poll-fd` repair `88fc06bc`/tree `b9e1ed0` rejects exact f32 descriptor or timeout before raw extraction or poll; a real ready-pipe witness proves the alias cannot report readiness while the INT64 descriptor still can. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, language surface O0/O2 23/23, AOT and cache-disabled JIT O0/O2 sanitizer gates pass; sealed `f32-poll-fd-20260924/SHA256SUMS` is root-verified. Windows no-op path is source-verified only. The independently reviewed bounded `file-chmod` repair `2b9a562f`/tree `6b2c04c` rejects exact f32 mode before path/capability lookup or filesystem mutation; real-file O0/O2/JIT witnesses preserve 0644 after f32 rejection then admit INT64 0600. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, language surface O0/O2 23/23, AOT and cache-disabled JIT sanitizer gates pass; sealed `f32-file-chmod-20260924/SHA256SUMS` is root-verified. Windows no-op path is source-verified only. The independently reviewed bounded `process-kill` repair `fdefda7e`/tree `a319165` rejects exact f32 PID or signal before raw extraction/OS action, with a readiness-handshaked real-child delivery witness and bounded cleanup. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, language surface O0/O2 23/23, signal/subprocess regressions 2/2, and native/AOT/JIT sanitizer gates pass; sealed `f32-process-kill-20260924/SHA256SUMS` is root-verified. Windows no-op path is source-preserved only. The independently reviewed bounded `process-kill-tree` repair `d6bec5c`/tree `afde2dc` rejects exact f32 PID or signal before either OS action. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, real process-group delivery, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass; sealed `f32-process-kill-tree-20260924/SHA256SUMS` is root-verified. Windows non-f32 behavior is source-preserved only. The independently reviewed bounded `process-setpgid` repair `52bfac9`/tree `b5f56a3` rejects exact f32 PID or PGID before OS mutation. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, real child-group controls, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass; sealed `f32-process-setpgid-20260924/SHA256SUMS` is root-verified. Windows non-f32 source behavior is preserved only. The independently reviewed bounded `process-read-nonblocking` repair `4679ead`/tree `ba92403` rejects exact f32 FD or max bytes before flag change, allocation or read; real O0/O2 pipes retain all bytes on rejection. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, process-tree/signal regressions and native/AOT/JIT sanitizer gates pass; sealed `f32-process-read-nonblocking-20260924/SHA256SUMS` is root-verified. Windows/WASM non-f32 source paths are preserved only. The independently reviewed bounded `socket-send` repair `5ee5247`/tree `1e0cfd8` rejects exact f32 descriptor before payload extraction or send; real AF_UNIX O0/O2 witnesses preserve peer non-readiness and then deliver exact INT64 bytes. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess regressions and native/AOT/JIT sanitizer gates pass; sealed `f32-socket-send-20260924/SHA256SUMS` is root-verified. Windows/WASM non-f32 source paths are preserved only. The independently reviewed bounded `socket-recv` repair `b440eec`/tree `40fae18` rejects exact f32 descriptor or maximum before queue/flag mutation; real AF_UNIX O0/O2 witnesses preserve all bytes and recover exact payloads through INT64. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess regressions and native/AOT/JIT sanitizer gates pass; sealed `f32-socket-recv-20260924/SHA256SUMS` is root-verified. Windows/WASM non-f32 source paths are preserved only. The independently reviewed bounded `socket-close` repair `b3db4f5`/tree `f6e0221` rejects exact f32 descriptor before close; real AF_UNIX O0/O2 witnesses preserve endpoint liveness, deliver exact bytes, then close through INT64 and confirm EBADF. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess regressions and native/AOT/JIT sanitizer gates pass; sealed `f32-socket-close-20260924/SHA256SUMS` is root-verified. Windows/WASM non-f32 source paths are preserved only. At that clean commit, `term-set-scroll-region` still accepts f32 aliases for top/bottom and emits exact DECSTBM escape bytes on real PTYs at O0/O2; the independently reviewed counterexample is sealed at `f32-term-scroll-counterexample-20260924/SHA256SUMS`, root-verified. The independently reviewed `term-set-scroll-region` repair `db74c1b8`/tree `97e89ec` rejects f32 top then bottom before raw reads, range/TTY checks, escape output or wrapper assignment. Fresh real PTYs prove zero-byte rejection and exact six-byte INT64/raw-DOUBLE controls at O0/O2; supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2, native/AOT sanitizer 3/3 and JIT sanitizer 2/2 pass. Root verified `f32-term-scroll-20260924/SHA256SUMS` (`90640fcc...`). The next clean real-file/live-watcher counterexample proves `fs-watch-poll` f32 timeout aliases consume the exact event; root verified `f32-fs-watch-poll-counterexample-20260924/SHA256SUMS` (`ffceb81e...`). The independently reviewed `fs-watch-poll` repair `bcca8e44`/tree `ae16d2f` rejects f32 timeout before watcher lookup/snapshot/event consumption. Real-file O0/O2 witnesses preserve the pending event for same-handle INT64 recovery; supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass. Root verified `f32-fs-watch-poll-20260924/SHA256SUMS` (`af58496d...`). At that clean commit, the next source-order `fs-unwatch` real-watcher counterexample proves f32 slot aliases remove a live watcher and discard the pending event; root verified `f32-fs-unwatch-counterexample-20260924/SHA256SUMS` (`e8ef228f...`). The independently reviewed `fs-unwatch` repair `81e5fa33`/tree `9b6211b` rejects f32 slot before watcher deletion or event loss. Real-file O0/O2 witnesses preserve the pending change for same-handle INT64 recovery and release; supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass. Root verified `f32-fs-unwatch-20260924/SHA256SUMS` (`71ba2203...`). At that clean commit, the next source-order `string-truncate-display` counterexample proves f32 2.0 raw width 1,073,741,824 returns `abcdef` instead of INT64 2 result `..`; root verified `f32-string-truncate-display-counterexample-20260924/SHA256SUMS` (`0ca86a74...`). The independently reviewed `string-truncate-display` repair `5075edb6`/tree `eb85c1d` rejects f32 width after existing null-input precedence but before raw width/substring/allocation/output mutation. Public O0/O2 and native controls preserve INT64 width2 and raw DOUBLE word2 result `..`; supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass. Root verified `f32-string-truncate-display-20260924/SHA256SUMS` (`a1ca5fc6...`). The next source-order `string-index-of` start counterexample proves canonical f32 payload word2 aliases integer start2 and changes the returned match; root verified `f32-string-index-of-counterexample-20260924/SHA256SUMS` (`bab7ec36...`). The independently reviewed `string-index-of` repair `529c3535`/tree `9d8602f` rejects f32 start after existing haystack/needle invalid-input precedence but before raw index/search/output mutation. O0/O2 public/native controls preserve INT64 starts0/2 and raw DOUBLE word2; supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass. Root verified `f32-string-index-of-20260924/SHA256SUMS` (`b236ad08...`). Bounded direct-read inventory now finds one shared string-pad helper, two remaining direct raw numeric reads and four exposed argument positions in `system_builtins.c`; it is not a whole-program closure claim. The next `string-pad-left/right` width counterexample proves f32 3.0 becomes raw width 1,077,936,128 and clamps to a million-character result rather than INT64 width3 result `007`; root verified `f32-string-pad-width-counterexample-20260924/SHA256SUMS` (`5142be00...`). The independently reviewed shared `string-pad-left/right` width repair `795ae8bd`/tree `34f67f1` rejects f32 width after existing input precedence but before clamp/codepoint/allocated result. O0/O2 public/native controls preserve INT64/raw DOUBLE width3 results `007`/`700`; supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass. Root verified `f32-string-pad-width-20260924/SHA256SUMS` (`8f8dca65...`). Bounded direct-read inventory in `system_builtins.c` now has one unguarded shared codepoint read exposed by two public wrappers; it is not whole-program closure. The next O0/O2 counterexample proves f32 codepoint48.0 reads raw 0x42400000 and silently pads spaces rather than the INT64 codepoint48 `0` result; root verified `f32-string-pad-codepoint-counterexample-20260924/SHA256SUMS` (`f3bf924f...`). The independently reviewed shared string-pad codepoint repair `5946fd9d`/tree `e8770eb` rejects f32 only when padding is required, after input/width precedence and before raw codepoint/fallback/allocation. Public/native O0/O2 controls preserve INT64/raw DOUBLE codepoint48 `007`/`700`; supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass. Root verified `f32-string-pad-codepoint-20260924/SHA256SUMS` (`f65bca77...`). The sealed bounded direct-read inventory in `system_builtins.c` is not closed: file-lock and file-unlock each still read one unguarded descriptor payload; all other classified direct tagged numeric reads in this translation unit are guard-dominated or non-exposed. This is not a whole-program proof. The independently reviewed `file-lock` descriptor repair `af6dbe56`/tree `3cbf650` rejects f32 before the raw fd/fcntl/output, while INT64 and historical raw DOUBLE controls preserve Linux advisory-lock behavior. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass; root verified `f32-file-lock-20260924/SHA256SUMS` (`fc6bdd44...`) and its real-lock before-state seal (`c06e28e5...`). The bounded direct-read inventory now has one unguarded resource read in this translation unit: file-unlock descriptor, one public position. A separate real parent/child advisory-lock O0/O2 counterexample proves f32 alias unlocks the live lock and permits child acquisition; root verified `f32-file-unlock-counterexample-20260924/SHA256SUMS` (`8268524f...`). The independently reviewed `file-unlock` descriptor repair `c27f85eb`/tree `a34b595` rejects f32 before raw fd/fcntl/output, preserving a real held lock on canonical and malformed failure; same-handle INT64 unlock then releases it. Supported f31/LLVM21 focused 5/5, full f32 label 53/53, O0/O2 system 23/23, signal/subprocess 2/2 and native/AOT/JIT sanitizer gates pass; root verified `f32-file-unlock-20260924/SHA256SUMS` (`1af667f1...`). The sealed bounded direct tagged numeric payload inventory is now closed for audited integer/resource positions in `lib/core/system_builtins.c`. It does not prove whole-program, VM, persistence, AD, accelerator or full-F32 closure; format-relative and ISO-8601 quantities remain separately documented. A sealed read-only contract audit on clean `c27f85eb` finds the bounded system source closure insufficient for full F32 acceptance: public source-level `float32?` is missing; native/AOT and VM `type-of` disagree; VM hash-key operations reject F32; the exhaustive successor-era classifier/dispatch audit, full admitted-operation parity table, exact-bit closure/continuation/exception/VM egress witnesses, flat allocation-count and prior-runtime symbol-absence measurements, two fresh deterministic AOT builds and consolidated supported-host gate remain pending. Root verified `f32-contract-gap-audit-c27f85eb-20260924/SHA256SUMS` (`f3413eef...`). Contract deferrals for positive persistence/AD/accelerators remain deferrals, not gaps. The F32 head is a clean partial baseline, not an accepted F32 side for the TR3-guard union. Full upstream acceptance, F32/guard runtime union and transformer runtime-pin adoption remain pending. No positive f32 persistence or full F32 claim | active |
| TR3-METRICS | [Public trainer metrics successor design](TR3_PUBLIC_METRICS_PROPOSAL.md) | E3, TR3-F32, TR3 lease | Accepted design for one source-private metrics authority, exact f32-bit/i64 result schemas, measured linear process-lifetime retention and E3-owned selected-bit projection after successful evaluation; runtime f32 implementation/repin, E3 seam, trainer source, public tests and aggregate acceptance remain pending | active |
| TR3 | [Trainer state machine and exact resume](TR3_P1_FIXED_SET.md) | M3, L2, O2, D2, C2, E3, SHARED-R2, TR3-F32 | #116 [private objective bridge candidate](TR3B_OBJECTIVE_BRIDGE.md) is preserved from exact source `071abb72` with independent approval and a passing supported Ubuntu 22.04/LLVM 21.1.8 focused gate. The bounded P1 slots69/70 fixed-set leaf preserves the accepted E3 prefix and public surface with exact zero-allocation/count/retention evidence and independent review. [#119 private O2 update-and-clear candidate](TR3_O_OPTIMIZER_TRANSACTION_PROPOSAL.md) is preserved from exact implementation source `783dc576` with local atomicity, numerical, lifetime and CI-registration evidence; independent review approved the exact `2006cde` B/O integration. The accepted [TR3-C live restore contract](TR3_C_LIVE_STATE_CONTRACT.md) now has bounded checked-I2 cleanup and feature-local private D2 restore leaf seams with focused sanitizer, symbol-isolation, strict-AOT, lifecycle, and independent-review evidence. The independently reviewed [source-grounded joint composer plan](TR3_C_JOINT_RESTORE_IMPLEMENTATION_PLAN.md) maps exact C2/D2/I2/O2 APIs, ownership transfers and commit/abort order. The independently reviewed 22-slot lease successor now owns epoch-start/RNG/T/U/E and narrow restore enter/recheck/abort/commit-controls/finish transitions. Exact production-OFF de0b249 supported O0 and O2 allocation-failure matrices each pass 1,280 cases/65,235 checks with byte-identical case statuses after the rooted D2 u64-limit repair; O2 compiled under the pinned 1,200-second bound in 1,105.48 seconds, 5,483,212 KiB peak RSS and zero process swaps. Sealed O0/O2 evidence is `tr3-c-lease-o0-full-8b102ed-20260924T071200Z`/`tr3-c-lease-o2-full-a190ca4-20260924T040000Z`. Reviewed lease/D2 source is integrated at `8f64aad`. The independently reviewed partition-2 private bindings are integrated at `b428da3`: fixed hidden I2/O2 source ABI adapters, exact O2-plan-as-I2-authority, and assignment-only transfer of all 28 O2 stage wrappers. After repairing sealed-tail transitive fallibility, focused source contracts, 340-check Clang/GCC and sanitizer/symbol-isolation gates, nine source-ledger runtime checks, and strict exact-81298 package compilation passed; sealed evidence is `tr3-c-restore-bindings-5a0b9e6-20260924T143000Z` (`SHA256SUMS` `cf5e86a4...`). The independently reviewed private partition-3 composer `dff93fb` is source-staged in the integration branch at `530f470`: it stages the accepted C2 borrow into 14 parameter and 28 moment owners, ends active P1 tensor access before C2 borrow shutdown, prepares D2/O2/I2 in fixed order, authenticates the exact native O2 plan through the 22-slot lease, and uses a checked assignment-only irreversible tail. Its sealed `tr3-c-joint-composer-dff93fb-20260924T050500Z` evidence (`SHA256SUMS` `9b340e91...`) records 34 focused source tests, the Clang/GCC/sanitizer binding gate, and exact-81298 O0 aggregate compilation (94.84 seconds, 4,652,876 KiB peak RSS, zero swaps). The independently reviewed authentic joint-restore repair `a8c6dd0` authenticates the full 15-path tied P1 projection, stages its 14 primaries, normalizes C2 expanded cursor fields 3/4 to lease-owned bytevectors, and compares canonical X1 bytes with the accepted D2 comparator. Root integrated the exact four source/test blobs at `5b3a31b`, tree `5788e8b`, and reran the genuine supported f31/81298 strict-AOT gate on that clean combined tree: 67 checks, all 42 tensor bytes and controls match after restore, injected last-recoverable O2-check abort preserves the receiver, and same-owner retry succeeds. Compilation took 95.06 seconds and 4,800,592 KiB peak RSS with zero swaps; runtime took 0.29 seconds and 79,444 KiB peak RSS with zero swaps. Sealed root evidence is `tr3-c-joint-root-5b3a31b-20260924` (`SHA256SUMS` `b6805ffa...`). The bounded actual-transaction O0 failure-cut gate at isolated test-only `b8a1372` passed 1,905 arena cut cases/76,200 checks with receiver/source byte preservation and same-owner retry, then stopped at the first genuine counterexample: arena-string allocation denial at cross-validation ordinal 1,905 caused runtime SIGSEGV before recovery. Sealed evidence is `tr3-c-joint-alloc-o0-b8a1372-20260924T110000Z` (`SHA256SUMS` `31367be7...`). Read-only attribution on both exact production-OFF `de0b249` and final runtime `81298` found a shared upstream AOT string-copy null-allocation bug: `eshkol_utf8_substring` returns null, while `StringIOCodegen::substringImpl` packs it without the existing constructor-allocation check, then a null-minus-eight header read faults. The required O2 config deep copy is correct; transformer source remains unchanged. Sealed diagnosis is `tr3-c-joint-alloc-1905-attribution-20260924T113500Z` (`SHA256SUMS` `ba1f0e1f...`). The independently reviewed isolated upstream repair `b28f43f`/tree `cdf5735` adds the existing constructor-allocation guard before pointer packing; five strict-AOT/emergency/IR tests pass (`eshkol-string-allocation-guard-b28f43f-20260924T122500Z`, seal `ff49d51b...`). A production-OFF guarded runner passed ordinal 1905 with 40 preservation/ownership/retry checks (`tr3-c-joint-ordinal1905-c8c4ce2-b28f43f-20260924T121500Z`, seal `a1134ebf...`), then passed arena cuts 1906–2198 (293 cases/11,720 checks) before stopping at a new genuine arena-header denial/SIGSEGV at ordinal 2199 (`tr3-c-joint-arena-resume1906-c8c4ce2-b28f43f-20260924T122000Z`, seal `e63ba80d...`). The guarded runtime and transformer test-only manifest repin remain isolated; root verified all seals. Ordinal 2199 was attributed to upstream `string->utf8` packing a null denied allocation before a length store/copy. Independently reviewed Eshkol guard `71cbd69a`/tree `f64e00d` inserts the existing constructor-allocation check, passes seven focused f31/LLVM21 strict-AOT/IR tests, and leaves production archive bytes unchanged (`eshkol-string-utf8-allocation-guard-71cbd69-20260924T132000Z`, seal `67bb53b5...`). Isolated transformer test-only repin `da5f139` passed ordinal 2199 with 40 preservation/ownership/retry checks (`tr3-c-joint-ordinal2199-da5f139-71cbd69-20260924T133000Z`, seal `a4141912...`), then stopped at the next arena-header denial/SIGSEGV at ordinal 2200 (`tr3-c-joint-arena-resume2200-da5f139-71cbd69-20260924T133500Z`, seal `4493141d...`). Root verified the source and ordinal-2199 evidence seals. Distinct upstream `utf8->string` null-allocation guard `bce74b4`/tree `0f07e7a` passed seven supported strict-AOT/IR checks; isolated ordinal 2200 then passed 40 checks and cuts 2201–2436 passed 236 cases/9,440 checks (`tr3-c-joint-arena-resume2201-849b156-bce74b4-20260924T142500Z`, seal `64a53990...`). Independently reviewed upstream `make-bytevector` guard `cb8abee`/tree `e29c4c3` passed nine focused checks; isolated ordinal 2437 passed 40 checks and cuts 2438–2474 passed 37 cases/1,480 checks (`tr3-c-joint-arena-resume2438-d07025b-cb8abee-20260924T152500Z`, seal `b2c0f6d9...`). Root verified both source and ordinal/resume seals. Ordinal 2475 was an upstream escaped mutable-capture allocation null-store defect. Independently reviewed closure-capture guard `36b0c2b`/tree `ef1b079` preserves conditional initializer and intervening branch writes while checking the 16-byte capture box before pointer use; supported f31/LLVM21 O0/O2 semantic/emergency/IR gates pass 11/11 (`eshkol-closure-capture-allocation-guard-36b0c2b-20260924T164500Z`, seal `b8528c45...`). Isolated ordinal 2475 passed 40 checks; 2476–2504 passed 29 cases/1,160 checks before the next denied 48-byte arena-closure allocation/SIGSEGV at ordinal 2505 (`tr3-c-joint-arena-resume2476-d39f33d-36b0c2b-20260924T170500Z`, seal `fc8c9e34...`). Root verified source/ordinal/resume seals. Ordinal 2505 was a capturing-lambda closure allocation followed by an offset-eight env read through null. Independently reviewed upstream guard `b6ff508`/tree `9072c74` covers capturing and zero-capture constructors before pointer use; supported f31/LLVM21 O0/O2 emergency/IR gates pass 13/13 (`eshkol-lambda-closure-allocation-guard-b6ff508-20260924T180000Z`, seal `38f6e1ed...`). Isolated ordinal 2505 passed 40 checks and cuts 2506–2742 passed 237 cases/9,480 checks before the next denied 32-byte arena-string allocation/SIGSEGV at ordinal 2743 (`tr3-c-joint-arena-resume2506-1bdef05-b6ff508-20260924T182000Z`, seal `ac424ec5...`). Root verified source/ordinal/resume seals. Ordinal 2743 was upstream `codegenSymbolToString` packing a denied arena-string allocation. Independently reviewed one-line guard `bba111d`/tree `6a018de` passes supported f31/LLVM21 O0/O2 emergency/IR 15/15; isolated ordinal 2743 passes 40 checks (`tr3-c-joint-ordinal2743-c6fb42a-bba111d-20260924T194500Z`, seal `dead5ebd...`). The next exact cut 2744 immediately denies a length-64/request-80 arena-string allocation and faults (`tr3-c-joint-arena-resume2744-c6fb42a-bba111d-20260924T200000Z`, seal `aeae37f1...`). Root verified source/ordinal/resume seals. The independently reviewed upstream `number->string` null-allocation guard `65db8d6`/tree `7e4f5b4` passes supported f31 O0/O2 emergency/IR 17/17; isolated ordinal 2744 passes 40 checks. Next cut 2745 denied a length-24/request-40 arena string. The independently reviewed upstream `string-append` guard `9e52e1a`/tree `926da9f` checks allocation before GEP/memcpy; supported f31 O0/O2 semantic/emergency/IR 19/19 and isolated ordinal 2745 passes 40 checks. Cuts 2746–3331 pass 586 cases/23,440 checks; exact cut 3332 denies a length-65/request-80 arena string and faults. Root verified `eshkol-string-append-allocation-guard-9e52e1a-20260924T230000Z/SHA256SUMS` and `tr3-c-joint-arena-resume2746-d5ad553-9e52e1a-20260924T235000Z/SHA256SUMS`; the independently reviewed upstream `make-string` guard `ae214210`/tree `5534420` checks allocation before memset/pack. Supported f31 O0/O2 runtime, IR mutant and checked-promotion 34/34 pass; isolated ordinal 3332 passes 40 checks. Cuts 3333–3398 pass 66 cases/2,640 checks; exact cut 3399 denies an arena-header request320/object312 and faults during cross-validation. Root verified `eshkol-make-string-allocation-guard-ae214210-20260925T010000Z/SHA256SUMS`, `tr3-c-joint-alloc-3332-ae214210-20260925T013000Z/SHA256SUMS`, and `tr3-c-joint-alloc-resume3333-ae214210-20260925T014000Z/SHA256SUMS`; the independently reviewed upstream `bytevector-copy` guard `d9e6908`/tree `3c8e7d0` checks allocation before copying/packing. Supported f31 O0/O2 runtime, IR mutant and checked-promotion 36/36 pass; isolated ordinal 3399 passes 40 checks. Cuts 3400–3412 pass 13 cases/520 checks; exact cut 3413 denies an arena-vector capacity-8/request-144 allocation and faults during cross-validation. Root verified `eshkol-bytevector-copy-allocation-guard-d9e6908-20260925T023000Z/SHA256SUMS`, `tr3-c-joint-alloc-3399-d9e6908-20260925T030000Z/SHA256SUMS`, and `tr3-c-joint-alloc-resume3400-d9e6908-20260925T031500Z/SHA256SUMS`; the independently reviewed upstream Scheme-vector `vector-copy` guard `29b9b0a`/tree `6bbddf3` checks allocation before length store, copy and packing. Supported f31 O0/O2 denial/retry and IR mutant gates plus checked-promotion 38/38 pass; exact ordinal 3413 passes 40 checks. Cuts 3414–3450 pass 37 cases/1,480 checks and 3451–4437 return authenticated `END`; the repeated arena census is exactly 3,451 allocation sites and ends at the irreversible tail. Root verified `eshkol-vector-copy-allocation-guard-29b9b0a-20260925T043000Z/SHA256SUMS`, `tr3-c-joint-alloc-3413-29b9b0a-20260925T050000Z/SHA256SUMS`, and `tr3-c-joint-alloc-resume3414-29b9b0a-20260925T051500Z/SHA256SUMS`. The distinct O0 `f32` class is now fully enumerated on the same exact pins: 282 authenticated recoverable cuts at ordinals 0–281, first authenticated `END` at 282, 283 cases/11,308 checks, zero process failures, receiver/source preservation and same-authority retry. Root verified `tr3-c-joint-f32-complete-29b9b0a-1aa6b08-20260924T141000Z/SHA256SUMS` (`a3d65060...`); no code repair was needed. O0 `i2` is also complete on the exact pins: 86 authenticated recoverable cuts at ordinals 0–85, first `END` at 86, 87 cases/3,468 checks, zero process failures, exact receiver/source byte and ownership preservation, and same-authority retry. Root verified `tr3-c-joint-i2-complete-29b9b0a-1aa6b08-20260924T143200Z/SHA256SUMS` (`0598c327...`); no repair was needed. The final O0 `o2` class is complete too: ordinal 0 has an authenticated typed `internal` allocation failure with exact receiver/source preservation and same-authority retry; ordinal 1 is first `END`, 2 cases/68 checks/zero failures. Root verified `tr3-c-joint-o2-complete-29b9b0a-1aa6b08-20260924T144800Z/SHA256SUMS` (`a9e5cd39...`). All four O0 classes are closed on exact isolated pins: arena 3,451 cuts, f32 282, i2 86, o2 1; no root toolchain pin changed. The joint O2 optimized allocation matrix also closes on the same isolated guarded runtime/source pins: independently reviewed test-only selector `875f16c`/tree `68da887` defaults to unchanged O0 and chooses supported O2 with a manifest-pinned 1,200-second compiler bound and exact 1,200-byte diagnostic hash. Supported f31/LLVM21 O2 passed 3,824 cases/152,912 checks with zero process failures: arena 3,451 recoverable cuts then END at 3,451; f32 282 then END at 282; i2 86 then END at 86; o2 1 then END at 1. Every recoverable cut preserved receiver/source bytes and ownership and passed same-authority retry. The optimized compile took 18:57.39 and peaked at 6,743,552 KiB RSS with zero swaps. Root verified `tr3-c-joint-o2-full-875f16c-20260924T150000Z/SHA256SUMS` (`43c76e3f...`). The test selector and upstream guarded runtime remain isolated; root runtime pin is unchanged. The bounded [private snapshot-to-checkpoint publication seam](TR3_C_SNAPSHOT_SAVE.md) composes an exact C2 owner, uses accepted C2 SAVE, and releases that owner on success and writer failure; its focused supported gate passes deterministic bytes and malformed/precommit/postcommit cleanup cases. The bounded [private checkpoint-to-live restore seam](TR3_C_CHECKPOINT_RESTORE.md) uses accepted capability-gated C2 LOAD and joint restore with exact detached-owner release; focused compiled evidence covers checksum/torn/malformed files, forged K2 report, X1 receiver mismatch, and all 42 restored tensor/control images. The [private trainer load-state result-cell entry](TR3_C_PRIVATE_LOAD_ENTRY.md) now delegates to the accepted joint transaction and proves malformed/busy/rollback and exact-image behavior under a supported strict-O0 gate; public trainer wiring still requires a public trainer constructor and package identity. Public trainer, exact resume and #121 final acceptance remain pending. The [bounded I2 restore candidate](TR3_I2_RESTORE_INTEGRATION.md) selectively replays accepted `154d9c3` while preserving the shared provider; its exact-union supported native gate and independent integration review passed at `b3e8464`; broader runtime/package and trainer acceptance remain pending. The independently reviewed [development reference](TR3_REFERENCE_INTEGRATION.md) from `cc474900` is preserved exactly: 33 trajectory points, 14 parameters, 28 moments, weighted held-out and D2 cursor/EOS cases, with strict checksum/type validation; this supplies no native training or resume proof. The bounded TR3-B/O prerequisite union is merged through PR #124 at `f74fede` after supported CI `35925066612` and exact-tree evidence reuse. The [bounded native O2 restore integration](TR3_O2_RESTORE_INTEGRATION.md) at `2aeefad` passed independent review and supported O2/I2/G3/C2 coexistence checks; it preserves linear terminal-control retention and adds no joint trainer transaction. Shared D2 boundary repair `63756bd` preserves same-aggregate invalid-argument errors and admitted typed allocation failures; its supported compiled public gate passed 24 checks. The independently reviewed scoped lease failure runner at `7c83038` passed both canonical O0/O2 matrices on production-OFF de0b249: each 1,145 cases/54,921 checks, identical case status, authenticated evidence; O2 admitted only the exact pinned LLVM vectorization diagnostics. The accepted lease-successor allocation diagnosis requires the width-8 D2 cursor store to reuse the rooted `d2-core-u64-space`; after review its affected-gate union must include E3 because the active E3 run remains pinned to the preceding exact tree and is not repinned in flight. Full supported successor-union CI, package localization/facade remapping, trainer/compositor integration, public trainer, joint restore, one-batch overfit, held-out improvement and interrupted/resumed equivalence remain pending | active |
| CLI3 | Corpus, tokenizer, pretrain, evaluate, generate and inspect CLIs | T2, TR3, G3, X1 | The bounded six-command CLI3-A source from clean `392fac8` is selectively integrated at `62fa411`: tokenizer byte/train-bpe/inspect, corpus build/inspect and checkpoint inspect. Root verified all 21 CLI paths against that candidate and the 1,204 non-CLI base paths against `fc924df`. On the supported immutable Ubuntu 22/Clang 21 image with authenticated runtime `81298`, two fresh network-disabled builds produced 16 byte-identical regular files. The exact 8-global/2-export/83-string/34-source/36-formatter/54-native/157-undefined boundary, 35 public E1B AOT checks, six-command behavior/artifact/grammar/I/O tests, formatter-to-production negative, and genuine condition-5 handler-allocation plus published-shard rollback/static-fallback witness passed; root verified the 83-file evidence manifest. This accepts the bounded CLI3-A source and gate, not the root aggregate CI or full CLI3. Pretrain/evaluate/generate await their actual upstream APIs and end-to-end evidence | active |

The private G3-C4 ordinal-10 A2 candidate transaction is integrated from
`4f3f5e1` and the handle-pair correction `6e00c29` as `6d36e1b` and
`3ac66ce`. It stages genuine T1/T2/decode KH/VH into a frame-owned candidate,
dispatches the accepted causal-attention provider once, and leaves the
committed cache unchanged until a later frame commit. Supported normal,
repeat, and sanitizer runs each pass 15,760 checks across three routes and
50 allocation cuts; the malformed ordinal-11 handle-pair, provider failure,
borrow-blocked abort, predecessor hashes, Q0 isolation, and merged structural
gates pass. Root sealed the exact candidate evidence at
`g3c4-manual-a2-6e00c296-20260925/SHA256SUMS` (`d82b1804...`). The private
ordinal-11 attention merge from exact source `b02bcf3`/tree `863af66` is
integrated byte-identically at `f3c0c52`: accepted G3-N/N3K head-layout
merge writes frame-owned AT only after one successful dispatch while
preserving the pending A2 transaction. Supported normal/repeat/sanitizer
runs each pass 1,914 checks across T1/T2/decode; provider failure/retry,
candidate and committed-cache invariance, borrowed-logits abort, no added
global symbol, predecessor hashes, Q0, and merged structural gates pass.
Root sealed `g3c4-manual-at-b02bcf3-20260925/SHA256SUMS` (`c85e5d29...`).
The private ordinal-12 attention-out projection from exact source
`63757e1`/tree `e02abb4` is integrated byte-identically at `6c63c0f`:
accepted G3-N/N3K linear uses pinned W_o and writes frame-owned AO only
after a successful dispatch. Supported normal/repeat/sanitizer each pass
1,937 checks across T1/T2/decode, including provider failure/retry, exact
weight identity, staged A2 and committed-cache invariance, and borrowed-
logits abort. No new global symbol, predecessor hashes, Q0, and merged
structural gates pass. Root sealed
`g3c4-manual-ao-63757e1-20260925/SHA256SUMS` (`faafc832...`).
The private ordinal-13 attention residual from exact source `cc89863`/tree
`1ec8ade` is integrated byte-identically at `3823a5f`. It dispatches the
accepted G3-N/N3K residual with X first and AO second, publishes R only
after success, and preserves the pending A2 candidate. Supported normal,
repeat, and sanitizer each pass 1,946 checks across T1/T2/decode, including
bitwise X+AO, injected provider failure/retry, candidate/cache/logits/
binding/RNG invariance and borrowed-logits abort. No added global symbol,
predecessor hashes, Q0 and merged structural gates pass. Root sealed
`g3c4-manual-r-cc89863-20260925/SHA256SUMS` (`e5c3708e...`). The exact-base
private numerical tail `c398b2e`/tree `de73fcb` is integrated byte-identically
at `d653508`: ordinals 14–20 dispatch N2, FU, FG, FD, Y, NF, and tied-head Z
through the accepted G3-N/N2/N3K routes. Pinned normal, repeat, and sanitizer
each pass 10,765 checks across T1/T2/decode with 21 provider failure/retry
cuts; root independently reran the focused gate and verified predecessor
hashes, Q0, structural checks, and identical integration blobs. Sealed root
evidence is `g3c4-manual-tail-c398b2e-20260925/SHA256SUMS`
(`e3e04058...`). The private manual frame prepare/commit continuation from
`a374fc0`/tree `ae5318bb` and exact Q0 allowlist correction `7382b41`/tree
`64460236` are integrated byte-identically at `c99e684` and `0cc147b`.
T1/T2/decode prepare the last 256 logits, recheck pinned bindings and the
staged A2 transaction, then publish the cache/result through a terminal
commit and drain the active call. Independent pinned normal, repeat, and
sanitizer gates each pass 6,055 checks across four routes, including failure
and retry before commit; Q0 passes 4/4 and the merged focused gate passes.
Root sealed `g3c4-manual-frame-commit-7382b41-20260925/SHA256SUMS`
(`417948bd...`). The source-private native 21-role dispatcher
`1a4af6f`/tree `30ad88f` is integrated byte-identically at `fd3534c`.
It authenticates the active manual frame, requires the exact next ordinal,
and delegates to the accepted seven helper groups without owning a new
numerical route or cursor mutation. Independent pinned normal, repeat,
and sanitizer gates each pass 6,199 checks across T1/T2/decode, including
21 provider failure/retry cases; Q0 4/4 and the merged focused gate pass.
Root sealed `g3c4-manual-role-step-1a4af6f-20260925/SHA256SUMS`
(`6bd0b5db...`). The source-private Eshkol manual transport witness from
`4418a0c`/tree `e0953cdc` is integrated byte-identically at `55c9311`.
It composes the accepted C4 owner, call ledger, T1 input and 21 explicit
native role steps through frame prepare/commit, without claiming the authentic
G3-T M3T/C2 package root. Root's pinned normal/repeat/ASan+UBSan runs each
pass 36 checks across P1/P2 prefill, decode and three rollback cases; merged
static and Q0 isolation gates pass. Root sealed
`g3c4-manual-transport-root-4418a0c-20260925/SHA256SUMS`
(`a8a84bd1...`). Public Eshkol role stepping, generation loop,
output/text publication, and save/reload remain pending.
The authentic G3-T M3T/C2 model-admission precursor `59558ba`/tree
`6c9225bf` is integrated byte-identically at `78d3d6f`. It authenticates
the exact live M3T model entry, initializer lineage, fixed 14 P1 handles,
eval mode, and absence of an active workspace without enrolling a generator.
Root's pinned normal/repeat/sanitizer witness passes 24 checks, the source
closure and merged Q0 4/4 pass, and evidence is sealed at
`g3t-model-admission-root-59558ba-20260925/SHA256SUMS` (`ab03f8e1...`).
The source-private native G3-T generator-context leaf `5a8d6c5`/tree
`cfeccbf` is integrated byte-identically at `234a520`. It authenticates
the actual M3T owner, enrolls an empty A2 cache only after allocation,
holds the shared fixed-14 pins during acquired calls, and retains dead
context tombstones after close. Root's pinned merged normal, repeat, and
ASan/UBSan gates each pass 47 checks, with genuine model construction,
wrong-owner/policy rejection, allocation retry, same-model exclusion,
pin drain, and source/Q0 isolation. Evidence is sealed at
`g3t-native-context-root-234a520-20260925/SHA256SUMS` (`fcc2dcfe...`).
The source-private seeded G3-T constructor `824f91e`/tree `5a98260` is
integrated byte-identically at `f8c359e`. It authenticates the fixed M3T
model and raw V256 tokenizer, validates a bounded C2 seed policy, roots the
native context and closes idle generators. Root's pinned normal/repeat/
sanitizer gates each pass 53 checks, sealed at
`g3t-generator-constructor-root-f8c359e-20260925/SHA256SUMS`
(`2e411a88...`). A repeated sanitizer audit observed model-seal failures
before any constructor call in both predecessor (17/100) and candidate
(2/100) binaries; the candidate normal binary passed 100/100. P1's seal and
prepare-eval compared parameter schedule rows structurally, including opaque
native handle pointers. Eshkol deep equality could interpret a handle's nonce
as a bignum limb count. The isolated P1 repair compares handles by identity
and compares path, shape, dtype, device, and ties by value. Its pinned G3-T
constructor gate passes 53 checks, with 0/100 normal and 0/300
ASan/UBSan/LSan fresh-process failures. A focused P1 witness rejects a
different authentic handle with equal metadata and changed metadata with the
same handle. Audit traces are sealed at
`g3t-constructor-sanitizer-audit-824f91e-20260925/SHA256SUMS`; repair
evidence is at `g3t-model-seal-fix-20260925/SHA256SUMS`. Root reviewed the
opaque-handle identity contract and integrated the four implementation/test
blobs byte-identically at `cf5e394`; the verified 21-file seal is
`567a4d36...`.
The isolated [P1/G1 prefill and speculative-sample leaf](g3/G3_T_PREFILL_SAMPLE_LEAF.md)
adds a guarded 21-role source-private prefill, prompt/cache/model binding,
and a speculative G3-S greedy or categorical candidate. Its witness compares
all 256 logits bitwise with independent M3T and exercises precommit,
attention, and stale-profile rollback. A token frame, published output,
continuation binding validation, and public generation remain pending.
The reviewed candidate `c31611c`/tree `99194ff` passed 125 checks in each
normal/repeat/sanitizer witness, 20 fresh sanitizer launches, and bitwise
agreement of all 256 logits with independent M3T. Its sealed evidence is
`g3t-prefill-sample-p1-combined-c31611c-20260925/SHA256SUMS`
(`013a0fba...`). Root integrated the feature, test, and source checker at
`74cff23`, `736f2f0`, and `3496c0d`; the separate candidate P1 repair was
already integrated at `cf5e394`. All in-repo source-closure files match the
candidate except this roadmap; root reran the static checkers and Q0 4/4.
The isolated [sampled-token frame staging leaf](g3/G3_T_TOKEN_FRAME_LEAF.md)
executes the 21-role position-1 append under the committed P1 binding and
holds the A2 K/V append transaction speculative. Its focused witness checks
all 256 position-1 logits bitwise against an independent M3T two-token
forward pass, rejects stale profile and wrong frame/order, and proves failed
attention and abort preserve the prompt cache and pre-call RNG. Final output
readiness, text decode, joint cache/ID/RNG publication and public generation
remain downstream.
Candidate `4a82887` and root integration `02c7b3f` have identical tree
`d4792525`; root source review found no concrete defect. The pinned normal,
repeat and ASan/UBSan/LSan witnesses each pass 146 checks with identical
stdout and empty stderr; Q0 passes 4/4. The exact-tree evidence is sealed at
`g3t-token-frame-root-02c7b3f-20260925/SHA256SUMS`.
The isolated [G1 numeric output preparation leaf](g3/G3_T_OUTPUT_PREPARE_LEAF.md)
reserves exact I1 `[1]` ID storage and copies the completed speculative token
there before recording lengths and successor RNG. Abort checks active I1 borrows
before unwinding the frame and scrubs the unpublished output. G0 admission,
ID-byte staging, T1 decode, text readiness and joint publication remain pending.
Candidate `64c613d` is integrated at `e8d38ea` with byte-identical native
transport; normal/repeat/ASan+UBSan+LSan witnesses each passed 170 checks,
and the root static source contract passes. Evidence is sealed at
`g3t-output-readiness-64c613d-20260925/SHA256SUMS`.

The isolated [G1 ID/raw-text readiness leaf](g3/G3_T_OUTPUT_TEXT_LEAF.md)
stages the pending output's I1 byte ID, decodes it through the authentic
same-package T1 raw decoder, and accepts the decoded byte with once-only
readiness and binding checks. The focused normal/repeat/ASan+UBSan+LSan gate
passes 56 checks per run in the pinned Ubuntu/LLVM21 image. Candidate
`ee29b72` plus the exact Q0 admission correction `727eecd` are integrated at
`0f2fdf7`/`a0b6adc`. Root verified Q0 4/4, the source closure, identical
normal/sanitizer output, and the sealed evidence at
`g3t-output-text-727eecd-20260925/SHA256SUMS` (`48513349...`).

The isolated [G1 final publication leaf](g3/G3_T_FINAL_PUBLICATION_LEAF.md)
adds final frame preparation, resource preflight, joint cache/ID/RNG/output
commit, immediate call finish, and independently releasable output ownership.
Candidate `2355d2c`/tree `a43a709` is integrated byte-identically at
`017a213`. The pinned normal/repeat/ASan+UBSan+LSan witnesses each pass 118
checks with identical stdout and empty stderr. Root reviewed source and tests,
reran the merged-tree pinned normal witness (118 checks), static contracts and
Q0 4/4, and sealed evidence at
`/home/gabe/.codex/evidence/eshkol-transformer/g3t-final-publication-2355d2c-20260925/SHA256SUMS`
(`eb7f3536...`). At that gate, G0, public facade/accessors, generation loop
and CLI remained open.

The isolated [P1/G0 zero-budget leaf](g3/G3_T_ZERO_BUDGET_LEAF.md) runs the
authentic 21-role one-byte prefill and commits a prompt-only A2 cache with a
live empty-ID/raw-text output and unchanged RNG. It preallocates output
ownership before prefill, rejects token-frame/sample/draw paths, and retains
G1's binding, cleanup, tombstone and source-private closure contracts. The
pinned f31/LLVM21 network-none normal/repeat/ASan+UBSan+LSan witness passes
122 checks with identical stdout and empty stderr; all 256 prefill logits
match independent M3T first-row bits. Static source contracts and Q0 pass;
sealed evidence is
`/home/gabe/.codex/evidence/eshkol-transformer/g3t-zero-budget-g0-20260925/SHA256SUMS`.
Candidate `2413189` is integrated as byte-identical source at
`423ef0f`/`da61f54` after independent source/test review. The merged P1
active-record + G0 full pinned gate passes normal/repeat/ASan+UBSan+LSan,
122 checks each with identical stdout, empty stderr, Q0 4/4, and five
static source contracts. Root sealed the merged witness at
`/home/gabe/.codex/evidence/eshkol-transformer/g3t-zero-budget-merged-da61f54-git-20260925/SHA256SUMS`
(`8ee1b6c3...`).
Public facade/accessors, P2 input, generation loop and CLI remain open.

The isolated [P2/G0 zero-budget leaf](g3/G3_T_P2_ZERO_BUDGET_LEAF.md)
adds an exact two-byte-ID input and a read-only `P+G<=2` admission before
pinning. Its genuine 21-role T2 prefill stages both K/V positions in A2 and
publishes the second-position logits, cache length two, empty output, and
unchanged RNG through the existing private G0 tail. P2/G1 and length above
two reject before pins. Supported f31/LLVM21 network-none normal, repeat and
ASan/UBSan/LSan each pass 201 checks with byte-identical output and empty
stderr, including the G1 route compiled into the P2-enabled native object.
The inherited exact-checkout G1 publication witness passes 118 checks. All
six G3-T source contracts and Q0 4/4 pass. This source-private candidate
adds no public facade, accessor, generation loop, or CLI.
Root independently reviewed source and tests, verified the sealed candidate
and G1 evidence (`2b282e30...` and `52f55d93...`), and integrated identical
source at `490b1fd`. Once the new Python source checker became Git-tracked,
Q0 correctly required its exact development-only admission; `c245550`
adds that admission, and the merged Q0 4/4 and P2 static contract pass.

The isolated Eshkol F32 numeric safety leaf `5091032b`/tree `71f6897` is
reviewed and composed onto the provisional compiler/runtime line at
`8b937b67`, with roadmap `95630bc6`. Native LCM and VM GCD/LCM/modulo/
quotient reject F32 before unsafe or unreviewed integer-domain paths; the
supported focused Release and sanitizer gates each pass 6/6 on the combined
source. Root sealed the combined logs at
`f32-integer-rational-combined-8b937b67-20260925/SHA256SUMS`
(`89fe8f98...`). The reviewed VM modulo/quotient parity leaf
`b9f90ec9`/tree `90c0a2c` is composed byte-identically on that line at
`527f9fb1`, roadmap `837b7677`: stored DOUBLE paths now match native
floored-modulo and truncated-quotient values/result kinds, and direct/stored
VM canonical F32 uses the checked inexact route. All-INT64/bignum controls
remain pinned; exact-tree Release and sanitizer gates each pass 8/8,
sealed `f32-modulo-quotient-parity-20260924/SHA256SUMS` (`6f9b0041...`).
The exact-base VM inexact remainder-zero correction `b74ab5f`/tree `fa2e455`
is composed byte-identically at `8afd55e`: VM direct/stored F64 and canonical
F32 now raise a catchable error on ±0 divisor like native, while exact
INT64/bignum zero retains its historical fatal VM boundary. Pinned Release
and ASan+UBSan+LSan focused gates each pass 10/10; root verified
`f32-remainder-zero-parity-20260924/SHA256SUMS` (`7e8f6f29...`). The
two-commit `numerator` successor `17289866`/`31ed68f2` is composed
byte-identically at `7bd9cbb7`/`b143176f`; `4e483c89` corrects a stale
inventory row. The VM now matches native non-rational result kinds without
unsafe float-to-int conversion, while native and VM direct/stored canonical
F32 return the checked inexact DOUBLE promotion. Independent pinned Release
and ASan+UBSan+LSan focused tests each pass 7/7 across ABI, VM and AOT/JIT
O0/O2; root verified `f32-numerator-parity-31ed68f2-20260924/SHA256SUMS`
(`3a3999de...`). This is native compatibility, not full R7RS inexact
fraction semantics. GCD/LCM integer-domain policy, other operation gaps,
whole-F32 acceptance and the transformer runtime repin remain pending.
The sealed no-change `gcd`/`lcm` audit on provisional Eshkol `4e483c89`
found that native direct/stored DOUBLE GCD already disagree, both native
and VM truncate fractional DOUBLE inputs, bignum paths diverge, and VM
sanitizer catches an out-of-range float-to-int cast at `gcd 1e300 4.0`.
F32 remains explicitly rejected on these operations until the non-F32
integer domain, result kind, direct/stored parity, and overflow contract
are repaired. Root verified
`f32-gcd-lcm-domain-blocker-4e483c89-20260924/SHA256SUMS`
(`5aa474e5...`). The bounded VM safety prerequisite `46e97186`/tree
`21cf93e` is composed byte-identically on the provisional runtime line
at `312af5e0`: VM direct/stored GCD/LCM now raise catchable errors before
coercing invalid types, fractional/nonfinite/out-of-range DOUBLE,
`INT64_MIN`, unsupported bignums, or overflowing LCM. Valid supported
INT64 behavior and F32 refusal remain; independent pinned Release and
ASan+UBSan+LSan focused gates each pass 2/2. Root verified
`f32-vm-gcd-lcm-guards-46e97186-20260924/SHA256SUMS`
(`b56b0c32...`). Native parity, wide integer semantics and inexact result
kind remain open. The bounded native safety leaf `fcfb14a6`/tree `f31d63f`
is composed byte-identically on the provisional runtime line at
`09e65bf8`. Raw, tagged/stored, and dual-primal GCD/LCM reject invalid
DOUBLE/type/magnitude before conversion, integral stored GCD no longer
fabricates zero, and LCM rejects int64 overflow. Independent pinned
Release and ASan+UBSan+LSan JIT/AOT O0/O2 source matrices pass, as do the
four host-bit F32 route tests in each build; root verified
`f32-native-gcd-lcm-guards-fcfb14a6-20260924/SHA256SUMS`
(`1ce5f97c...`). The subsequent VM exact-wide GCD leaf
`95697c0f`/tree `985b003a` is composed byte-identically on the provisional
runtime line at `f748077a`. VM binary direct/stored GCD now uses the
existing bignum kernel for all-exact INT64/bignum operands, normalizing a
fitting answer back to INT64; mixed wide/inexact operands explicitly reject.
Independent pinned Release and ASan+UBSan+LSan CTest each pass 4/4, with
worker JIT/AOT O0/O2 parity in both builds. Root verified
`f32-gcd-lcm-result-contract-95697c0f-20260924/SHA256SUMS`
(`3487a9c3...`). The bounded non-F32 result-kind leaf `e460d26f`/tree
`8bf82ecd` is composed byte-identically at provisional Eshkol `861f4398`:
accepted integral DOUBLE GCD/LCM now yields inexact DOUBLE across native/VM
direct/stored routes, while exact INT/bignum stays exact. Both substrates
reject mixed bignum/DOUBLE GCD, including the huge-integer/zero case that
previously split into NaN versus infinity. Root's pinned Release and
ASan+UBSan+LSan focused CTest each pass 8/8; worker source JIT/AOT O0/O2
matrices pass 16/16. Root verified
`f32-gcd-lcm-inexact-result-e460d26f-20260925/SHA256SUMS`
(`4e824bea...`). The reviewed exact-wide LCM leaf `4e7f26bd`/tree
`5af729e` is composed byte-identically at provisional Eshkol `e8fbb90b`.
Native tagged and VM binary direct/stored routes now compute exact
INT64/bignum LCM for all-exact inputs with a wide peer, including zero,
negative values, and INT64_MIN mixed with 2^70; a missing native tagged
INT64/INT64 comparison case is repaired. Worker pinned Release and
ASan+UBSan+LSan focused CTest each pass 8/8, with 16 native JIT/AOT O0/O2
fixture executions. Root reviewed the source, verified the exact-tree seal
`f32-exact-wide-lcm-4e7f26bd-20260925/SHA256SUMS` (`9ab09ac5...`), and
repeated both pinned focused CTest groups at 8/8. INT64-only overflow still
rejects, mixed wide/DOUBLE and F32 still reject, and allocation faults were
not injected. At that exact LCM commit, native/VM arity and AD policy,
whole-F32 acceptance, and transformer runtime repin remained pending.
The bounded non-F32 variadic/AD prerequisite `065766e0` then `c8069863`/tree
`f888ff27` is composed byte-identically at provisional Eshkol `d432dc11`.
Native stored and VM direct/stored GCD/LCM now accept zero, one, and three or
more operands without truncation; native duals explicitly reject instead of
fabricating a zero tangent. The first-class native wrapper reads the complete
16-byte tagged cons car at offset zero. Root's merged pinned LLVM21 Release
and ASan+UBSan+LSan focused CTest each pass 17/17, plus six native JIT O0/O2
fixtures in each mode. Evidence is sealed at
`f32-gcd-lcm-root-d432dc11-20260925/SHA256SUMS` (`0eed9b20...`). Canonical
F32 is still rejected; all-INT64 `INT64_MIN`, mixed wide/inexact, reverse-tape
AD-node policy, whole-F32 acceptance, and transformer repin remain open.
The bounded canonical F32 GCD/LCM admission `8cb2c0d8` and root correction
`58cc9a3f` are composed on the provisional Eshkol branch at `e459fbab`/tree
`d1da01d`. Native JIT/AOT and VM direct/stored routes accept finite integral
F32 within the signed-int64 magnitude domain and return inexact DOUBLE;
fractional, nonfinite, out-of-range, malformed, mixed exact-wide/inexact, and
AD inputs reject. The root correction removes two hidden zero fallbacks when
F32 promotion cannot be emitted. Root's pinned LLVM21 Release and
ASan+UBSan+LSan focused CTest pass 16/16 each; the sanitizer runtime tests
retain leak detection, while AOT fixture compilation disables it for known
frontend parser allocations. Evidence is sealed at
`f32-gcd-lcm-root-e459fbab-20260925/SHA256SUMS` (`1a2b3551...`). Broader
Eshkol CI and the transformer runtime repin remain pending.

The independently reviewed [TR3-C private snapshot lease authority](TR3_C_SNAPSHOT_LEASE_AUTHORITY.md)
`00d17cb`/tree `00a0091` is integrated as identical runtime/test source at
`c35bf99`. Exact-parent enter/recheck/abort/finish uses existing trainer slot 4;
supported f31 strict O0 AOT and 110 runtime checks with a 1,024-iteration
zero-growth horizon pass. Root verified
`tr3-c-snapshot-lease-00d17cb-20260924/SHA256SUMS` (`c2a64b13...`), and
merged source contracts pass 26/26. The independently reviewed
[private snapshot composer](TR3_C_SNAPSHOT_COMPOSER.md) is integrated from
`4534b21`/tree `d5cad25` with genuine P1/O2/D2/X1/C2 seams, request-led
ownership transfer, ordered failure cleanup, source preservation, and retry.
Root verified `tr3-c-snapshot-composer-final-4534b21/SHA256SUMS`
(`3cf350f...`), 18/18 focused source and lease contracts, and pinned f31
strict O0 AOT runtime with 27 checks and 42 tensors. Public trainer state,
serialization, restore invocation and resume acceptance remain pending.

The [TR3-C private root closure](TR3_C_PRIVATE_ROOT.md) now source-composes the
accepted lease, snapshot, C2 SAVE/LOAD and result-cell restore under one
identity. Its pinned internal-object gate fixes the 46-source/334-undefined
boundary and exercises a real construct/snapshot/file/restore continuation.
The canonical shared-library package path still fails in the pinned compiler
at `Tail transfer: no public entry` for accepted private functions; bridge,
localization and public trainer construction remain blocked on that boundary.
The documented private `--shared-lib -c` object flavor compiles the same root
with an exact 46-source/2,593-defined/330-undefined boundary and no public
trainer or C ABI thunk. A focused negative gate reproduces all 52 linked-mode
tail-transfer failures. The pinned compiler must bind each tail-body forwarder
to its renamed internal implementation before a C-callable package can be
admitted; public trainer and resume claims remain pending.
The reviewed isolated Eshkol `97c40c9d` fix now permits a
[linked private TR3 package boundary](TR3_C_LINKED_PRIVATE_PACKAGE.md) without
changing the transformer pin. The bounded gate composes the accepted 46-source
root and exact 29-object/66-path native closure, localizes all but the
versioned private initializer bridge, links with no unresolved trusted
authority, and rejects hostile external links to the raw initializer and
private TR3/native symbols. Its linked `.so` now owns the AOT-style
shared-arena setup and E1B-style exception/parallel scope; real `dlopen`
invocation and repeat pass. An injected initializer fixture proves failure
status, raised-value and handler cleanup, reentrant busy, and retry. Real TR3
initializer failure retention is unmeasured, and no C-callable trainer bridge
is accepted; public trainer and resume acceptance remain pending. The
source `4675500`/tree `b5bd6d0` is integrated byte-identically at `675f1de`;
root verified `tr3-dynamic-init-bridge-4675500-20260924/SHA256SUMS`
(`0bf88f01...`) and exact integrated source blobs.
The [private C trainer entry audit](TR3_C_PRIVATE_TRAINER_ENTRY_BLOCKER.md)
finds the real five-operand lease constructor in the compiled root, but no
same-package C producer for its operands. The
[private lease-unenroll seam](TR3_C_PRIVATE_LEASE_UNENROLL.md) removes one
authenticated idle trainer record and returns its five receivers to their
callers without destroying them. A C construct/handle contract remains
pending; no public trainer or resume claim follows from unenrollment.
The follow-on [producer/ownership decision](TR3_C_PRIVATE_TRAINER_ENTRY_BLOCKER.md#producer-and-c-ownership-decision-at-ccf1ab4)
maps all five genuine X1/T2/D2/M3T/O2 source producers. Their separate
contracts do not define one data-only C request, child ownership/cleanup, or
trainer handle/status ABI; CLI3 deliberately leaves `pretrain` flags undefined.
The linked package therefore retains only its initializer export until those
contracts are accepted.
The [private factory/close contract](TR3_C_PRIVATE_FACTORY_CONTRACT.md)
specifies a one-attempt data-only request, X1-derived M3T seed, D2 corpus
identity, fixed O2 paths, failure retention and exact C handle/status behavior.
The isolated implementation now composes authentic same-package producers
through candidate-only `tr3_c_private_factory_root.esk`; the production root
and initializer-only export manifest remain unchanged. On `b0cd341`, the
production linked gate passes its exact 46-source/29-native boundary with two
dynamic-defined symbols (`tr3-linked-production-b0cd341-20260925`), while
the candidate linked gate passes 48-source/30-native construction, lifecycle,
negative probes and four candidate dynamic-defined symbols
(`tr3-factory-candidate-b0cd341-20260925`). Both evidence sets are sealed.
The distinct production `--emit-object` and `--shared-lib -c` gates pass on
`436efad` after exact P1 hash-symbol repins; the latter also pins three
preexisting lease-unenrollment definitions and retains its expected linked
tail-transfer negative. Sealed evidence is `tr3-private-object-436efad-20260925`
and `tr3-private-mode-436efad-20260925`.
The bounded `28cc85a` candidate linked rerun now proves a valid-X1 fixed
profile rejection (`unsupported/X1`), create/close at both X1 seeds 1729 and
2718 through M3T's native initializer-seed check, and an early D2 batch-limit
cut (`invalid-argument/D2`) before filesystem access. Exact symbols and the
candidate export boundary stay unchanged; sealed evidence is
`tr3-factory-profile-seed-d2-28cc85a-20260925`. Model-state bytes and
independent T2 fault injection were not observed. Other producer/lease cuts,
cleanup-failure behavior, process-lifetime retention, broader E1 category
evidence and an exact merged-union gate remain pending. Root independently
reviewed the prior bridge, source wrapper, linked
scripts, fixtures, runtime lifetime and sealed evidence without finding a
blocker. Public trainer/resume status is unchanged.
The [private single-update composition leaf](TR3_STEP_PRIVATE_COMPOSITION.md)
now combines accepted lease, D2/M3/L2/L3S numerator VJP, and O2
update-and-clear source in one test-only aggregate. Its pinned f31/LLVM21
witness and independent PyTorch checker prove one fixed-profile two-microbatch
update with wrong-weight rejection, abort/retry, gradient clear, and a numerical
key-weight comparison. Source `365a180` plus audit `e461cc2`/tree `cbf8ca09`
are integrated at `d341ae4`/`07ebba1`, with formatting `d821900`. Root reran
the exact merged pinned gate: 21 static loads, 41 runtime checks, key-weight
postupdate maximum absolute difference `1.86e-9`, and a rejected nonfinite
mutant; the 1:26 AOT build peaked at 4,402,264 KiB. Root sealed
`tr3-step-leaf-root-d821900-20260925/SHA256SUMS` (`05bf6a62...`). Trainer
phase/counter publication is still test-owned;
whole-step rollback, EOS replay, metrics, public `trainer-step!`, and resume
acceptance remain pending. The follow-on [private pre-write transaction leaf](TR3_STEP_PRIVATE_TRANSACTION.md)
now binds the fixed-profile step-start cursor, RNG, train-mode witness, and
counters to an authenticated rooted frame, rejects malformed/busy entry, and
supports D2/gradient rollback and O2 prepare/abort before parameter writes.
It does not own the no-fail post-write trainer-control tail or public step.
The bounded [private no-EOS commit tail](TR3_STEP_PRIVATE_COMMIT_TAIL.md)
stages exact token/update controls and frame retirement before O2's accepted
first-write boundary, then performs only primitive no-fail writes after O2
success. Its pinned genuine two-microbatch numerical witness passes, while
EOS/epoch replay, whole-step cleanup beyond the fixed profile, metrics,
public trainer, and resume remain pending.
The [private fixed-profile no-EOS step composer](TR3_STEP_PRIVATE_COMPOSER.md)
now owns the two D2/M3 microbatches, transient releases, pre-write failure
abort, O2 prepare, and the accepted no-fail commit tail in one source-private
operation. Its injected early/mid/precommit and same-trainer retry witness is
bounded to context/accumulation two; EOS replay, general trainer metrics,
public trainer packaging and resume remain pending.
Source `1d0057f`/tree `7a1b3cf` is integrated byte-identically at `f5649ab`.
Root's pinned merged f31/LLVM21 network-disabled gate passes 20 compiled
checks with 27 source loads: early, live-graph midstep, and post-O2-prepare
failure cleanup, same-trainer retry, genuine two-microbatch commit, and an
explicit EOS negative. The independent PyTorch key-weight maximum absolute
error is `1.86e-9`; a nonfinite mutant is rejected. Evidence is sealed at
`tr3-step-composer-root-f5649ab-20260925/SHA256SUMS` (`b78043c9...`).
The follow-on source-private [finite-D2 EOS composer extension](TR3_STEP_PRIVATE_COMPOSER.md)
stages epoch rewinds and proves the accepted global update/epoch/cursor
equation before the first O2 write. Its genuine one- and two-row fixtures
exercise EOS before and between microbatches, multiple boundaries per update,
empty-D2 constructor rejection, and rollback after rewind. Public trainer,
streaming EOS, metrics, C ownership, and full resume equivalence remain pending.
Source `848a143`/tree `45462cb` is integrated byte-identically at `ba726a1`.
Root's pinned merged f31/LLVM21 network-disabled composer gate passes 33
compiled checks with 27 source loads, including an independent two-update
PyTorch key-weight trajectory (maximum absolute error `1.86e-9`) and a
nonfinite mutant rejection. The worker's exact-tree transaction and commit-tail
gates pass 63 and 91 checks. Root evidence is sealed at
`tr3-finite-eos-root-ba726a1-20260925/SHA256SUMS` (`0e092f6a...`).
The bounded [private checkpoint trajectory witness](TR3_PRIVATE_RESUME_TRAJECTORY.md)
now composes accepted TR3-C SAVE, C2 LOAD/joint restore, and the finite-D2
step composer in one source identity. At `K=1, R=1`, a fresh restored trainer's
next EOS-crossing update matches the uninterrupted trainer's 42 tensor images,
RNG/cursors/counters, and final canonical C2 bytes; corruption, X1 mismatch,
and post-rewind retry are negative-tested. The longer supported resume gate
in the TR3-C contract and public trainer remain pending.
Source `9efdc65` is integrated with byte-identical implementation and test
files at `c8b9e94`. Root's merged pinned f31/LLVM21 network-disabled gate
passes 77 compiled checks from the 42-source closure; the uninterrupted and
restored final canonical C2 files match byte-for-byte. Evidence is sealed at
`tr3-private-resume-root-c8b9e94-20260925/SHA256SUMS` (`025e2705...`).
The private successor `172bb8f` is integrated as source at `16c3c47` and
documentation at `c07068a`; it generalizes the finite-D2 composer
to positive configured accumulation with an explicit exact-f32 weight bound.
Its pinned f31/LLVM21 network-disabled compiled trajectory gate passes 167
checks over `A=1,2,3`, `K=1`, and `R=3`: each resumed suffix crosses EOS,
all intermediate 42-tensor/control images and final C2 bytes match, and
injected prewrite failure preserves the checkpoint image for retry. The
`A=1` and `A=3` cases have unequal active-token weights across updates.
The unchanged fixed-`A=2` composer gate also passes 33 compiled checks,
including independent PyTorch numerical and two-update EOS trajectory
oracles (maximum key-weight error `1.86e-9`) and nonfinite mutant rejection;
its seal is `tr3-general-composer-172bb8f-20260925/SHA256SUMS`
(`836d0cf3...`). Resume evidence is sealed at
`tr3-general-resume-172bb8f-20260925/SHA256SUMS` (`a64c79a6...`).
Root's merged pinned composer gate repeats 33 checks and both independent
numerical oracles, sealed at
`tr3-general-composer-root-16c3c47-20260925/SHA256SUMS` (`bdb1a4e5...`).
Root's merged trajectory gate repeats 167 checks and final C2 byte equality,
sealed at `tr3-general-resume-root-c07068a-20260925/SHA256SUMS`
(`b75ff3a1...`).
The [private fresh-process continuation gate](TR3_FRESH_PROCESS_RESUME.md)
runs separate uninterrupted, K-checkpoint producer, and fresh LOAD/restore
processes for `A=1,2,3`, `K=1`, and `R=3`. It compares canonical C2 bytes at K,
after restore, and after every suffix update, and checks corruption, X1
mismatch, and prewrite failure followed by exact retry. Public next-step
metrics/effective-rate observations and the remaining TR3-C §13
failure/environment gates remain pending.
The pinned f31/LLVM21 network-disabled run on clean `15a7c8a`/tree
`68b2d56` passed 246 checks in nine distinct processes and 15 C2 byte
comparisons; its seal is
`tr3-fresh-resume-15a7c8a-20260925/SHA256SUMS` (`f11dc3a6...`).
Root independently reran the same source/tree with the same 246 checks and
15 byte comparisons; the verified 87-file root seal is
`tr3-fresh-resume-root-d835a51-20260925/SHA256SUMS` (`1967e564...`).
The bounded [private TR3-C retention diagnosis](TR3_C_RETENTION_CYCLE_DIAGNOSIS.md)
passes pinned f31/LLVM21 ASan/UBSan/LSan tests at 32 and 128 genuine
snapshot→C2 SAVE/LOAD→joint restore cycles after one P1/C2 identity prewarm,
including pre-seal failure/preservation/retry. Native and Scheme live
authority counts stay flat, while F32 retired controls grow exactly 17,552
bytes per cycle, alongside linear P1/O2/C2/I2 terminal records. Section 13's
1,024/8,192 live-authority horizons remain unproved; the accepted joint plan
already permits linear terminal records and makes no flat-total-memory claim.
Clean `cd2899c`/tree `81a9980` evidence is sealed at
`tr3-retention-cycle-cd2899c-20260925/SHA256SUMS` (`b8280e90...`).
Root attempted the 1,024-cycle ASan horizon on that sealed candidate after
integration. It timed out at 1,200 seconds before the horizon (exit 124,
peak RSS 1,334,512 KiB), so it gives no 1,024-cycle acceptance evidence;
the bounded 32/128-cycle results above remain the measured limit. The
attempt is sealed at `tr3-retention-1024-integrated-20260925/SHA256SUMS`.
The isolated test-only timing successor `b9824e9` passes pinned sanitizer
32/128-cycle gates; at 128, genuine SAVE takes 37.721 s and LOAD/joint
restore 32.418 s of 71.63 s wall. An instrumented 128-cycle profile identifies
the largest self-time hotspot as an append-only P1 shell-registry scan invoked
through the accepted fixed-set recheck during measured joint restore (15.58
of 65.95 sampled CPU seconds). Evidence is sealed at
`tr3-retention-phase-b9824e9-20260925/SHA256SUMS` (`392787c0...`) and
`tr3-retention-gprof-b9824e9-20260925/SHA256SUMS` (`4639f62b...`). A
test-only partition or terminal-registry reset would invalidate the single-
process retention horizon. The 1,024/8,192 live-authority gate remains open;
the next dependency is an accepted runtime identity-indexing contract or a
measured larger runtime budget.
Root integrated that timing and diagnosis at `dc965bc`/`7acfd66`, without
changing the accepted SAVE, LOAD or joint restore path.
An isolated P1 identity-index candidate keeps the append-only shell
registry authoritative and indexes published native shell pointer bits for
exact-identity lookup. It changes no public or native API. Pinned f31/LLVM21
sanitizer retention passes 32/128 genuine cycles with 91/283 checks, flat live
authority counts and exact images; 128-cycle wall time is 54.90 seconds versus
the preceding 71.63-second diagnosis. Focused P1 structural, registry, and
construction AOT tests pass 419, 169, and 23 checks. The canonical generated
root check passes. Root integrated the index at `6027c01`/`223085f`/`07dbc6e`.
The pinned registry-publication IR and poison proof passes after both
publication paths index only the canonical post-barrier shell (`3b81bd5`).
Its first direct 1,024-cycle reuse of the same sanitized binary hit the
pinned runtime's default 1,024 MiB heap ceiling and the 1,200-second time
bound, so it emitted no final census. With `ESHKOL_MAX_HEAP=4G` and a
2,400-second bound, that exact binary passed 1,024 genuine cycles and 2,075
checks: flat live authorities, exact final images, 1,251,758,464 arena bytes,
728.239 seconds SAVE, 872.951 seconds LOAD/restore, 26:45.61 wall, and
2,139,608 KiB peak RSS, with no sanitizer diagnostic. The 8,192 horizon and
practical runtime budget remain open. The full pinned P1 package gate exposed
pre-barrier shell reuse in the original index; the canonical-readback fix
passes focused proof, with a fresh exact-head package gate pending. Evidence is sealed at
`/tmp/p1-index-retention-evidence/SHA256SUMS` (`c6d91e9f...`) for the
default-limit attempt and
`/home/gabe/.codex/evidence/eshkol-transformer/p1-index-retention-1024-4g-20260925/SHA256SUMS`
(`c090b6cd...`) for the passing 4 GiB run. The canonical-readback focused
proof is sealed at
`/home/gabe/.codex/evidence/eshkol-transformer/p1-postbarrier-3b81bd5-20260925/SHA256SUMS`
(`b147066a...`); the 1,024-cycle result predates this repair.
Source `0fdefc8`/tree `df5ccd9` is integrated byte-identically at `4d56381`.
Root's pinned merged gate passes 91 runtime checks with 25 source loads, exact
cursor/RNG/counter publication, key-weight PyTorch maximum absolute error
`1.86e-9`, and a rejected nonfinite mutant. Evidence is sealed at
`tr3-step-commit-tail-root-4d56381-20260925/SHA256SUMS` (`ba79c74d...`).
Source `25673ef`/tree `5cd4abb` is integrated byte-identically at `c7cdf00`.
Root's pinned merged gate passes 63 runtime checks with 23 source loads;
`tr3-step-transaction-root-c7cdf00-20260925/SHA256SUMS` is sealed at
`e6e852da...`. The source-private frame is mutable, so a public trainer
must keep its saved controls opaque or seal them before exposure.
The exact private source `8a29dcf`/tree `43bff41` is integrated with
identical runtime, test, and script blobs at `cc84b49`. The worker's pinned
normal 1,024/8,192 and sanitizer 1,024 runtime gates each pass 133
unenrollment checks, with head/interior/tail unlink, reentry, failure,
re-lease, and constant registry-count cases; the full linked package and
hostile-link gate also passes. Root independently reran the pinned runtime
gate with the same counts and verified all 56 evidence files at
`tr3-private-lease-unenroll-root-8a29dcf-20260925/SHA256SUMS`
(`663c2825...`). Worker final mapping is sealed at
`tr3-private-lease-unenroll-final-8a29dcf-20260924/SHA256SUMS`
(`7eb65cae...`). Physical arena reclamation and parallel registry races
remain unproved.
The source-backed audit `6293a30` is integrated at `858ce0d`, with seal
`tr3-private-trainer-entry-blocker-6293a30-20260924/SHA256SUMS`
(`c52bba5d...`); it adds no runtime entry.

The isolated root integration tree `463a06b` passed its combined supported
E3-private/CLI3-A package and focused runtime gate in the immutable f31
Ubuntu 22.04/LLVM 21 image with networking disabled. E3 private package policy,
localization, hostile link and poisoned-runtime checks passed, with the selected
metric-bit accessor linked `LOCAL HIDDEN`; the root-composed CLI package retained
the 8-global/2-export/83-string/34-source/54-native/157-undefined boundary and
passed six-command, grammar, artifact, I/O-fault and formatter-negative tests.
The E3 artifact used the recovered canonical d5c compiler; the CLI artifact used
the accepted ordinary 81298 compiler. Exact artifacts, logs, inputs and verified
hashes are recorded in
`/home/gabe/.codex/evidence/eshkol-transformer/e3-cli3-root-union-4f6e81b/RESULTS.md`.
This accepts only the bounded root package/runtime union; registered aggregate
CI, public E3 and full CLI3 remain pending.

The bounded [P1 post-index retention profile](P1_POST_INDEX_RETENTION_PROFILE.md)
on `07dbc6e` passes pinned f31/LLVM 21 sanitizer/gprof 128/256 genuine-cycle
gates with 283/539 checks, exact images, and flat live authorities. Wall time
grows 60.03 to 147.59 seconds for twice the cycles; state/module registry
scans and raw-shell/native record lookups all grow superlinearly. No P1 runtime
optimization is accepted from this profile. An active-only record index needs
a publication/release/rollback and authoritative-fallback contract before
implementation; the 8,192-cycle horizon remains open.
An isolated [active-record candidate](P1_ACTIVE_RECORD_INDEX_CANDIDATE.md)
implements that contract for state/module ownership scans. Focused pinned
publication/rollback checks pass 17/23/169 cases; sanitizer 128/256 genuine
cycles pass 283/539 checks in 49.53/112.01 seconds, compared with the older
cross-commit 60.03/147.59-second profile. Root reviewed source and tests and
integrated identical trusted source at `6caae36`. E3/TR3/prepared structural
hash pins were repaired to the separately reviewed E3 active-list,
construction-handle identity, indexed-shell runtime symbols, and I2 rollback
changes; all three checks pass. The full pinned P1 package gate on immutable
`8b5b125` passed with the supported 900-second per-compile allowance:
E1B-integrated packaging, structural/native/registry checks, root-publication
IR and poison proof, sanitizers, negatives, atomicity and determinism. It took
3370.85 seconds, peaked at 3,958,628 KiB RSS, and is sealed at
`p1-full-package-8b5b125-900-20260925/SHA256SUMS` (`ab2fbb1e...`). The
8,192-cycle resource horizon remains open; the cross-commit timing comparison
is not an exact-base performance proof.
Hosted CI run `36125833885` on PR #126 exposed three new P1 hash-table
runtime references in the I2 aggregate that its exact undefined-symbol
manifest did not yet admit. The standalone P1 object at `76299fd` matches
its 110-symbol manifest. The 13 source aggregates that load P1 now each
admit those same three runtime references. The exact I2 aggregate rebuilt
at `8dc1e97` and its 157-symbol generated evidence and fresh `nm -u` output
match the manifest byte for byte; sealed evidence is
`i2-index-symbols-8dc1e97-20260925/SHA256SUMS` (`0c20051f...`). Other
aggregate and CI verification remain pending. The local full P1 gate reached trusted test
compilation, then hit its default 360-second per-compile timeout. Its
earlier package and sanitizer gates passed; a full 900-second-bound rerun
remains open.

## Wave 4 — practical pretraining and performance

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| MP4 | True f32 plus f16/bf16 policy, master weights and loss scaling | TR3, K1 | Numerical and storage-size evidence; overflow recovery tests | planned |
| AC4 | Activation checkpointing and memory-bounded training | TR3 | Same update within tolerance and measured memory reduction | planned |
| AMD4 | AMD HIP/rocBLAS backend and device observability | K1, TR3, B0 | Direct device execution, parity and benchmark evidence | planned |
| PA4 | Optimized/fused attention path | A2, B0 | Parity plus measured speed/memory improvement | planned |
| DATA4 | Filtering, deduplication, provenance and dataset reports | D1, T2 | Deterministic reports and contamination/provenance metadata | planned |

## Wave 5 — ecosystem expansion

| ID | Workstream | Depends on | Acceptance evidence | Status |
|---|---|---|---|---|
| IO5 | Safe import/export for safetensors and GGUF | C1, M3 | Cross-format round trips and malformed-file tests | planned |
| MOD5 | RMSNorm/SwiGLU/GQA/MQA/ALiBi/sliding-window model variants | M3 | Per-variant parity and training smoke tests | planned |
| DIST5 | Data-parallel training and collective abstraction | TR3, K1 | Multi-process deterministic smoke and failure handling | planned |
| EVAL5 | External evaluation adapters and contamination checks | E3, T2 | Reproducible benchmark manifests | planned |
| DOC5 | Tutorials, API reference and from-scratch corpus-to-model guide | CLI3 | Fresh-environment walkthrough succeeds | planned |

## Orchestrator rules

- Maintain one integration owner and a contract-change log.
- Create implementation tasks only when their declared dependencies are merged.
- Require each task to use subagents for at least independent testing/review when the
  work is non-trivial. Default to one independent reviewer for the bounded source
  and test change; add another only for a named unresolved risk.
- Prefer several bounded tasks over one cross-cutting task.
- Prefer completion-triggered handoffs and bounded waits over process polling.
  Do not repeatedly inspect an unchanged build or request duplicate status.
- Keep task handoffs to accepted contract, relevant files, exact evidence,
  unresolved findings, next gate, and commit/tree; link detailed logs.
- Review and integrate in dependency order; rerun affected downstream gates.
- Treat Eshkol-core defects as explicit upstream issues or isolated native-extension
  work, never as silent library fallbacks.
