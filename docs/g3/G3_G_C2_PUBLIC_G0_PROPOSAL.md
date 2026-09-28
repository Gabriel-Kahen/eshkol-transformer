# G3-G public diagnostic-C2 zero-budget extension proposal

**Independently accepted bounded contract; isolated implementation candidate locally sealed, pending independent source review and hosted integration.** This is a successor
to the reviewed [public P1/G1 package](G3_G_C2_PUBLIC_LEAF.md). It exposes the
already accepted source-private [P1/G0](G3_T_ZERO_BUDGET_LEAF.md) and
[P2/G0](G3_T_P2_ZERO_BUDGET_LEAF.md) paths alongside P1/G1. It does not expose
manual prefill/decode, P2/G1, a second sample, cache replacement, N>1,
persistence or CLI. The diagnostic-C2 model, tokenizer, sampler and
two-position A2 limit do not change.
Implementation depends on acceptance and integration of that P1/G1 package
and its private seeded route; this proposal changes neither branch.

## Public names and fixed package revision

Keep the thirteen `transformer/generation.esk` names and arities from the
[P1/G1 contract](G3_G_C2_PUBLIC_CONTRACT_PROPOSAL.md); add no public
`generation-pair-input-create` name in this revision. `generator-create`
widens its configured `:max-new-tokens` to exactly `0` or `1` (greater values
remain `invalid-argument`/source code 3 before publication).
`generation-input-create` widens its authentic same-aggregate sealed T1 input
from length one to length **one or two**. It copies into a distinct G3-owned
CPU i64 `[1,P]`, where `P∈{1,2}`; the existing scalar
`generation-token-input-create` remains exact byte `0..255`, `P=1`.
Length-zero, length-above-two, non-byte, copied/foreign/wrong-kind and dead
T1 inputs remain rejected through the accepted T1/G3-T categories. P2 is
available through the T1 constructor only; the existing private
`g3t-generation-pair-input-create` is not installed. For budget one, only P1
remains eligible, so P2/G1 rejects before native pins.

This is **G3-G C2 package revision 2**, a new fixed E1B tuple rather than a
silent mutation of the reviewed `g3g_package_*` tuple. Fixed repository
stem: `g3g_g0_package_*`; fixed single-owner archive/member:
`libeshkol_transformer_g3g_g0.a` / `g3g_g0_package.o`, built by
`scripts/build-g3g-g0.sh`. Its default artifact directory is
`$(project_build_dir)/g3g-g0`, distinct from v1's
`$(project_build_dir)/g3g`; a sequential default build must leave the v1
archive, object and installed facades byte-identical. It installs the same
eight facades, with the
generation facade copied for this artifact. The public source names,
arities and thirteen boxed C signatures remain unchanged, so the target is
still 100 boxed exports and 106 globals/public-name strings; exact manifests
must be measured rather than inferred. The archive source-composes the
reviewed G3-G P1/G1 closure plus
`native/g3t_zero_budget_extension.esk` and
`native/g3t_p2_zero_budget_extension.esk`, and compiles the existing G3-T
transport with `ET_G3T_ZERO_BUDGET_PRIVATE` and
`ET_G3T_P2_ZERO_BUDGET_PRIVATE` in addition to its reviewed P1/G1 flags.
It retains one G3-T/M3 registry owner and ordinary N2/N3K/A2/G3N/G3S
providers. The two G3-G archives must never be linked together: their boxed
symbols and owner registries are separate package choices. A shell made by
one aggregate is foreign to the other. The original P1/G1 package and its
fixed manifests stay byte-identical and remain independently testable.

## Admission and exact source-private schedule

The current G3-T input owner exposes no production length query: only the
`ET_G3T_TESTING` observer reports `input->length`. The new facade therefore
needs a source-private **prompt-provenance record**, keyed by the exact
G3-owned input shell. `generation-input-create` records the authentic T1
length read before its successful native copy; the scalar constructor records
one. Preallocate the record and registry cons before invoking the existing
private owned-input factory, and enroll it only after that factory succeeds.
The record is an internal dispatch index, not an input owner or a substitute
for native validation. It cannot be supplied or edited by the caller, and
typed input release marks it dead only after successful native release;
blocked release leaves it live. Retain a tombstone for exact identity. The
facade first authenticates the G3-T generator and input shells through their
existing registries; a missing/forged/dead provenance record cannot route.
It must not choose P from an Eshkol tag, caller T1 value, failed P2 probe or
test-only length observer. No new native query or public ABI is assumed.

For `generator-generate!`, preserve the existing P1/G1 delegation to
`g3m-generate-p1-g1!` for budget one. For budget zero, read the authenticated
recorded P, run the existing read-only
`et_g3t_private_full_request_preflight_v1(generator,input)` through its
already declared source-private `g3m-native-full-request-preflight` adapter
before the private call, then delegate **once**: P1 to `g3t-zero-generate!`, P2 to
`g3t-p2-zero-generate!`. The latter retains its own P2-only
`et_g3t_private_prompt_preflight_v1` defense. The full-request preflight is
the authority for native owner identity, live/busy/model state, stored byte
IDs and `P+G<=2`; prompt provenance only selects the already accepted route.
On nonzero preflight status, snapshot the G3-T native domain/category/code
through `g3t-native-fail-raw` before any cleanup or second native call; the
public E1 boundary substitutes the invoked `generator-generate!` operation.
The P1 path reserves an empty ID/staging/raw output before its 21-role T1
frame. The P2 path reserves the corresponding empty output, runs one 21-role
T2 frame through N3K/N2/A2 with causal mask `{1,0;1,1}`, and publishes the
true second-row logits privately. Both use
`g3t-output-prepare`, `g3t-t1-decode-output!` and
`g3t-final-commit!` for adjacent no-failure publication. Neither calls
`g3t-sample`, begins a generated frame, consumes RNG, or exposes logits.
An exhausted categorical counter is eligible for G0; it is not a draw.

The allowed pairs are exactly `(P,G)=(1,0),(2,0),(1,1)`; P2/G1 and any
`P+G>2` fail `shape-mismatch` before pins via the native full-request
preflight. A committed prefix, busy owner/model, stale binding or an old-cache/
output borrow that blocks the accepted preflight or commit follows the native
`invalid-state` path. An underlying read-only I1 borrow may coexist with
generation where the private route accepts it; typed input release remains
blocked until that borrow ends. Budget above one
fails at constructor admission; wrong-kind/forged shells are
`invalid-argument`, and exact dead-shell use is `invalid-state`. Every public
failure reports the invoked facade operation with bounded data-only
source domain/category/code and `cause #f`; native error details remain
authoritative. An allocation, role, A2, binding, output or text failure
before final commit aborts the pending call/output and preserves the entry
cache, binding, RNG and older outputs. No G0 prompt is published on failure.
Successful final commit publishes cache length P and one live empty output
atomically; an impossible postcommit failure is fail-stop. A subsequent
`generator-generate!` on that generator rejects the committed prefix: this
revision adds no continuation or reset.

## Output ownership and acceptance gates

For both G0 paths, `generation-output-ids` returns a fresh one-element list
with an owned empty CPU i64 `[0]` clone;
`generation-output-lengths` returns owned CPU i64 `[1]` value `0`;
`generation-output-text` returns a fresh one-element list of a detached empty
bytevector; `generation-output-cache-lengths` returns owned CPU i64 `[1]`
value P; and `generation-output-rng` returns an independent snapshot of the
unchanged RNG words. The output and every accessor clone survive generator
close and release independently through the existing typed operations.
Wrong-kind, foreign, dead and borrowed-owner behavior, idempotent release,
preallocation and no-orphan clone rules remain those of the reviewed P1/G1
package and accepted private owner wrappers. EOS is irrelevant at G0: no
token is sampled or appended.

The implementation gate must prove the fixed revision-2 tuple, one owning
archive member, eight installed facades, exact source/native/object/export/
undefined/string manifests and no reachable private or test symbol from a
fresh-cache AOT public caller. Test genuine T1-backed P1 and P2 plus scalar
P1, budget 0/1, all three admitted pairs, rejected P2/G1 and overlength
before pins, malformed/foreign/dead/borrowed inputs, empty ID/text and 0/P
length clones, unchanged greedy/categorical RNG including exhaustion, old
output lifetime, EOS no-op, failure rollback and same-generator retry.
Source-private normal/repeat/ASan+UBSan+LSan witnesses must retain bitwise
P1/P2 last-logit and K/V parity with independent M3T, the T2 provider route,
allocation/A2/binding cuts, and all reviewed P1/G1 regression checks.
Exercise wrapper-level E1 operation/details and native clone cuts, Q0,
feature-off inventory, mixed-tuple rejection and v1 package isolation.
This proposal adds no generated-token loop or public manual operation.

## Isolated candidate evidence

The revision-2 implementation is at `1086582` / tree `5ccfa9f` before this
documentation seal. `native/g3g_g0_public_extension.esk` owns prompt provenance;
`native/g3g_g0_package_root.esk` source-composes only the accepted G0/G1
routes. `scripts/build-g3g-g0.sh` and its exact tuple/manifests keep a
separate one-owner artifact. The pinned network-none package build passed
exact source/native/object/export/undefined/string manifests and installed
one archive member and eight facades. The fresh-cache public AOT caller
passed normal/repeat with identical output, E1 operation/details, the three
admitted `(P,G)` pairs and P2/G1 pre-pin rejection. The v2 testing closure
passed 20 identical normal/repeat/ASan+UBSan+LSan checks, 24 source contracts,
Q0 4/4 and production test-symbol exclusion, including failed-release
provenance, A2 rollback/retry, exact G0 clone values and seeded/exhausted RNG
witnesses. The unchanged inherited private suite passed 47,342 identical
checks in those three modes with 23 source contracts and Q0 4/4, retaining
bitwise M3T logit/K/V parity. A sequential default v1→v2 build kept the
reviewed v1 archive, object and facades byte-identical; the v1 fresh-cache
public P1/G1 caller also passed normal/repeat. Local evidence is sealed under
`/home/gabe/.codex/evidence/eshkol-transformer/g3g-public-g0-candidate-20260928/`.
No hosted CI, public manual route, continuation or N>1 claim follows.
