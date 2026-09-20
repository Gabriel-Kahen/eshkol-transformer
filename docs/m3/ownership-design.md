# M3 ownership and contribution review

Historical contract-checkpoint review for issue #85. The consolidated proposal was
subsequently accepted at [issue #1 comment 5744129957](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5744129957).
The recommendations and preimplementation validation statements below record that
checkpoint; current implementation/evidence belongs to `docs/M3_MODEL.md`.

## Existing evidence

- A0 `model-output-logits` returns a newly owned contiguous tensor and retains the
  forward graph (`docs/PUBLIC_API_CONTRACT.md:392–401`). P1L state-backed borrowed
  handles are a different, explicitly accepted exception; they cannot silently
  replace this tensor contract.
- M3T admits one active frame per model, snapshots parameters/IDs/mode, and resets
  readiness when ending that frame. Its mutable logits copies carry no graph
  (`docs/M3T_TRANSPORT_PROPOSAL.md:247–293`). Keeping one workspace active per
  immutable output blocks later model calls; resetting it loses frame authority.
- `m3t-workspace-contributions-internal` already has arity 1 and returns 14 cached
  canonical-handle/carrier rows (`native/m3t_transport_extension.esk:575`). Native
  `et_m3t_private_workspace_gradient_v1(workspace,index)` requires reverse phase and
  readiness of every final gradient slot (`src/eshkol_transformer/m3t_transport.c:454`).
  Index 10 is GTIED, produced by head-then-token N3K sum; GHEAD and GTOKEN are not
  separate parameter destinations.
- Existing I2 native `et_f32_gradient_plan_prepare_v1`/5, `...commit_v1`/2 and
  `...release_v1`/2 already stage one whole unique-handle batch. Prepare pins
  destinations and sources; commit marks consumed before writes, then allocates
  nothing and has no recoverable branch. Release unpins and frees scratch and
  entries (`native/f32_tensor.c:1692,1942,1968`). Its retained plan shell is real:
  `retire_gradient_plan` clears authority but appends the shell to a retired list.
- The current public `diagnostic-workspace-check-primals` grants no future commit
  authority. The native check compares all 14 shapes/bytes plus mode and fixed
  constants; the Eshkol wrapper additionally authenticates profile/P1 mode.

## Recommended public lifetime

Use the proposed eight-name model surface: A0 `model-forward`/2+options,
`model-output-logits`/1, `model-output-loss`/1, `model-output-rng`/1; additive
`model-output-release!`/1, `diagnostic-output-vjp!`/4,
`diagnostic-model-logits-bits`/1 and `diagnostic-model-logits-release!`/1.

An immutable output owns one lease on a private immutable graph. The graph owns
14 detached saved parameter tensors (1,184 f32 values, 4,736 bytes), detached
I1 IDs (16 bytes), and detached logits (512 f32 values, 2,048 bytes): exactly
6,800 tensor payload bytes. Snapshot constants are epsilon bits `0x3727c5ac`,
positions `[0,1]` and four canonical true keep bytes: another 24 data bytes. Model,
profile/topology identity, captured train/eval mode and lifetime fields are control
metadata, counted separately. The graph does not own every forward activation;
Eshkol recomputes those from its saved primals when differentiation is requested.

`model-output-logits` allocates a distinct immutable owned f32[1,2,256] tensor
and acquires a graph lease. It never returns graph storage, a workspace alias, or
a state-backed borrow. Each returned tensor must be released explicitly. Its
detached 2,048-byte observational readout does not carry graph authority.

Release of the output drops its root lease exactly once. Retained owned logits
remain readable and differentiable, because VJP accepts exactly two authenticated
kinds: a live output or live graph-retaining model logits. Releasing the last lease
destroys graph tensor storage exactly once. The exact previously issued output or
logits shell releases idempotently; every other operation on that dead shell rejects
`invalid-state`. Forged/copied/unregistered/wrong-kind/cross-aggregate shells reject
`invalid-argument`; a live graph associated with the wrong model/workspace rejects
`invalid-state`. Graph-busy release rejects before changing any reference count.
No finalizer is assumed. Dead native and Eshkol identity records must clear all
payload/backreferences except inert authentication data needed for stale rejection.

Forward uses the model's private reusable workspace, captures the graph only after
all forward roles succeed, then ends the frame before exposing the output. Later
forwards therefore do not modify or monopolize earlier graph snapshots. VJP restores
the selected graph to the same idle workspace and runs both forward recomputation
and analytic reverse in Eshkol. Exact source bytes and owned logits stay unchanged
on success or rejection. Different seeds can differentiate the same live graph
repeatedly; contribution ordinals prevent replay from being confused with retry.

## Minimal candidate private seams

All names below are proposed private source seams, suffix `_v1`, initial contract
version 1.0; they are localized in the single M3 sibling owner and are not an
installed C ABI, provider-ID change, generic resolver or graph executor.

| Candidate name | Arity | Exact authority |
|---|---:|---|
| `et_m3_private_graph_capture_v1` | 1 | active completed forward workspace → newly owned immutable graph; allocate/copy fully before publication |
| `et_m3_private_graph_restore_v1` | 2 | idle workspace, graph → restore saved primals/IDs/constants and captured mode; authenticate same model, publish forward phase last |
| `et_m3_private_graph_check_primals_v1` | 2 | graph, current P1 mode → exact saved/live values, identities and constants check; observational only |
| `et_m3_private_graph_release_v1` | 1 | graph → drop the output root lease once; surviving logits leases preserve graph liveness |
| `et_m3_private_model_logits_create_v1` | 1 | graph → distinct immutable owned logits tensor plus graph lease, atomically published |
| `et_m3_private_model_logits_copy_to_v1` | 3 | owned model logits, exact bytevector header, byte count → synchronous detached 2,048-byte copy |
| `et_m3_private_model_logits_release_v1` | 1 | owned model logits → preflight, invalidate exact owner, destroy its tensor and drop graph lease |
| `et_m3_private_workspace_contribute_v1` | 5 | restored workspace, graph, current P1 mode, f32 normalization-weight bits, expected ordinal → validate then one complete I2 contribution and cleanup |
| `et_m3_private_workspace_reset_v1` | 1 | exact M3-managed workspace → allocation-free preflight and frame/graph-use reset; no authority over arbitrary diagnostic workspaces |
| `et_m3_private_i64_unborrowed_v1` | 2 | native-only I1 tensor, stack I1 error record → registry-first liveness and absent active-borrow check; read-only status return |

Root-lease release and graph liveness must be separate fields: a graph whose output
root lease is released remains usable through an authenticated logits lease. The
Eshkol record already knows its graph pointer; no graph-get or generic tensor
resolver is necessary. A restored frame records its exact graph identity, rejecting
a same-model but different-graph argument at contribution. Restore must not call
existing M3T begin, which would snapshot current live values instead of graph values.
No native seam schedules forward or reverse kernels.

The M3 source-composed sibling needs a private restored-graph association and
graph-use guard in workspace state. Its trusted reset clears both on success and
failure cleanup, while preserving existing M3T public semantics. Existing M3T
`active` and `workspace_unborrowed` reject `busy`; do not set that flag and then
call their public native entrypoints as if nested calls were admitted. Prove an
exact owned-call internal validation/cleanup path with no generic guard bypass.

Two additional cleanup seams are necessary: ordinary M3T `workspace_unborrowed`
allocates I1 borrow leases for both IDs and positions, and I1 borrow begin calls
`i64_calloc` (`native/i64_tensor.c:788`). Therefore that existing reset cannot be
the promised allocation-free post-commit or allocation-failure cleanup. The new
M3-only reset authenticates a private M3-managed-workspace marker and checks every
I2 owner with existing stack guards plus both I1 owners with the new leaf.

The leaf's exact proposed signature is `int32_t
et_m3_private_i64_unborrowed_v1(const et_i64_tensor *tensor,
et_i64_tensor_error *error)`. It belongs to an M3-only source-including I1
integration translation unit that replaces, never duplicates, the ordinary I1
object. First scan `live_tensors` using pointer equality; only after a match inspect
that registered record's magic and `active_borrow`. Do not substitute existing
`require_tensor`: it calls `valid_tensor`, which dereferences the candidate magic
without registry lookup (`native/i64_tensor.c:298`). Use existing I1 status/error
conventions with the caller's valid stack error record; create no lease, mutate no
tensor or registry, allocate nothing, and expose no Eshkol extern or installed
native symbol. Standalone I1 sources/public ABI/archive remain unchanged.

Contribution executes the same static complete reset-eligibility check before
I2 prepare. Its own plan pins arise only afterward; release removes them after
commit, then a fixed nonraising field-reset tail clears workspace/model/graph-use
state. It does not rerun the old allocation-capable reset. On pre-commit failure,
the new trusted reset can perform a complete allocation-free preflight. Tests must
disable I1 allocations entirely during this cleanup and separately hold an active
borrow on each I1 owner, proving rejection before any state changes and successful
cleanup after that exclusion ends. Test wrong/unregistered/freed pointers against
the leaf without dereference and non-M3 workspace identities against reset.

## Same-call atomicity and normalization

`diagnostic-output-vjp!(graph-bearing-value, seed, weight-bits, expected-ordinal)`
accepts the existing mutable diagnostic logits owner as seed and snapshots it before
computation. Seed is the already weighted logit cotangent. Numerator is exactly
`J(logits,parameters)^T seed`; weight changes metadata only. If the desired mean VJP
uses cotangent `u` and weight `w`, the caller must provide `seed=w*u` explicitly.
M3 adds no multiplication, mask reduction or loss kernel through this contract.
Weight bits must be a representation-real exact u32 encoding positive finite f32;
ordinal must be a representation-real exact nonnegative signed-i64 value.

The Eshkol operation authenticates the closed value kinds, same model/profile,
immutable topology and current P1 mode under a serialized nonreentrant admission.
It restores the graph and runs the complete Eshkol schedule. The final native
contribution call validates the exact restored graph, all 14 ready gradients and
canonical P1 bindings, repeats graph/live primal comparison, and invokes ordinary
I2 prepare/commit/release without returning control or invoking an application
callback between validation and commit. Native pointer possession alone never
authenticates a public P1 handle. No caller supplies contribution rows or destinations.

Every destination occurs once, including the tied sum. One successful call advances
all counts by one and adds the supplied normalization weight by I2 binary32 rules;
it never counts the 15 logical parameter paths or two token positions. Absent means
state absent/count 0/weight exact +0/backing exact +0. From absent, a finite all-zero
seed creates present all-zero numerators with count 1 and positive weight. When a
nonzero numerator is already present, zero seed preserves it while advancing count
and weight. Further contributions
require identical prior state/count/weight across all 14 slots and matching expected
ordinal; reject duplicate destinations, inconsistent metadata, nonfinite seed or
gradient, count/weight/arithmetic overflow, aliases, active borrows and unsupported
floating environment before changing any gradient byte or metadata field.

Prepare failure leaves the plan slot null. A successful prepare is released exactly
once on both pre-commit rejection and success. Error status is captured before
cleanup overwrites native status. Commit and its admitted release/reset tail must
contain no recoverable public-error path after the first gradient write. With the
same-call private stack slot, validated owner lifetimes and serialized execution,
I2 release's parameter checks are proved infallible, rather than ignored. Test that
proof, including release of source pins before workspace reset. If a true invariant
break makes post-commit cleanup fail, it is not a rollback-capable failure. Workspace
and graph busy flags clear on every rejected pre-commit path; the immutable graph
remains retryable. Existing explicit `module-zero-grad!` resets all 14 slots to
canonical absent; VJP itself does not reset or normalize accumulated numerators.

## Retained-memory boundary

Ordinary I2 plans are recommended initially; do not add a new scoped-plan API in
this milestone. On the inspected x86-64 C layouts, gradient plan preparation uses
48 control bytes + 14×24 entry bytes + 4,736 staged numerator bytes = 5,120 bytes
at peak, then release retains the 48-byte inert shell. An I2 reset plan retains
40 bytes after freeing its 14×8-byte pointer array. Local compile-only static-assert
probes including the actual I1/I2 sources confirmed these sizes and the 88-byte I2
and 72-byte I1 controls using GCC 16.1.1. These are local layout evidence, not a
supported-host memory trajectory; verify testing counters on the supported exact head.

Each fully released graph additionally leaves 15 ordinary I2 tensor shells. At
88 bytes each that is 1,320 retired bytes before M3 graph identity metadata. Each
released owned model-logits accessor clone adds another 88-byte I2 shell. I1 frees
its carrier/borrow controls. Successful internal clones later rolled back during
failed capture also contribute their ordinary I2 retired shells; count them even
when no public graph was published. Live graph tensor shape/stride storage is
432 bytes (22 parameter ranks, 3 logits ranks and 2 ID ranks, at 16 bytes/rank),
and live I1/I2 carrier controls are 1,392 bytes (15×88 + 72), beyond the 6,800 data
bytes and 24 constant bytes. Native graph/lease records and all Eshkol retained
registry/condition/vector allocations are additional measured terms.

The honest memory witness reports live data/control counts, retired control counts
and exact Eshkol arena retention at multiple iteration counts, for (a) reused frame,
(b) repeated VJP on one graph with explicit zero-grad, and (c) fresh output/accessor
creation and release. For case (b), successful prepare+reset alone adds at least
88 native retired bytes per cycle, plus any retained P1/Eshkol plan bookkeeping.
For case (c), graph/output creation adds cumulative identity and I2 shell overhead.
Live payload may return to baseline while total retained storage grows linearly.
No flat training-loop, indefinitely bounded process, or RSS-only claim follows.
If bounded total retention is operationally required before a separately reviewed
region-owned identity/plan design, use a bounded diagnostic worker lifetime and
report process teardown as its reclamation boundary.

## Required rejection and atomicity tests

1. Preserve logits bytes and graph VJPs across later forwards, mutated/released
   input owners and workspace resets. Release output before logits, logits before
   output, multiple logits clones, repeated releases, and final graph release.
2. Reject wrong-kind/forged/copied/stale/foreign/wrong-owner values, wrong graph for
   a restored same-model frame, active release/restore, nested VJP, reused seed
   mutation and incomplete reverse slots. Retained logits remain differentiable
   after their output root lease dies.
3. Mutate each of 14 live parameter tensors independently after forward, plus mode
   and test-only constants/topology. VJP must fail without contribution; exact
   restoration may pass. Check-now/mutate/commit must fail at the final same-call
   validation even when an earlier observational check passed.
4. Sweep every capture/clone/restore/contribution allocation failpoint, including a
   last-entry failure. Compare every destination value, gradient byte, state/count/
   weight, graph and seed byte before/after; live plans/borrows/pins return to baseline.
   Test failed prepare, prepared-but-uncommitted cleanup, success cleanup, duplicate
   commit rejection, and source release/reset blocked while a plan pins them.
5. Distinguish absent from present-zero. Test ordinal 0→1→2, replayed/stale ordinal,
   differing prior slot metadata, zero/negative/nonfinite weight, bad FP environment,
   weight/count/sum overflow, explicit reset and exact-positive-zero canonicalization.
   Weight-only changes must not scale numerator; tied head/token edges are summed
   once, and non-token/head parameters still each receive one metadata increment.
6. At 10/100/1000 or larger runs, record exact native live/retired bytes and Eshkol
   arena retention for success and failure separately. Verify expected slopes;
   explicit release is not evidence that every identity allocation was reclaimed.

Validation: `cc -std=c11 -Iinclude -Inative -fsyntax-only
/tmp/m3-ownership-i2-size.c` and the matching `/tmp/m3-ownership-i1-size.c`
both passed. The I2 probe statically asserts tensor/gradient-plan/reset-plan/entry
sizes 88/48/40/24; I1 asserts size 72. No runtime tests or performance measurements
were run. No runtime file, package manifest or ROADMAP status changed.

Independent disposition: the consolidated `docs/M3_COMPOSITION_PROPOSAL.md` eight
public names, ten fixed private seams, immutable snapshot/recompute design and
ordinary-plan accounting are suitable for the integration contract checkpoint.
Implementation review must prove, rather than assume, the admitted nonraising
post-commit cleanup tail and exact region-retention slopes described above.
