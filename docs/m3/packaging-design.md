# M3 packaging and A0 contract subreview

Nonbinding supporting subreview for [the consolidated proposal](../M3_COMPOSITION_PROPOSAL.md),
based on main `21b1df13940e94adce998ab88cb45c329c956b58` and merged M3T commit
`f50160968bd97677632f3539fd2e278a63624cf9`. No public API, manifest, runtime,
roadmap status, or accepted contract changes are made by this document.

## Recommended candidate

Implement the four existing A0 names with an explicitly bounded supported subset
and four additive names. Keep construction at the existing
`diagnostic-model-create/2`; A0 specifies no competing model constructor.

| Eshkol operation | Arity | Proposed result/contract |
|---|---:|---|
| `model-forward model input-ids . opts` | minimum 2 | New exact authenticated owned output and analytic graph for the diagnostic profile. Borrow a live same-aggregate M3T input owner containing dense CPU i64[1,2]. No mutation of input or model values. |
| `model-output-logits output` | 1 | Fresh owned read-only dense CPU f32[1,2,256] snapshot, with an independent lease on the same graph. This is a distinct kind from a mutable diagnostic logits receiver. |
| `model-output-loss output` | 1 | `#f` in the supported no-target subset. |
| `model-output-rng output` | 1 | `#f` in the supported no-RNG subset. |
| `model-output-release! output` | 1 | Release this exact output lease; idempotent for an exact dead output. Sibling logits copies remain usable. |
| `diagnostic-output-vjp! result seed normalization-weight-bits expected-ordinal` | 4 | Accept an exact live output or model-logits snapshot retaining its graph, and a live mutable diagnostic logits seed owner. Complete the Eshkol analytic reverse schedule and atomically contribute once to all 14 unique model parameters. |
| `diagnostic-model-logits-bits logits` | 1 | Fresh detached 2048-byte little-endian exact-bit copy of the read-only model-logits snapshot. |
| `diagnostic-model-logits-release! logits` | 1 | Independently release this exact owned logits copy and graph lease; idempotent for its exact dead identity. |

The two proposed tensor operations are deliberately fixed-profile inspection and
release, not generic tensor/provider access. The logits type's shape, dtype,
device, layout, graph identity, and mutation rules are specified above without
adding a generic tensor API. An ordinary bytevector is an explicit data export,
not the tensor returned by `model-output-logits`.

`model-forward` preserves A0's complete spelling and variadic option syntax.
First reject malformed/cyclic lists, unknown/duplicate/missing-value options,
unpaired `:targets` and `:loss-mask`, and a nonboolean `:deterministic?` as
`invalid-argument`. Then reject the presence of paired targets/mask or `:rng`
as `unsupported`, before dereferencing the unavailable carrier or allocating a
graph. This includes `:rng #f`; omit RNG for the supported subset. Feature
rejection does not authenticate or accept the supplied opaque value. M3T's
initializer is not an RNG substitute. `:deterministic?` true requests the reviewed
fixed deterministic operations, and false permits the same deterministic path.
No arbitrary shapes, alternate dtypes/devices, cache, or dropout are admitted.
Train and eval modes are both admitted and captured. This option precedence is
an explicit proposed subset decision for integration, not a claim that all A0
loss/RNG argument-validation paths have been implemented.

This is an A0-compatible bounded implementation, not complete first-release A0
support. In particular a foreign D2 tensor shell cannot cross this sibling's
registry boundary. The M3T input owner has genuine i64[1,2] native storage; an
ordinary integer list or T1 i64[2] result is not directly relabeled as that tensor.
Existing explicit input-copy operations remain its ingress routes.

If integration requires leaving all A0 runtime names unimplemented, the exact
alternative renames the first four names to `diagnostic-forward`,
`diagnostic-output-logits`, `diagnostic-output-loss`, and
`diagnostic-output-rng`; release then also becomes
`diagnostic-output-release!`. Counts and ownership work are unchanged. This
alternative must say that A0 remains declaration-only. It is not a reason to
substitute mutable workspace/receiver identities for owned outputs.

## Counts and package boundary

Current checked manifests contain 79 boxed package exports and 85 global defined
symbols/public-name strings: the difference is six E1B accessors. M3T adds 38
operations to the I2 base's 41 exports / 47 globals. The candidate adds exactly
eight wrappers, yielding **87 exports / 93 globals / 93 public-name strings**.
Counts assume every proposed operation gets one boxed exported wrapper and all
new native/Eshkol implementation symbols are localized.

Add only `transformer/model.esk` to the installed facade closure, for seven
facades total: config, diagnostic_transport, error_consumer, error_public,
model, module, tokenizer. Keep one completed registry-owning aggregate, one
archive member, and explicitly reviewed full source/native closures. A candidate
`m3_package_root.esk` may load the existing M3T root and the new Eshkol model
schedule, but it must compile them into one object; linking an M3 object that
owns another registry beside `libeshkol_transformer_m3t.a` is not composition.

The candidate preserves the accepted M3T package as a separate predecessor
artifact. Its tuple policy should have its own exact lexical root/bridge/
rename/export files and independently frozen manifest set. Do not weaken
`m3t-package-policy.sh` into caller-selected arbitrary object/source admission.
The three existing primitive objects remain N2, N3K, and A2; additional native
model code is ownership/transport/atomicity only, never a full-model schedule.

No D2, O2, K2 or C2 facade/import/registry is added. D2's compiled shell identity
is an aggregate-local capability, O2 does not bind this sibling's handles, K2's
current report verifies only its fixed I2 rank-zero/rank-one subset, and public
C2 rejects rank-two model tensors. A generic same-provider spelling or pointer
does not bridge these boundaries. Trainer/loss/optimizer integration,
capability-report widening, checkpoint support, and the full training aggregate
remain separate dependency decisions.

## Narrow private seams actually missing

1. Owned output/model-logits authentication, storage copying, graph lease, and
   release. M3T's reusable workspace and mutable logits receiver cannot themselves
   be public immutable model outputs. The selected design owns independent graph
   snapshots and resets the model's private reusable workspace before output
   publication. Surviving outputs do not reserve the active-frame slot. VJP
   restores captured primals/IDs/constants into the idle workspace, replays the
   21-role Eshkol forward schedule, then the 25-role Eshkol reverse schedule.
   Independent outputs coexist; shared public/private M3T active-frame exclusion
   still applies while a call is executing. A graph owns 14 saved f32 parameter
   tensors, one I1 input, and one f32 logits tensor: 6,800 tensor payload bytes,
   plus constants, identities, headers and shape storage accounted separately.
2. One fixed, private atomic contribution operation taking the authenticated
   workspace plus current mode, exact weight bits, and expected ordinal. It
   revalidates immutable model/profile binding, all current primals and mode in
   the same public VJP call, obtains the canonical 14 gradients, prepares one
   I2 plan, commits once, and unconditionally releases plan ownership. It accepts
   no user-provided arbitrary parameter list, provider, callback, or native plan.
3. Explicit accounting for ordinary native plan ownership. The selected design
   keeps `et_f32_gradient_plan_prepare/commit/release_v1` unchanged: release frees
   staged payloads but retires a control object. There is no new scoped/reusable
   plan seam and no flat repeated-output/VJP/training-memory claim. Measure the
   exact native live/retired and Eshkol retention trajectories. Graph and logits
   tensor release also retains I2 controls, independently of reclaimed payloads.

The selected ten new localized native seams are exactly:

| Private native suffix | Arguments | Arity |
|---|---|---:|
| `graph_capture` | workspace | 1 |
| `graph_restore` | workspace, graph | 2 |
| `graph_check_primals` | graph, current-mode | 2 |
| `graph_release` | graph's output lease | 1 |
| `model_logits_create` | graph | 1 |
| `model_logits_copy_to` | model-logits, byte span, exact count 2048 | 3 |
| `model_logits_release` | model-logits | 1 |
| `workspace_contribute` | workspace, graph, current-mode, weight-bits, ordinal | 5 |
| `workspace_reset` | exact M3-managed workspace | 1 |
| `i64_unborrowed` | `const et_i64_tensor *`, caller-stack `et_i64_tensor_error *` | 2 |

Prefix each suffix with `et_m3_private_` and append `_v1`. Capture and logits
create return pointers; the native-only I1 leaf returns int32 I1 status, and
other seams return i64 status. The leaf has no Eshkol extern/boxed wrapper.
Graph release tracks the
output lease independently so idempotent repeat release cannot decrement a
surviving logits lease. Existing M3T status accessors carry new seam errors.
Snapshot/restore transfers fixed data and ownership only; all numerical schedule
steps remain Eshkol-authored. Private native seam count does not change P1's
18-public/46-private/64-total Eshkol entries or 36 native identity symbols.

The tenth seam belongs to an M3-only integration translation unit that
source-includes `native/i64_tensor.c` exactly once, replacing ordinary I1 once
in the candidate's fixed native tuple. It authenticates the candidate pointer
by equality against the private live-tensor registry before dereferencing the
matched record or checking its active borrow. Existing I1 `require_tensor`
dereferences magic before registry authentication and is not sufficient for
this seam. The leaf returns existing I1 diagnostics through a caller-stack
error and performs no allocation or mutation. Standalone I1 and ordinary M3T
artifacts/semantics remain unchanged; source/native dependency and localization
manifests must prove the replacement has neither a duplicate I1 owner nor an
exported leaf. Test foreign/arbitrary/dead tensor addresses, live active borrow,
and successful unborrowed admission under allocation denial.

There is already a trusted Eshkol contribution selector:
`m3t-workspace-contributions-internal/1` creates no new public ABI and returns
the workspace's preallocated 14 rows of canonical handle/native-gradient pairs.
The underlying native selector requires all unique gradient roles ready, including
the summed tied tensor. The existing native I2 gradient-plan API already performs
multi-parameter staged prepare/commit/release. What is absent is their protected
Eshkol-callable whole-workspace connection, not the underlying numerical addition.

VJP weight bits encode a positive finite f32. The seed is the derivative of a
weighted objective numerator; the weight is metadata and is not multiplied into
the seed or divided out of the returned parameter derivatives. Expected ordinal
equals every current unique parameter contribution count before prepare.
One contribution increments each count by one and adds the weight once,
including the single head/token shared handle. A zero seed still produces
present-zero gradients with positive count and weight. Only explicit
`module-zero-grad!` clears to absent. Post-commit cleanup failure must not be
reported as a retryable unchanged-gradient failure; the final implementation
must prove its private exact cleanup tail cannot fail after validated commit or
explicitly document a distinct committed-error disposition.

No loss kernel/target-mask transport, generic rank adaptation, RNG ingress, or
cross-aggregate bridge is silently added by these seams.

## Evidence required for this candidate

- Public AOT callers import only the seven installed facades and link only the
  completed candidate archive plus the pinned normal runtime. They call the
  actual `model-forward` and VJP functions; test-local literal schedules do not
  prove those functions implement the model.
- Two fresh builds, a hostile ambient include build, exact all-symbol/export/
  public-string/source/native/object/archive/facade manifests, real compile-only
  depfiles, private binding/link negatives, every new fixed/minimum arity,
  reverse import order, duplicate/cross-aggregate rejection, and no production
  Python/PyTorch dependency. Preserve the predecessor's precise fixed-path
  reproducibility claim; relocation invariance needs separate evidence.
- Exact complete logits and all 14 unique VJPs against the frozen independent
  development reference, multiple seeds/inputs, numerical gradients, and tied
  omission/duplication mutations. Public native-readiness success is insufficient
  to observe a gradient value. Test-only private observation may inspect gradient
  bytes, with its linkage kept separate from canonical public AOT evidence.
- Output/logits independence; output release before logits release and reverse
  order; multiple simultaneously live graphs; reusable workspace reset before
  output publication; graph restoration and surviving graph VJP;
  stale/foreign/wrong-kind/wrong-
  owner/active identities; mode and parameter mutation with exact restoration;
  failure before and after plan staging; duplicate VJP and exact count/weight
  overflow. Every transient plan/native payload must return to its stated baseline.
- Increasing output-create/release and full forward/VJP/release trajectories,
  not just reusable M3T frames: native live and retired/control counts plus exact
  Eshkol arena retention, independently accounting output data, graph leases,
  workspace, plans, and identity shells. RSS is supplementary. No finalizer claim.
- Supported exact-head Ubuntu 22.04 x86-64 / LLVM-Clang 21.1.8 full CI through the
  accepted CI engine and independent M3-R approval, then integration merge and
  retest. M3T's supported predecessor run is not M3 acceptance or JIT evidence.

## Source evidence and checks in this subreview

`docs/PUBLIC_API_CONTRACT.md:392` defines A0 forward; lines 394–401 define owned
logits/loss/RNG and retained graphs. The declaration's exact variadic definition
is `tests/fixtures/a0/transformer/public.esk:139`.

`native/m3t_transport_extension.esk` defines the model/profile registry and
private contribution selector. `src/eshkol_transformer/m3t_transport.c:54`
shows the distinct native input/logits owners; lines 334–462 show workspace
storage, single-active ownership, snapshots, primal checks, and fully ready
canonical gradient selection. `native/f32_parameter_internal.h:90` is the
existing multi-contribution plan; `native/f32_tensor.c:1787` verifies ordinal,
1942–1966 commits all prepared rows, and 1968–2006 releases/retains control.

`tests/m3t/test_package_contract.py` binds the current exact 38/79/85 counts,
six-facade closure, one member, and predecessor pins. The native package bridge
includes the I2 bridge once. `scripts/test-m3t.sh` contains the existing fresh
AOT, dependency closure, localization, arity, duplicate-registry and exact arena
checks. `toolchain/eshkol.lock` pins compiler commit
`90cbd7130f47b8184bcc77b8d5c1b0026da980de` and supported host/toolchain.

This subreview ran `python3 -m unittest -q tests.m3t.test_package_contract`:
**9 tests passed**. It made no runtime edits and ran no compiler, sanitizer,
full-suite, supported CI, or model execution test. The candidate remains subject
to the orchestrator's concrete contract decision on issue #1.

The consolidated eight-public/ten-private candidate was rechecked against the
actual source contracts. Its count arithmetic, separate logits kind, snapshot/
restore model and explicit ordinary-plan growth are consistent. Specify the
variadic facade's fixed boxed transport as four pointers (model, input, options
list, output), calling a localized fixed-arity-three Eshkol implementation; the
public binding still has minimum arity two. Implementation
must verify finite seed admission before the first reverse numerical invocation:
existing M3T VJP-begin copies the seed but does not itself perform that validation;
the first N3K linear VJP validates finite input during dispatch. No additional
public seam or broad native schedule is justified by that check.

The selected additional cleanup seams address this source finding: ordinary M3T workspace
reset is fallible and allocating. Its `workspace_unborrowed` calls I1 borrow-begin
twice (`m3t_transport.c:433`), which allocates leases
(`native/i64_tensor.c:785`). It cannot serve as the proposed nonallocating
post-commit tail. `workspace_reset/1` authenticates the exact M3-managed workspace,
preflights all I2 storage through existing stack guards and both I1 tensors
through `i64_unborrowed/2`, then clears readiness/active/restored-graph/phase
fields without allocation. Eshkol exception guards invoke this private reset
before contribution is reached. `workspace_contribute/5` uses the same static
cleanup eligibility preflight before prepare/commit, releases plan pins, and
performs the nonraising field reset directly after commit. Eshkol phase/owner
metadata is cleared by nonallocating updates after native success. Allocation-
denied success and error cleanup tests must verify both paths. This concrete
ten-seam choice resolves the packaging proposal blocker while preserving the
public 93/87 counts; ordinary M3T reset is not substituted into either tail.
