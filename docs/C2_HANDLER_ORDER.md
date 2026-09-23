# C2 handler installation and release authority — issue #121

Status: **in progress; final transformer candidate frozen and independently
approved; pin adoption, full affected-gate union, and supported integration
evidence pending**. This records the current
source contract and focused final-runtime evidence for the issue #121 correction.
It is not completion evidence.
The previously accepted [C2 component evidence](C2_TRAINING_STATE.md) does not
prove the new handler-reservation or allocation-failure behavior.

## Scope and ownership rule

Installing an exception handler can allocate. Cleanup protection must therefore
exist before an operation claims, publishes, borrows, or consumes ownership, or
the operation must use a verified runtime reservation made before that change.
Allocating an ordinary cleanup vector after acquisition has the same ordering
problem. Error bookkeeping uses a separate failure flag because `#f` is a valid
raised value.

The correction covers C2 composition, borrow activation, SAVE, O2 reconstruction,
public LOAD staging, LOAD reconstruction/rollback, C2 owner release, and the P1
owned-state release call chain used by those operations. Public checkpoint and
trainer release arities and serialized checkpoint formats do not change. Private
source composition and the external runtime dependency do change.

## Audited ordering sites

| Site | Required authority and ordering |
|---|---|
| [C2 composition](../native/c2_training_state_extension.esk), `c2-training-state-compose-internal` | Allocate claim progress first; install rollback before claiming P1. Record completed P1 claim before attempting O2. Failed O2 claim rolls back exact P1/token authority; success keeps the assignment-only request publication tail. |
| C2 borrow activation, `c2-training-state-borrow-begin-internal` | Allocate borrow/progress records before entering scratch. Install the outer guard before publishing busy; keep it active while installing the inner activation guard. Read canonical identities back through the owner after publication. |
| C2 borrow rollback, `c2-training-state-borrow-activation-rollback!` | Re-read authority from the canonical owner graph. Deactivate P1 only when activation completed, clear the exact backlink, and restore live state. Repeated outer rollback after successful inner rollback is harmless. |
| [SAVE](../native/c2_checkpoint_save_extension.esk), `c2-save-with-owner-borrow` | Allocate `[borrow-or-#f, ended?, failed?, caught-value]` and install the guard before borrow-begin. Enroll the returned canonical borrow before preflight/build. End it before publication; writer failure must not end it again. |
| [O2 reconstruction](../native/c2_o2_reconstruct_extension.esk), `o2-state-reconstruct-internal` | Allocate the builder cell and install abort protection before native create. Enroll a nonnull builder immediately; consume the cell before abort and clear it after successful commit before publishing the O2 state. |
| [Public LOAD](../native/c2_public_extension.esk), `c2-public-checkpoint-load` | Allocate the stage cell and install its cleanup guard before stage construction/publication. Enroll the exact stage, release it on failure, and disarm after reconstruction proves staging dead. |
| [LOAD reconstruction](../native/c2_checkpoint_load_extension.esk), `c2-checkpoint-load-reconstruct-internal` | Preallocate owner cells, outcomes, and the C2 release record. Reserve handler capacity and install all five persistent cleanup guards plus the reconstruction guard before marking staging consuming or decoding P1. |
| LOAD component rollback | Use the already-installed guards to continue through ordinary P1/O2 cleanup, claimed C2 P1/O2 cleanup, and finalization. Move exact composed-owner authority into the preallocated release record; do not install a fresh whole-C2 release wrapper during rollback. |
| C2 owner release, `c2-training-state-release-internal!` | Allocate release/error records and reserve capacity before both child-cleanup guards. Only then transfer ownership, mark releasing, and invalidate dependents. Drain both children, finalize dead, record the first caught value for defect bookkeeping, and report the normalized internal defect without retrying consumed authority. |
| [Public P1 state release](../internal/p1/lib/transformer/module.esk) | Read preallocated lifecycle scratch and reserve five frames before native release admission, registry compaction, and the public cleanup guard. Exact-dead admission returns before dereferencing a dead raw-state marker. |
| Claimed P1 state release | Validate the exact claim and obtain lifecycle slot 8 before native admission or claim clearing. Depend on the enclosing C2/LOAD reservation; introduce no independent reserve or scratch allocation after C2 consumption. |
| P1 `state-dict-release-owned!`, `release-owned-list!`, and `tensor-release-owned!` | Use preallocated state scratch and the reserved capacity for state, per-item, and callback handlers. Keep ordered, exactly-once callbacks, carrier clearing, first-error propagation, and continuation through remaining items after a post-consumption provider defect. |

Setup failure before ownership changes must leave the original authority intact.
Once release begins, the existing provider contract distinguishes consumed
authority from a defect reported after consumption. A blind retry is not an
acceptable repair. LOAD remains one-shot after entering consuming; failure then
clears staging references and drains every enrolled component.

## Private P1 lifecycle contract

The P1 lifecycle vector has **exact length 10**. Slots 0–7 keep their existing
meaning. Every one of the three state constructors supplies these new private
slots when the state is created:

| Slot | Initial value | Use |
|---|---|---|
| 8 | `(vector 'empty #f #f #f)` | Owned-release scratch: phase, failed flag, caught value, exact lifecycle. |
| 9 | `(vector #f #f)` | Public release outcome: success flag and result/caught value. |

The internal function is now
`(state-dict-release-owned! state release-outcome)`. Its scratch argument must be
the canonical slot-8 vector, not a release-time allocation or copied authority.
Public release obtains slots 8 and 9 from the same canonical lifecycle before
native admission; claimed release obtains slot 8 before admission. The internal
callee retains its cleanup guard but constructs no scratch vector.

[The P1 template](../templates/p1/module_roots.esk.tmpl) and
[generated source](../internal/p1/lib/transformer/module.esk) must agree. This is
an internal layout change, not a new serialized P1/C1/C2 schema or public export.

## Runtime reserve ABI and budgets

The external runtime symbol is
`eshkol_runtime_reserve_exception_handlers_v1`. Eshkol declares one `i64`
argument and an `i64` result, using private aliases
`p1-runtime-reserve-exception-handlers` and
`c2-runtime-reserve-exception-handlers`. Call sites require zero on success and
reject a returned nonzero status. The allocation witness below demonstrates
runtime failure delivery and recovery; the structural checks alone do not.

| Caller | Literal reserve | Placement and budget |
|---|---:|---|
| Public trusted P1 `state-dict-release!` boundary | 5 | Before native release-begin and the public cleanup guard: public, state, item, provider-callback, and E1 shell-construction frames. |
| `c2-training-state-release-internal!` | 6 | Before the two C2 guards and authority transfer: two C2 frames plus P1 state/item/callback and E1 shell-construction frames. |
| `c2-checkpoint-load-reconstruct-internal` | 11 | Before six LOAD guards, reconstruction setup, consuming, and decode: the maximum is six LOAD frames plus C1 ownership, P1 adoption cleanup, item, provider-callback, and E1 shell-construction frames on the reachable adoption/cleanup dual-fault path. |

These are explicit bounded call-chain budgets, not a statement that every
provider can install arbitrary additional guards. Claimed P1 release does not
reserve independently after its parent has consumed authority. The runtime must
make the reserved frames available across sequential cleanup/rethrow paths;
source order alone cannot prove that implementation property.

The [toolchain lock](../toolchain/eshkol.lock) still names the existing pin
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`. The final reviewed runtime successor
used here is commit `81298b4a9608fb92eb6f351a2eabd8392da7d9ef`, tree
`7669312845a9d8d372006af52271045e69505813`, built as Release with Clang/LLVM
21.1.8 and promotion testing disabled. Pin adoption remains external to this
source correction. Silently satisfying the new symbol through another runtime
or library is not acceptable evidence.

## Current checks and pending evidence

[The structural checker](../tests/c2/test_handler_order.py) parses source forms,
distinguishes active guards from handlers/sibling/quoted/lambda decoys, verifies
canonical authority bookkeeping and template/generated parity, and checks the
reserve ABI, exact literal budgets, and ordering. Mutations cover removed,
incorrect, and late reserves; all three lifecycle constructors; release-time
scratch allocation; and incorrect or late canonical scratch reads.

The last recorded local structural invocation was:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.c2.test_handler_order
11 tests passed in 34.994s
```

This is source evidence for the working candidate, not final exact-tree
acceptance. Whitespace checks were clean.
[The owner gate](../scripts/test-c2-training-state-owner.sh) now invokes this
checker before its compiled checks.

[LOAD runtime tests](../tests/c2/c2_checkpoint_load_runtime.esk) add six cases
that report `#f` or a structured defect after real claimed-P1/O2 release, singly
and jointly. They assert P1 → O2 → finalization exactly once, exact owner/record
retirement, idempotence, and the existing native resource formula
`6 / 1 / 1 / 1 / 2 / 1`. These added cases have received source and parenthesis
review. The focused gate expects exactly 290 checks.

| Required evidence | Current record |
|---|---|
| Final transformer candidate commit/tree and review | `beb5821726d704604f8335dd85172b260d9a609f` / `24ca2dda199e5bd60efe8f54911de735cf44e61a`; independent `/root/c2_final_review` APPROVE with production aggregate `d37cc7df927ba8c5a8a7b3f731e035d2695148d70e64170257bf662496e135e2`, witness/test `c3879710159b0cd8bec404643d53d2613fecb09816202f454bf0f1a2b0f45f77`, and documentation `1a5410978cb61201ddd2fe96034843a3465641f86c05c2ea5d65e5a6d7e75096` |
| Final runtime source commit/tree | `81298b4a9608fb92eb6f351a2eabd8392da7d9ef` / `7669312845a9d8d372006af52271045e69505813`; independently approved runtime review and PR #714 |
| Runtime library/compiler provenance | `/tmp/eshkol-rethrow-final-81298b4a-20260923T200234Z`; runner `4a0e6303f7b85ed06fb753b52b62155235a3a77bca6c32aeb17241a28ed80be1`; archive `c32bb593ac1f365f3cbeaefd581704c4be029a4aa8877db29463d0e12356c168`; network-disabled supported image digest `f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6` |
| Allocation-failure/recovery witness | PASS twice at exact 5/6/11 plus controlled diagnostic-10 exit 97 on the eleventh LOAD push; `/tmp/c2-handler-dual-final-evidence.7xpz53`, manifest `e76ac1556dd418e36f47d61d9dfb9e56fd1c21a827c1e0ba2eedf0a467140bcd`, repeated stdout `351a1cf1c2625dc7edc275950a8c91542ccd7c5bbddd1e2313acda19db634784`, empty stderr |
| Final structural test rerun | PASS: 11 tests in 34.994s as part of the final-pin owner gate |
| `scripts/test-c2-training-state-owner.sh` | PASS on the final runtime: repeated AOT/runtime, ownership/failpoints/topology, deterministic fixtures, ASan/UBSan and source closure; log `8555fbf08120719e13211172c2f1bfe822ee473316d6b5b4a09667a4609de0c4` |
| `scripts/test-c2-checkpoint-load.sh --load-only` | PASS on the final runtime: strict Clang/GCC/C++, deterministic AOT/runtime, exact 290-check reconstruction/rollback, parser/reader failpoints, sanitizers and source/symbol closure; log `372f86a1f5666cb8958f02b9bd2f58c746f6a8219154e754366367e46848496d` |
| P1 release/idempotence and affected P1/C1 package gates | PENDING: exact commands/results/log hashes |
| SAVE/public/operational C2 gates | PENDING: exact commands/results, root retention, elapsed time, peak RSS, log hashes |
| Supported full CI and integration acceptance | PENDING: exact candidate/runtime provenance, run URL, reviewer and adoption decision |

## Explicit limitations and adoption boundary

- The runtime's active handler stack and current exception are process-global.
  The supported contract is one runtime execution thread; a thread-local free
  pool does not make exception execution thread-safe.
- The handler pool is TLS-backed and has no thread-exit destructor. Capacity is
  retained, and exiting transient threads can leak their pooled frames. There is
  no transient-thread reclamation or multithreaded exception-safety claim.
- Budgets 5/6/11 cover the identified fixed release call chains. Arbitrary
  provider-internal guard depth is outside the reservation guarantee.
- Allocation-failure recovery is proven for the fixed 5/6/11 paths exercised by
  the focused witness. It is not a claim for arbitrary provider-installed guard
  depth or every failure path.
- Pin adoption, bootstrap/provenance integration, and supported CI are external
  acceptance dependencies. No adoption or portability result is implied here.
- Larger lifecycle records and retained handler capacity require fresh retention
  and operational measurements. Historical C2 memory/timing results are not
  measurements of this candidate. No relaxed operational ceiling is authorized.

This correction does not expand C2 into live-trainer resume, generation
equivalence, arbitrary-provider recovery, or general concurrency support.
