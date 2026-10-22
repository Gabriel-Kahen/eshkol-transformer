# Source-private CPU4 training lease

Status: implementation candidate of the independently reviewed CPU4 typed lease
revision2 (`f97e2177ae06bbe0e5de6031e1af3b0bf91b95c1281c6e3f1502be4cabdfe8ab`).
No compiled result, model-training or CPU-v1 admission is established here.

`ET_CPU4_TRAINING_PRIVATE` adds a registered control-only training lease to the
canonical C4 owner. It borrows the existing fourteen parameters/identities;
it creates no second parameter set or P1 model and never uses `owner->active`
as a training-pointer cast. Canonical C4 Scheme slot7 remains unchanged.

A whole training lease excludes every live native generation context,
including idle contexts. The common native publication tail counts one
context once; failure/repeated close cannot decrement it, and overflow rejects
before publication. Training admission requires sealed exact membership,
canonical shape/identity checks, idle storage and zero contexts. New raw,
seeded and RNG-based contexts reject until training releases the lease.

The nine private v1 methods in `g3c4_model_owner_internal.h` implement the
reviewed IDLE/READ/WRITE/RELEASE_PREPARED/DEAD lifecycle. Read-phase native child
hooks register exact child identity/kind/current ordinal, excluding duplicate,
stale and foreign resource registrations. All children must be released before
phase leave or lease release. Release preflights canonical value and gradient
storage, supports cancellation/retry, and commits by assignments without
destroying C4 parameters. Inert control/child records remain rooted to reject
address reuse; this bounded retention must be measured at integration.

The native transport is responsible for authenticating its own child registry
before these native-only hooks. This leaf does not yet implement CPU4
workspace/input/graph transport, P1 trainer enrollment, config/seed sideledger,
O2 updates, checkpoint restoration, evaluation or generation from trained
weights. Those callers must enforce the READ gradient-only/WRITE value-update
contract and release their real children/pins before leaving a phase.

Run `scripts/test-cpu4-training-lease.sh EVIDENCE_DIR` in the supported locked
environment. It checks feature-off object identity against the frozen source,
then normal/repeat and leak-enabled ASan/UBSan lifecycle witnesses. Native tests
cover idle/active generator exclusion, context/ordinal overflow, allocation
failures, foreign/mixed rosters, phase/child mismatches, borrowed value/gradient
release rejection with retry, cancellation, idempotent commit and exact
preservation of all fourteen parameter/gradient bytes and identities.
