# O2 adversarial implementation matrix

This matrix is design input for the post-I2 implementation. It is not runtime
evidence and must not be marked complete before I2 merges.

## Construction and parameter groups

- malformed, cyclic, improper, over-depth, over-count, wrong-version, nonempty-
  feature, and unknown algorithm/config values;
- zero groups, empty group, unsorted group paths, unsorted groups, alias path,
  duplicate path/handle, missing handle, unknown path, stale/foreign/cross-aggregate
  tree or handle;
- malformed/noncanonical binary32 strings, negative zero, NaN, infinity, invalid
  learning rate/beta/epsilon/decay/clip/minimum ratio, invalid schedule counters;
- wrong shape, dtype, device, layout, offset, provider, storage alias, and unsupported
  mixed-precision/accelerator/sparse options;
- all failures before retained optimizer identity, moment allocation publication, or
  parameter/gradient mutation.

## Gradient accumulation and clipping

- unequal mask weights proving global weighted-objective equivalence rather than a
  mean of microbatch means;
- missing first/middle/last gradient; present all-zero gradient; mixed missing/zero;
- gradient shape/dtype/device/layout mismatch, alias, stale slot, empty contribution
  metadata, zero/negative/nonfinite total weight, and counter overflow;
- global L2 zero norm, exact 3-4-5 boundary, immediately below/above boundary,
  multiple groups, tied aliases counted once, subnormal inputs, avoidable square-sum
  overflow, true norm overflow, NaN, and both signs of infinity;
- injected I2 preparation failures at first/middle/last handle and every scratch
  allocation. Parameters, moments, gradients, contribution metadata, and counters
  must remain bit-identical.

## AdamW, schedules, and counters

- steps 1, 2, and a later step for bias correction; beta zero and near-one values;
- positive/negative/zero parameters and gradients; decoupled decay with present zero
  gradient; epsilon placement; multiple groups with distinct hyperparameters;
- constant and linear schedules at first step, warmup endpoint, first decay step,
  final configured step, and post-end clamp; failed update does not advance;
- completed-update signed-i64 overflow, nonfinite prepared moment/parameter/update,
  bias-correction underflow/invalid result, and injected arithmetic failpoints;
- validation and all scratch allocation precede the first write; commit is one I2
  nonfailing transaction that preserves every parameter storage identity.

## Zeroing, state, and continuation

- zero-grad clears each tied unique slot once and is idempotent; it changes no
  parameter, moment, group, schedule, or completed-update value;
- snapshot, load, and release are each exercised while the optimizer mutates, while
  a moment/state borrow is live, and with first/middle/last canonical gradient or
  contribution metadata present. Each is `invalid-state`, performs zero
  cleanup/commit, preserves a live/resolvable state and all values/counts, then
  succeeds after the exact busy cause is cleared;
- detached repeated snapshots and caller-mutation probes for every list/string/tensor;
- wrong format/version/features/algorithm/precision/device/provider, duplicate/
  missing/unexpected/alias paths, invalid/incomplete/duplicate state groups, receiver
  path/alias topology mismatch, invalid counter, malformed metadata, aliased moment
  snapshots, and nonfinite moments;
- failed load preserves all prior optimizer state and never changes parameter values
  or gradients;
- each successful moment clone is entered into the sole-owner ledger before the next
  fallible action; every partial/rejected snapshot releases the exact ledger prefix
  once, and publication transfers the complete ledger once;
- incomplete, duplicate, wrong-owner, wrong-path, and wrong-provider ledger entries
  reject during release admission with the accepted category before cleanup starts;
  every carrier/borrow/plan/owner count and all values remain unchanged;
- release admission requires an idle update boundary, absent canonical gradient
  slots, and zero moment/state borrows; it atomically closes resolution with
  `live -> releasing`, executes the nonallocating/nonraising exact-once tail, then
  publishes `dead`;
- first/middle/last release-tail provider-invariant injections prove that resolution
  stays closed, cleanup continues without rollback, aliases do not multiply releases,
  exactly `2 * unique-parameter-count` clones are released once, `dead` publishes
  only after the complete tail, counts return to baseline, and exact-dead repeat
  invokes no callback;
- malformed/non-state/wrong-kind/forged/copied/unregistered/cross-aggregate release
  receivers are `invalid-argument`; recognized busy/reentrant/releasing/owner-conflict
  states are `invalid-state`; exact registered dead repeat succeeds; post-admission
  provider defect is `internal`;
- a dead state used by `optimizer-load-state!`, trusted inspection, C2 serialization,
  or any state-backed moment handle rejects before I2/K1 dereference with the exact
  integration-approved non-release category; only exact-token release succeeds, and
  values/counts remain unchanged;
- C2 failpoints at temporary-state construction, borrow begin, every handle resolve,
  every I2/K1 borrow, validation, and encode end every acquired borrow, release every
  temporary owner, leave caller-owned state live/unconsumed, and return counts to
  baseline. Serialized output is scanned to exclude owner tokens, callback identity,
  provider authority (not the required inert provider identity), and capability
  evidence;
- live carrier/borrow/plan/owner counts return to baseline after repeated release,
  every clone failpoint, rejected/successful load, exact-dead repeat, busy release,
  use-after-release, and C2 borrow loops;
- exact deltas require one owner and two detached carriers per canonical unique
  parameter after snapshot, no retained input-snapshot carrier/borrow/owner after
  load (optimizer-owned replacement moments remain valid), zero delta for rejected
  busy/reentrant/forged/dead-use attempts, and the pre-snapshot baseline after
  successful release;
- the same snapshot loads repeatedly into two compatible optimizers and remains
  inspectable/serializable until release. Releasing it bit-preserves both optimizers'
  moments/parameters/gradients/counters/config, every P1 state, and a sibling
  snapshot;
- load after N steps into an independent optimizer, then identical next gradients:
  require bit-identical parameters, moments, effective learning rates, and counters;
  release the borrowed input snapshot and require the restored optimizer's next
  update to remain bit-identical;
- forged, stale, and cross-aggregate optimizer identities or untrusted copied state
  shells reject before mutation; detached state returned by `optimizer-state` and a
  future trusted C2 reconstruction remain valid load inputs.

## Packaging and isolation

- exact post-I2 aggregate defined/undefined symbol manifests and private localization;
- fixed O2-specific release wrapper accepts only the exact accepted I2 provider and
  exact O2 ledger entry; guessed callback tokens, caller-selected providers, P1
  state, live optimizer moments, parameters, and gradients are authority negatives;
- both import orders with one E1/P1 registry; reject already-localized and standalone
  registry-owning inputs;
- fresh-cache strict-source, object, and AOT authority negatives reject guessed O2
  release names, copied tokens, wrong aggregates/providers/ledgers, and private
  wrapper access without producing residual artifacts;
- hostile `ESHKOL_PATH`, `ESHKOL_LIB_DIR`, `ESHKOL_JIT_CACHE_DIR`, `XDG_CACHE_HOME`,
  `PATH`, locale, working directory, and Python environment do not change compiled
  output;
- production depfiles, symbols, strings, dynamic dependencies, and source closure
  exclude Python, PyTorch, fixtures, oracle readers, test providers, and failpoints;
- ASan/UBSan and LSan where supported cover every native malformed handle, alias,
  borrow, plan, ownership, release, and failpoint case admitted by I2/P1L.
