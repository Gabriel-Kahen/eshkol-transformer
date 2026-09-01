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
- snapshot with any present/pending I2 gradient state is `invalid-state`;
- detached repeated snapshots and caller-mutation probes for every list/string/tensor;
- wrong format/version/features/algorithm/precision/device/provider, duplicate/
  missing/unexpected/alias paths, invalid/incomplete/duplicate state groups, receiver
  path/alias topology mismatch, invalid counter, malformed metadata, aliased moment
  snapshots, and nonfinite moments;
- failed load preserves all prior optimizer state and never changes parameter values
  or gradients;
- load after N steps into an independent optimizer, then identical next gradients:
  require bit-identical parameters, moments, effective learning rates, and counters;
- forged, stale, and cross-aggregate optimizer identities or untrusted copied state
  shells reject before mutation; detached state returned by `optimizer-state` and a
  future trusted C2 reconstruction remain valid load inputs.

## Packaging and isolation

- exact post-I2 aggregate defined/undefined symbol manifests and private localization;
- both import orders with one E1/P1 registry; reject already-localized and standalone
  registry-owning inputs;
- hostile `ESHKOL_PATH`, `ESHKOL_LIB_DIR`, `ESHKOL_JIT_CACHE_DIR`, `XDG_CACHE_HOME`,
  `PATH`, locale, working directory, and Python environment do not change compiled
  output;
- production depfiles, symbols, strings, dynamic dependencies, and source closure
  exclude Python, PyTorch, fixtures, oracle readers, test providers, and failpoints;
- ASan/UBSan cover every native malformed handle, alias, borrow, plan, and failpoint
  case admitted by I2.
