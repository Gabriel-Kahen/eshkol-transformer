# First-release public API contract

Status: **A0 reviewed draft; declaration harness passing on the F0-pinned canonical
`tsotchke/eshkol` compiler at commit
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`**. This document specifies the target
first-release contract. It does not
claim that Eshkol core implements any tensor, autodiff, device, compiler, or
persistence capability. R0 must verify each runtime capability, and downstream
workstreams must fail explicitly when a required capability is absent.

The declaration fixtures exercise only syntax, imports, public names, and selected
arities. The local A0 harness compiles them twice and verifies the expected negative
cases only after F0 verifies the exact source checkout, compiler identity, build
provenance, and compiler binary hash; that is declaration evidence, not
runtime-capability evidence or supported-host CI evidence.

## 1. Stability and naming

The public package is split into these conceptual modules:

- `transformer.capabilities`
- `transformer.config`
- `transformer.tokenizer`
- `transformer.data`
- `transformer.module`
- `transformer.model`
- `transformer.optim`
- `transformer.trainer`
- `transformer.generation`
- `transformer.persistence`

The current compiler accepts `provide`/`require` modules, while A0 intentionally
implements no public subsystem. It therefore records names in
`transformer.api_contract` and uses declaration-only stubs under `tests/`; downstream
workstreams own the real modules and must keep these names and contracts unless a
contract change is accepted through issue #1.

Names ending in `!` mutate documented in-memory arguments or external state. Other
public operations have no mutation or external side effect. Opaque values must be
inspected only through public accessors. Every accessor raises `invalid-argument`
when given the wrong opaque value type unless a narrower error is stated below.
Flat option lists use quoted colon symbols, for example `':device 'cpu`; unknown,
duplicate, missing-value, or ill-typed options are `invalid-argument` errors.

Every operation below is part of the library SemVer surface. Changing a name, arity,
option, result, error, shape, dtype/device, layout, gradient, ownership, alias, or
mutation rule is an API compatibility decision. An operation that reads or writes an
artifact also affects that artifact's independent version domain.

## 2. Shape notation

Shapes are row-major dimension lists. Symbols mean:

| Symbol | Meaning | Constraint |
|---|---|---|
| `N` | batch size | `N >= 1` |
| `T` | current token sequence length | `T >= 1` |
| `U` | unbatched encoded/decoded token length | `U >= 0` |
| `V` | tokenizer/model vocabulary size | `V >= 1` and IDs are in `[0,V)` |
| `D` | model hidden width | `D >= 1` |
| `L` | decoder layer count | `L >= 1` |
| `Hq` | query-head count | `Hq >= 1` |
| `Hkv` | key/value-head count | `Hkv >= 1`, `Hq mod Hkv = 0` |
| `Dh` | per-head width | `Dh >= 1`, normally `D = Hq * Dh` |
| `C` | cache capacity | `C >= T`, bounded by configured context length |
| `P` | number of unique parameter tensors | `P >= 0` |

Canonical tensors are:

- token IDs and targets: `i64[N,T]`;
- loss mask: `bool[N,T]` or `f32[N,T]`;
- hidden states: floating `[N,T,D]`;
- logits: floating `[N,T,V]`;
- per-token loss, where exposed internally: `f32[N,T]`;
- KV cache storage: two distinct floating tensors, `keys` and `values`, each with
  shape `[L,N,Hkv,C,Dh]`, plus one `i64[N]` logical-length vector shared by both.
  A cache owns all three objects; callers never alias key storage as value storage.
  The cache layout is an API shape contract, not a file-format or native ABI layout
  commitment.

No public operation broadcasts tensor arguments. Shapes shown as equal must match
exactly. Scalars in configuration are ordinary control-plane values, not broadcast
tensors. A public tensor boundary requires dense row-major contiguous storage with
zero offset; no implicit materialization or copy is allowed to disguise a strided,
overlapping, or unsupported layout. An implementation may offer explicit conversion
outside this API.

## 3. Dtypes and devices

The first-release target requires `i64` token/index tensors, `bool` and `f32` masks,
and `f32` parameters/activations/gradients on `cpu`. `f16`, `bf16`, other index
dtypes, and accelerator devices are optional capabilities and must not be selected
unless discovery reports them as `verified` for the requested operation and shape.
Automatic dtype narrowing, host fallback, device fallback, and scalar fallback are
forbidden.

A device is the symbol `cpu` or an opaque backend device descriptor. Every tensor in
one model operation, including parameters, buffers, cache, IDs, targets, and masks,
must be on the declared operation device unless the operation explicitly belongs to
the CPU control plane. Tokenization, configuration parsing, metadata inspection, and
filesystem I/O are CPU control-plane operations. Transfers must be explicit in the
tensor runtime; this package does not define an implicit transfer API.

Floating model output has the model compute dtype and device. Reductions used for
reported losses and metrics accumulate and return `f32` in the first release.

## 4. Ownership, lifetime, mutation, and aliases

- Configuration values, tokenizer values, capability reports, cursors, error values,
  and persistence metadata are immutable snapshots owned by the caller.
- A dataset, module/model, optimizer, trainer, generator, and cache is an opaque,
  exclusively mutable object. It is not safe for concurrent mutation.
- Ordinary tensor inputs are borrowed for the duration of a call. Returned tensors
  and snapshots are newly owned and do not alias inputs or mutable internals.
- The one exception is a parameter tree: `module-parameters` returns opaque live
  parameter handles, not tensor-value copies. Handles have the module's lifetime and
  authorize controlled optimizer/gradient mutation. `optimizer-create` retains those
  handles and never updates copies.
- `module-state-dict` and all `*-state` operations deep-copy tensor values and include
  an alias graph for tied parameters. `module-load-state-dict!` copies values into
  existing storage, preserves declared ties, and rejects missing, unexpected,
  duplicate, or conflicting aliases.
- `module-parameters` represents each unique trainable tensor with one live handle
  while retaining every logical path and tie relationship. `module-buffers` snapshots
  non-trainable tensors in the same path order using the state-dict accessors below. Optimizer
  parameter groups refer to stable parameter paths, not raw addresses.
- Dataset and generator calls invalidate only their receiver's previous transient
  iteration result; returned batches/tensors remain owned by the caller.
- `token-dataset-close!` is idempotent. Other dataset operations after close raise
  `invalid-state`. Checkpoint writes retain no caller-owned state after return.
- `trainer-create` takes an exclusive lease on its dataset, model, and optimizer until
  the trainer is unreachable; callers must not mutate them concurrently.
  `generator-create` retains read-only model/tokenizer references plus its own
  cache/RNG; the model must not be mutated concurrently. Retained objects outlive the
  retaining object.

In this contract, an immutable snapshot is immutable as observable model or runtime
state: it never aliases mutable source state and cannot be used to mutate that source
or a later snapshot. The pinned Eshkol runtime does not physically freeze standard
string and list carriers. When a snapshot accessor returns those carriers, it returns
a fresh deep-owned copy; caller mutation may change only that detached copy. Opaque
values and immutable scalar values remain physically immutable where their operation
contract says so.

## 5. Error model

Public operations raise an opaque structured error inspected by
`transformer-error?`, `transformer-error-category`,
`transformer-error-operation`, `transformer-error-message`,
`transformer-error-details`, and `transformer-error-cause`. Accessors return new
detached CPU snapshots, do not mutate error state, and have no gradient. Categories:

| Category | Use |
|---|---|
| `invalid-argument` | malformed option, range, identifier, or configuration |
| `shape-mismatch` | wrong rank, extent, vocabulary range, cache length, or tree shape |
| `dtype-mismatch` | dtype is well formed but not accepted by the operation |
| `device-mismatch` | tensors or object state are on different devices |
| `noncontiguous` | input violates the dense row-major boundary |
| `unsupported` | unavailable verified capability, dtype, device, mode, or operation |
| `invalid-state` | closed object, wrong train/eval phase, absent gradients, or stale cursor/cache |
| `io` | open, read, write, fsync, rename, or permission failure |
| `corrupt-data` | checksum, bounds, truncation, duplicate entry, or malformed payload failure |
| `version-mismatch` | unsupported API, ABI, schema, or format major version |
| `determinism-unavailable` | deterministic execution was required but cannot be proved/provided |
| `internal` | invariant violation; never used to conceal a category above |

Errors are never converted to empty tensors, partial state, silent skips, fallback
execution, or approximate gradients. Validation precedes mutation. Operation-specific
commit points below define failure atomicity.

## 6. Capability discovery

```scheme
(capability-discover)
(capability-request operation dtype device shape deterministic?)
(capability-verified? report capability request)
(capability-require report capability request)
```

`capability-discover` returns an immutable report. Each entry has a stable capability
symbol, status (`verified`, `unsupported`, or `unverified`), implementation/version,
supported dtype/device/shape constraints, determinism properties, and evidence ID.
`capability-verified?` is true only for `verified` and a matching request; absence
and `unverified` are false. `capability-require` returns the matching verified entry
or raises `unsupported`.
Results may be cached only for one process and must not be serialized as proof for a
different host.

Reports and entries are inspected with `capability-report-entries`,
`capability-entry-name`, `capability-entry-status`,
`capability-entry-implementation`, `capability-entry-version`,
`capability-entry-constraints`, `capability-entry-deterministic?`, and
`capability-entry-evidence`. `capability-request` validates and returns an immutable
CPU request with exactly `operation`, `dtype`, `device`, `shape` (a concrete
nonnegative integer dimension list), and `deterministic?`; it raises
`invalid-argument`, does not mutate, and has no gradient. Matching requires
operation/dtype/device membership, exact rank,
every extent within inclusive per-dimension bounds, and a deterministic entry when
requested. Missing/unknown keys are `invalid-argument`. Accessors return new immutable
CPU values, do not mutate, and have no gradient.

`capability-report-entries` is sorted by capability-symbol spelling. A constraint is
an immutable map with exactly `operations` (sorted symbols), `dtypes` (sorted
symbols), `devices` (stable canonical device descriptors), and `shape-ranges`.
`shape-ranges` is a list of alternatives; each alternative is a dimension-ordered
list of `(minimum maximum)` integer pairs, where `#f` maximum means unbounded. A
request matches when one alternative has its exact rank and all extents are in range.
Implementation and status are symbols; version and evidence ID are strings. Lists
and maps returned by accessors are newly owned snapshots.

Required target capability symbols are `tensor.i64`, `tensor.bool`, `tensor.f32`,
`tensor.contiguous`, `autodiff.reverse`, `kernel.matmul`, `kernel.embedding-backward`,
`kernel.norm`, `kernel.activation`, `kernel.causal-attention`, and
`kernel.indexed-cross-entropy`. Optional symbols include `dtype.f16`, `dtype.bf16`,
`device.accelerator`, fused kernels, and deterministic accelerator variants.

At A0 all runtime capability statuses are **unverified**. Source inspection is not
runtime evidence.

## 7. Configuration

| Operation | Contract | Ownership/errors/gradient |
|---|---|---|
| `config-parse text` | Parse UTF-8 text in X1's selected data syntax into an immutable unresolved configuration. File I/O belongs to the caller. No includes, environment reads, code evaluation, or implicit defaults. | New CPU value; `invalid-argument`; no gradient. |
| `config-resolve config overrides` | Apply schema-known overrides, defaults, and derived values in a specified precedence order; reject unknown/duplicate keys. | New CPU value; `invalid-argument`; no gradient. |
| `config-validate resolved` | Validate all cross-field invariants including `V`, `D`, `Hq/Hkv/Dh`, context, dtype, device, and reproducibility policy. Returns `#t` on success. | No mutation; contract errors above; no gradient. |
| `config-canonical resolved` | Return canonical UTF-8 text with stable key order and number encoding, excluding secrets and runtime evidence. | New string; `invalid-argument`; no gradient. |
| `config-fingerprint resolved` | Return lowercase algorithm-qualified digest of `config-canonical`. | New string; `unsupported` if digest unavailable; no gradient. |
| `config-ref resolved key` | Return a new immutable value for a schema-known key. | CPU; `invalid-argument` for unknown keys; no gradient. |

The resolved configuration schema is independently versioned (`config-schema`), and
its canonicalization version is part of the fingerprint input. X1 owns the concrete
schema and syntax.

## 8. Tokenizer

| Operation | Shapes and semantics | Dtype/device/ownership/errors/gradient |
|---|---|---|
| `tokenizer-byte config` | Construct the deterministic byte tokenizer described by validated config. | New immutable CPU value; `invalid-argument`, `unsupported`; no gradient. |
| `tokenizer-load path policy` | Load and fully validate a data-only tokenizer artifact under the persistence policy. | New immutable CPU value; `io`, `corrupt-data`, `version-mismatch`, `unsupported`; no gradient. |
| `tokenizer-save! tokenizer path policy` | Atomically save a data-only tokenizer artifact after canonical validation and enforce the persistence policy's size limits. | External filesystem mutation only; tokenizer unchanged; `io`, `invalid-state`, `unsupported`; no gradient. |
| `tokenizer-encode tokenizer text` | UTF-8 string/byte string -> newly owned `i64[U]`, `U >= 0`; no implicit BOS/EOS unless declared by tokenizer config. | CPU contiguous; `invalid-argument`, `unsupported`; no gradient. |
| `tokenizer-decode tokenizer ids` | Contiguous `i64[U]` -> newly owned bytes/string under the tokenizer's declared UTF-8 error policy. | CPU; IDs must be in `[0,V)`; shape/dtype/range errors; no gradient. |
| `tokenizer-vocab-size tokenizer` | Return `V`. | Immutable integer; `invalid-state`; no gradient. |
| `tokenizer-fingerprint tokenizer` | Digest covers format/schema version, vocabulary/merges, special-token table, normalization, and byte/UTF-8 policy. | New string; `invalid-state`, `unsupported`; no gradient. |
| `tokenizer-special-token-id tokenizer name` | Return the configured `i64` ID; never invent a default. | `invalid-argument` if absent; no gradient. |

T1/T2 own concrete vocabulary and tokenizer formats. No byte layout is committed here.

## 9. Token datasets and batches

| Operation | Shapes and semantics | Dtype/device/ownership/errors/gradient |
|---|---|---|
| `token-dataset-open config tokenizer` | Open a deterministic, finite or explicitly streaming token source. Tokenizer fingerprint must match source metadata. | New mutable receiver; CPU control plane; config/I/O/version/checksum/unsupported errors; no gradient. |
| `token-dataset-next-batch dataset` | Return end-of-stream or an immutable batch with `input-ids i64[N,T]`, `targets i64[N,T]`, `loss-mask bool[N,T]` or `f32[N,T]`. Targets are the declared next-token shift; all IDs are in `[0,V)`. | New contiguous tensors on configured verified device; no implicit copy/fallback; errors above plus `invalid-state`; no gradient. |
| `token-dataset-end? value` | True only for the unique immutable end-of-stream sentinel returned by `token-dataset-next-batch`. | CPU; no mutation or gradient. |
| `token-dataset-cursor dataset` | Snapshot all ordering/shuffle/packing/shard offsets needed to reproduce the next batch exactly. | New immutable CPU value; `invalid-state`; no gradient. |
| `token-dataset-seek! dataset cursor` | Validate identity/version/checksum then make the next batch identical to the captured continuation. | Atomic receiver mutation; `invalid-argument`, `version-mismatch`, `corrupt-data`, `invalid-state`; no gradient. |
| `token-dataset-close! dataset` | Release resources; idempotent. | Mutates receiver only; `io`; no gradient. |
| `token-batch-inputs batch` | Return newly owned `i64[N,T]`. | Same device, contiguous; no gradient. |
| `token-batch-targets batch` | Return newly owned `i64[N,T]`. | Same device, contiguous; no gradient. |
| `token-batch-loss-mask batch` | Return newly owned `bool[N,T]` or nonnegative finite `f32[N,T]`. | Same device, contiguous; no gradient. |
| `token-batch-validate batch` | Check ranks/extents/dtypes/device/contiguity/ranges and require positive total mask weight. | Returns `#t`; structured errors; no gradient. |

No batch dimension, sequence dimension, or mask broadcasting is permitted. D1/D2 own
the shard and cursor formats.

`token-dataset-seek!` commits only after complete cursor validation. A failed
`token-dataset-next-batch` does not advance the cursor. `token-dataset-close!` always
transitions to closed after best-effort release; an I/O release failure is reported,
but later calls still observe closed.
After end-of-stream, repeated `token-dataset-next-batch` calls return the same sentinel
without changing state until seek or close.

## 10. Modules and models

| Operation | Contract | Ownership/errors/gradient |
|---|---|---|
| `module-parameters module` | Stable lexical-path tree of opaque live handles, one per unique trainable leaf plus tie metadata. Each leaf is contiguous `f32` of arbitrary documented rank on the module device. | New tree retaining module-bounded handles; `invalid-state`; handles carry gradient state. |
| `module-buffers module` | Deep immutable state-dict snapshot limited to uniquely named contiguous `bool`, `i64`, or `f32` non-trainable leaves of arbitrary documented rank on the module device. Inspect with `state-dict-paths`/`state-dict-tensor`; alias groups are empty. | New owned tensors; `invalid-state`; no buffer gradient or live handle. |
| `module-state-dict module` | Deep data-only snapshot of parameters, buffers, names, shapes, dtypes, devices, and aliases. | New owned state; `invalid-state`, `unsupported`; no new graph. |
| `module-load-state-dict! module state` | Strict exact-name/shape/dtype/device load; preserve declared ties; no partial load in first release. | Atomic receiver mutation; structured mismatch/version errors; no gradient graph. |
| `module-train! module` / `module-eval! module` | Set explicit mode recursively. | Mutates mode only; `invalid-state`; no gradient. |
| `module-zero-grad! module` | Clear all parameter gradients to the runtime's documented absent-gradient state. | Mutates gradient slots only; `invalid-state`, `unsupported`; no gradient. |
| `model-forward model input-ids . opts` | `i64[N,T]` to one opaque model output. The only options are `':targets i64[N,T]`, `':loss-mask bool-or-f32[N,T]`, `':rng rng-state`, and `':deterministic? bool`. Targets and mask appear together; stochastic train mode requires RNG. | Inputs borrowed; no mutation. Output logits/loss differentiate w.r.t. floating parameters, never IDs/targets/mask. Tensor/unsupported/determinism errors apply. |

`model-output-logits` returns new contiguous floating `[N,T,V]`.
`model-output-loss` returns scalar `f32` when targets/mask were supplied and `#f`
otherwise. `model-output-rng` returns the new immutable RNG state, the unchanged input
state only when no stochastic operation consumed it, or `#f` if no RNG was supplied.
Deterministic seeded stochastic execution still advances RNG reproducibly. Indexed LM
loss is `sum(mask * per-token-loss) / sum(mask)` with boolean mask cast to 0/1 and
`f32` accumulation; zero total mask is `invalid-argument`. Accessors do not mutate or
create a new graph; logits/loss retain the forward graph.

Forward must not mutate parameters, buffers, inputs, or a cache unless an explicit
mutable cache receiver is supplied by a later accepted contract. Training mode may
consume explicit RNG state; it must not use hidden global randomness. P1 owns module
representation; N2/A2/L2/M3 own numerical implementation and gradient evidence.

Parameter paths are immutable lists of nonempty UTF-8 segments; no separator escaping
exists. Ordering is lexicographic by UTF-8 bytes segment-by-segment. Repeated
`module-parameters` calls return new tree containers whose unique handles have the
same module identity and whose complete logical path set is stable.
`parameter-tree-paths` returns every logical path, including aliases, exactly once.
`parameter-tree-handle` accepts every returned path; tied paths return the same handle
identity. Each handle's canonical path is the lexicographically first path in its tie
group, or its sole path when untied, and `parameter-handle-path` returns that canonical
path. `optimizer-create` binds the unique handle set, not the transient tree container.
Optimizer parameter groups use canonical paths only, include each unique handle in
exactly one group, and reject alias paths or duplicate handles as `invalid-argument`.
`parameter-tree-paths`, `parameter-tree-handle`,
`parameter-tree-tie-groups`, `parameter-handle-path`, `parameter-handle-shape`,
`parameter-handle-dtype`, and `parameter-handle-device` inspect it.
Paths returned by `parameter-tree-paths` are in the ordering above; an unknown lookup
path is `invalid-argument`. Tie groups contain at least two logical paths, with paths
sorted within each group and groups sorted by their first path; each tied path appears
in exactly one group and the first path is the handle's canonical path.

A state dict is an immutable data-only value. `state-dict-paths` returns every logical
parameter and buffer path in the ordering above. `state-dict-tensor` accepts every
returned path and returns a new owned tensor; tensors returned for tied paths are
value-equal independent snapshots, while `state-dict-alias-groups` preserves the tie
relationship. A strict load requires tied-path payloads to be equal before copying
once into the destination handle; conflicting alias payloads are `corrupt-data`.
`state-dict-alias-groups` returns tied paths in the same canonical group ordering.
Unknown paths are `invalid-argument`. Accessors are CPU control-plane, non-mutating,
and do not create gradient graphs.

## 11. Optimizer

| Operation | Contract | Ownership/errors/gradient |
|---|---|---|
| `optimizer-create config parameter-tree` | Validate unique paths, groups, dtypes/devices, hyperparameters, alias graph, and stable module/handle identities. | New mutable receiver retaining parameter handles; mismatch/unsupported errors; no gradient. |
| `optimizer-step! optimizer` | Require valid finite gradients as configured, compute one atomic update, and advance step/RNG/scheduler state only on success. | Mutates bound parameters and optimizer state; no implicit clipping, precision conversion, fallback, or approximate gradient. `invalid-state`, shape/dtype/device/unsupported/determinism errors. |
| `optimizer-zero-grad! optimizer` | Clear gradients of bound unique parameters once. | Mutates gradient slots; `invalid-state`; no gradient. |
| `optimizer-state optimizer` | Deep snapshot keyed by stable parameter paths, including groups, step counters, schedules, and precision policy. | New owned state; `invalid-state`; no graph. |
| `optimizer-load-state! optimizer state` | Strict atomic load against bound paths/aliases/shapes/dtypes/devices. | Mutates optimizer state, not parameter values; mismatch/version/corruption errors; no graph. |

O2 owns algorithms and state schema. Mixed precision is optional Wave 4 behavior and
must raise `unsupported` in the first release unless separately verified.

## 12. Trainer

| Operation | Contract | Ownership/errors/gradient |
|---|---|---|
| `trainer-create resolved tokenizer dataset model optimizer` | Validate identities, fingerprints, tree binding, capabilities, modes, device/dtype policy, and reproducibility state. Capture the dataset's current cursor as the immutable epoch-start cursor. | New exclusively mutable state machine retaining receivers; structured mismatch/unsupported errors. |
| `trainer-step! trainer` | Perform one optimizer update from exactly the configured positive integer `accumulation-steps` microbatches: fetch/forward/backward each, normalize by total mask weight, clip once, update once, then advance scheduler/counters. On finite-dataset end-of-stream, increment the epoch count, seek to the captured epoch-start cursor, and continue filling the same update; an empty finite dataset or an end sentinel from a declared streaming dataset is `invalid-state`. | Mutates leased state; returns immutable `f32` metrics. Exact gradients required; no approximations. |
| `trainer-stop-policy max-tokens max-updates max-epochs` | Construct an immutable policy; each limit is `#f` or a positive integer and at least one is present. | CPU; new value; `invalid-argument`; no gradient. |
| `trainer-train! trainer stop-policy` | Repeat updates until the first supplied limit is reached or explicit interrupt occurs; conditions are tested after each committed update. No hidden wall-clock stopping. | Same mutation; returns immutable summary; propagates errors. |
| `trainer-evaluate! trainer dataset` | Require a dataset distinct from the leased training dataset; snapshot its cursor, use eval/no-grad, then restore cursor and model mode on success or failure. Return token-weighted `f32` metrics. | Model parameters and both final cursors unchanged; no gradient. |
| `trainer-state trainer` | Deep complete resume snapshot: model, optimizer, scheduler, RNG, tokenizer fingerprint, dataset cursor, config, versions, and processed-token counters. | New owned data-only state; `invalid-state`, `unsupported`; no graph. |
| `trainer-load-state! trainer state` | Strictly validate all identities and atomically restore the next-step continuation. | Mutates all retained receivers only after validation; corruption/version/determinism/mismatch errors; no graph. |

TR3/C2 own state-machine and exact-resume implementation. Logging is observational and
must not alter numerical order, RNG, cursor, or failure semantics.

`trainer-step!` is transactional across cursor, RNG, gradients, parameters, optimizer,
scheduler, and counters. Before consuming a batch it records the small step-start
cursor, RNG, mode, and counter state. Accumulation may advance only the dataset cursor,
explicit RNG, and transient gradients. Any categorized failure in that phase restores
the recorded cursor/RNG/mode/counters and clears those gradients.

Before parameter mutation, the implementation validates every gradient and update,
preallocates required scratch, and proves that the deterministic commit pass contains
no fallible operation. It may do this with a validation pass plus a recomputed commit
pass and one-parameter-sized scratch; the contract does not require a full model or
optimizer snapshot. The commit point is the first parameter write. From that point the
validated pass completes the parameter/optimizer/scheduler/counter writes without
raising a recoverable public error. Process termination and hardware loss are outside
in-process rollback and are recovered from the last durable checkpoint. Thus a
recoverable failure leaves the whole current step uncommitted without imposing
model-sized rollback storage. `trainer-train!` commits after each successful step; on
failure earlier steps remain committed and only the current step rolls back. Metrics
are opaque immutable maps inspected with `metrics-ref`; unknown keys raise
`invalid-argument`.
`max-tokens` counts positions whose boolean mask is true or whose floating mask is
positive; an update may cross the limit because limits are checked only after commit.
One epoch ends at the finite training dataset's end sentinel. The sentinel itself is
not a microbatch; the trainer seeks to its captured epoch-start cursor before
continuing accumulation. One update may cross one or more epoch boundaries, so an
epoch limit, like a token limit, may be exceeded by the final committed update. The
epoch-start cursor and current epoch count are part of `trainer-state`. A streaming
dataset with `max-epochs` is `invalid-argument`.

All trainer result maps are newly owned CPU values with no gradient. A step map has
exactly `loss` (weighted mean) and `mask-weight` (weight sum) as `f32`, plus `tokens`,
`microbatches`, and `update` as nonnegative `i64` counters. A train summary has
exactly `loss` (weighted mean) and `mask-weight` (weight sum) as `f32`, plus `tokens`,
`updates`, and `epochs` as nonnegative `i64`. An evaluation map has exactly `loss`
(weighted mean) and `mask-weight` (weight sum) as `f32`, plus `tokens` and `batches`
as nonnegative `i64`.
Zero evaluated mask weight is `invalid-state`; no NaN sentinel is returned.

## 13. Generator

| Operation | Shapes and semantics | Dtype/device/ownership/errors/gradient |
|---|---|---|
| `generator-create model tokenizer config` | Validate tokenizer/model `V`, context, sampling policy, capabilities, dtype/device, and explicit RNG seed/state. | New mutable receiver retaining model/tokenizer; mismatch/unsupported errors; no gradient. |
| `generator-prefill! generator input-ids` | Contiguous `i64[N,T]` atomically creates/replaces cache and returns newly owned last-token logits `floating[N,V]`. | Mutates cache only and does not consume RNG; no-grad. Shape/dtype/device/context/unsupported errors. |
| `generator-decode-step! generator token-ids` | Contiguous `i64[N,1]` appends one position and returns `floating[N,V]`; cache length must remain `<= C`. | Mutates cache; output new; no-grad; strict errors, no cache fallback. |
| `generator-generate! generator input-ids` | Prompt is contiguous `i64[N,T]`; callers encode text explicitly. Return an opaque generation output. Each row stops on configured EOS or explicit maximum-new-token count. | Mutates cache/RNG; no-grad. Sampling is greedy or explicit temperature/top-k/top-p; seeded policies reproduce or raise `determinism-unavailable`. |

`generation-output-ids` returns a new list of `N` contiguous `i64[Gi]` tensors
containing only generated tokens, `0 <= Gi <= max-new-tokens`, without padding. An
emitted EOS is included in `Gi`, IDs, decoded text input, and cache length.
`generation-output-lengths` returns new `i64[N]`; `generation-output-text` returns a
new list of `N` decoded values; `generation-output-rng` returns the immutable
post-generation RNG; `generation-output-cache-lengths` returns new CPU `i64[N]` equal
to prompt length plus `Gi`, including EOS. ID tensors are on the generator device;
lengths/text/RNG metadata are CPU values. Accessors are no-grad/non-mutating and raise
`invalid-argument` for the wrong output type or `device-mismatch` if retained output
storage is inconsistent.

Greedy and probability-sort ties choose the lowest token ID. Top-k orders probability
descending then token ID ascending. Top-p uses that order and the shortest prefix
whose cumulative probability meets `p`. Each emitted token is a commit point. If a
later token fails, cache/RNG retain the last successful token and the call raises
without returning a partial output.

Cache and no-cache parity is required before a cache path is advertised. G3 owns the
sampling and cache implementation.

## 14. Persistence boundary

| Operation | Contract | Ownership/errors/gradient |
|---|---|---|
| `persistence-policy max-file-bytes max-metadata-bytes max-tensor-bytes max-tensors device` | Construct immutable hard limits and exact target device; integer limits are positive. | New CPU value; `invalid-argument`; no gradient. |
| `checkpoint-inspect path policy` | Validate envelope, policy bounds, versions, entry table, and checksums without constructing executable objects; return checkpoint metadata only. | New immutable CPU metadata; `io`, `corrupt-data`, `version-mismatch`, `unsupported`; no gradient. |
| `checkpoint-load path policy capability-report` | Verify all bytes/checksums/limits before exposing a data-only state snapshot; reject code, callbacks, foreign paths, and unknown required features. Require verified tensor/device capabilities for the policy device. Return exactly a deep-owned trainer state accepted by `trainer-load-state!`. | New owned CPU control metadata plus tensors on exactly the policy device; no implicit conversion/transfer/fallback and no graph. |
| `checkpoint-save! state path policy options` | Accept only a complete `trainer-state` result, validate it and policy limits, write a same-directory temporary file, flush as required, then atomically replace target. | State borrowed and unchanged; `io`, `invalid-state`, `unsupported`; no graph. |

The only save options are `':overwrite? bool` and
`':required-features symbol-list`. `checkpoint-metadata-ref metadata key` returns a
new immutable CPU value and raises `invalid-argument` for an unknown key; it has no
gradient. Required metadata keys and values are: `format-id` symbol,
`format-version` immutable `(major minor)` nonnegative-integer list,
`required-features` sorted symbol list,
`checksum-algorithm` symbol, `api-version` string, `config-schema-version` immutable
`(major minor)` nonnegative-integer list, `tokenizer-fingerprint` string,
`config-fingerprint` string,
`tensor-count` nonnegative `i64`, and `payload-bytes` nonnegative `i64`. These are
logical API metadata, not serialized field names or byte layout. Save commits only at
atomic replacement. Before that point the target is unchanged, though a separate
temporary may remain after I/O failure.

Checkpoint, tokenizer, token-shard, cursor, resolved-config, and native ABI formats are
separate version domains. This contract requires an envelope with a format identifier,
major/minor version, required-feature list, declared size limits, checksum algorithm,
and checksums. It deliberately commits to **no magic bytes, encoding, field numbers,
container layout, tensor encoding, or migration algorithm**. T1/D1/C1/X1/K1 must
propose those decisions through issue #1 before merging a public format or ABI.

Data loaders must enforce configurable hard limits before allocation and must never
evaluate code, resolve arbitrary object constructors, follow embedded paths, or load
native libraries.

## 15. Independent versioning policy

- Library API: SemVer string, currently pre-1.0. Removing/changing a public name or
  semantic contract is a breaking API change.
- Native ABI: independent `(major, minor)`. Major changes calling convention or
  binary layout; minor adds discoverable backward-compatible entry points.
- Each file/schema family: independent `(major, minor)` plus required-feature bits.
  Major changes require an explicit migration/rejection decision; minor readers must
  ignore only declared optional fields.
- Canonicalization/fingerprint algorithms: named and versioned independently. Their
  identifier is included in the digest input/output.
- Capability evidence: implementation and evidence versions, never a compatibility
  promise and never persisted as proof on another host.

Version numbers do not imply capability support. Readers reject unknown major
versions and unknown required features; they do not guess.

## 16. Acceptance boundary and downstream ownership

This A0 contract unblocks design work but is not accepted until the fixtures compile
and negative/repeatability checks pass on the configured compiler. Ownership is:

- P1: opaque module/parameter tree, alias graph, state dict;
- T1/T2: tokenizer behavior and formats;
- D1/D2: shard/batch/cursor behavior and formats;
- K1/R0: native ABI and verified capability evidence;
- C1/C2: safe checkpoint envelope and complete resume state;
- X1: configuration schema/canonicalization;
- numerical/model/training/generation workstreams: operation implementation and all
  forward, gradient, determinism, cache, and performance evidence.

Any incompatibility found by those workstreams is a proposed contract change for
issue #1; it must not be silently reinterpreted.
