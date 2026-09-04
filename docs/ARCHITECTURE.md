# Architecture

## Scope

`eshkol-transformer` owns the Eshkol-facing model, tokenizer, data, training,
checkpoint, evaluation, generation, configuration, and interoperability APIs.

Eshkol core or a reviewed native extension must provide efficient tensor storage,
device execution, and reverse-mode kernels. The library must not disguise missing
runtime capabilities with scalar implementations in production paths.

## Primary abstractions

- **Tokenizer**: deterministic text-to-token and token-to-text conversion with a
  stable fingerprint.
- **Token dataset**: a resumable stream of `(input-ids, targets, loss-mask)` batches.
- **Module**: a nested collection of named parameters, named buffers, configuration,
  and a forward operation.
- **Model**: a module whose forward path produces token logits and optional cache.
- **Optimizer**: parameter-group updates plus serializable state.
- **Trainer**: the state machine connecting data, model, loss, optimization,
  evaluation, logging, and exact resume.
- **Generator**: prefill/decode execution with KV cache and explicit sampling policy.

## Dependency direction

```text
tokenizer ───────┐
corpus/shards ───┼─> token loader ─────────────┐
                 │                             │
tensor backend ──┼─> modules/layers ─> model ──┼─> trainer ─> checkpoint
parameter tree ──┘                 │           │
                                   ├─> generation
indexed LM loss ───────────────────┴─> evaluation
```

Lower layers cannot import trainer, CLI, or model-family modules. File formats and
public contracts are versioned independently of implementations.

## Tensor contracts

- Token IDs are integer tensors shaped `[batch, sequence]`.
- Hidden states are floating tensors shaped `[batch, sequence, hidden]`.
- Attention projections explicitly document head and grouped-query layouts.
- Logits are `[batch, sequence, vocabulary]`.
- Indexed LM targets are `[batch, sequence]`; padding/document exclusions are carried
  by a boolean or numeric loss mask of the same leading shape.
- Every operation declares accepted dtypes, devices, contiguity, broadcasting, and
  gradient support.

## Native boundary

The first required native/runtime kernels are batched matmul, embedding scatter-add,
LayerNorm/RMSNorm, activations, causal attention, and indexed cross-entropy. Native
entry points require ABI versioning, capability discovery, shape guards, deterministic
test modes, and explicit unsupported errors.

## Persistence

Checkpoints are data, never executable code. A complete training checkpoint contains
model configuration and tensors, optimizer/scheduler state, RNG state, tokenizer
fingerprint, dataset cursor, resolved run configuration, library/compiler versions,
and total tokens processed. Writes use a temporary file plus atomic replacement.

Native tensor ownership is explicit. A provider-owned clone has one ledger owner and
must either transfer into one live receiver state or be released exactly once; GC
reachability and unproved finalizers are never reclamation mechanisms. P1 state
dictionaries expose idempotent release and only read-only state-backed handles.
Trusted consumers validate and borrow those handles synchronously, end the borrow in
the same call, and retain no raw native pointer. Optimizer snapshots require their
own versioned receiver ledger and release API; they cannot reuse P1's public state
release as generic construction or destruction authority.
