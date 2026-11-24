# Continuous Batching & Chunked-Prefill Inference Scheduler

This document specifies the standalone, Eshkol-native **Continuous Batching &
Chunked-Prefill Inference Scheduler**, its C11/C++17 public ABI, zero-heap
scheduling hot path, deterministic preemption rules, conservation invariants,
compiled Eshkol AOT driver, and measured verification evidence.

## Artifacts and public headers

- Public C11/C++17 ABI: `include/eshkol_transformer/continuous_batch_abi.h`
- Core scheduler library: `native/continuous_batch.c` -> `build/continuous_batch/libeshkol_transformer_continuous_batch.a`
- Eshkol AOT bridge: `native/continuous_batch_eshkol_bridge.h`, `native/continuous_batch_eshkol_bridge.c` -> `build/continuous_batch/libeshkol_transformer_continuous_batch_aot.a`
- Native C11 & C++17 tests: `tests/continuous_batch/test_continuous_batch_native.c`, `tests/continuous_batch/header_cb_cpp.cpp`
- Development Python oracle: `tests/continuous_batch/test_continuous_batch_oracle.py`
- Compiled Eshkol AOT driver: `tests/continuous_batch/test_continuous_batch_aot.esk`
- Build and test scripts: `scripts/build-continuous-batch.sh`, `scripts/test-continuous-batch.sh`

## Configuration bounds and admission contract

| Field | Allowed range | Rejection status |
|---|---|---|
| `max_running_seqs` | `1 .. 64` | `ET_CB_ERR_CONFIG` (`-2`) |
| `max_waiting_seqs` | `1 .. 256` | `ET_CB_ERR_CONFIG` (`-2`) |
| `max_tokens_per_step` | `1 .. 4096` | `ET_CB_ERR_CONFIG` (`-2`) |
| `total_kv_blocks` | `1 .. 4096` | `ET_CB_ERR_CONFIG` (`-2`) |
| `block_size` | `1 .. 64` | `ET_CB_ERR_CONFIG` (`-2`) |
| `policy` | `DECODE_FIRST_CHUNKED_PREFILL` (`0`) or `PREFILL_FIRST_CHUNKED` (`1`) | `ET_CB_ERR_CONFIG` (`-2`) |

### Request admission invariant

Every enqueued request specifies `(request_id, prompt_len, max_new_tokens, priority)`
with `prompt_len >= 1` and `max_new_tokens >= 1`. Over its full lifetime, a
request holds at most `prompt_len + max_new_tokens - 1` KV token entries. The
scheduler enforces at enqueue time:

```text
prompt_len + max_new_tokens - 1 <= total_kv_blocks * block_size
```

Requests violating this bound are rejected immediately with
`ET_CB_ERR_REQUEST_TOO_LARGE` (`-3`). Because `ceil(N / block_size) <= total_kv_blocks`
if and only if `N <= total_kv_blocks * block_size`, every admitted request is
mathematically guaranteed to fit within `total_kv_blocks` as a standalone
sequence (both during initial prefill/decode and during post-preemption
recomputation), eliminating single-request KV deadlock.

Additional fail-closed enqueue/control rejections:

- Duplicate `request_id` (active or historical): `ET_CB_ERR_DUPLICATE_ID` (`-4`)
- Full external waiting queue (`waiting_count >= max_waiting_seqs`): `ET_CB_ERR_QUEUE_FULL` (`-5`)
- Exhausted fixed-capacity request table (`512` slots): `ET_CB_ERR_CAPACITY` (`-6`)
- Unknown `request_id` on cancel/query: `ET_CB_ERR_NOT_FOUND` (`-7`)
- Double-cancel or cancel of a finished request: `ET_CB_ERR_ALREADY_TERMINAL` (`-8`)
- Double `step_plan` without commit, or cancelling a sequence active in an uncommitted plan: `ET_CB_ERR_PLAN_PENDING` (`-9`)
- `step_commit` with no pending plan: `ET_CB_ERR_NO_PENDING_PLAN` (`-10`)
- Commit mismatch (wrong `step_index`, wrong `item_count`, wrong `request_id` order, `generated_token_count != produces_output_token`, or `hit_eos == 1` on an intermediate prefill chunk): `ET_CB_ERR_COMMIT_MISMATCH` (`-11`), leaving all scheduler state untouched.

## Request lifecycle state machine

```text
                 admit (chunk >= 1)
   [WAITING] -------------------------> [PREFILLING]
      ^  |                                 |   ^  |
      |  | cancel                          |   |  | final prefill chunk commits:
      |  v                    KV exhausted |   |  |   gen += 1; if (eos || gen == max_new) -> FINISHED
      | [CANCELLED]             (preempt)  |   |  |   else -> DECODING
      |                                    v   |  v
      |         re-admit (front of queue)  [PREEMPTED]  [DECODING] --(eos || gen == max_new)--> [FINISHED]
      |         recompute prompt+gen toks      |            |
      +----------------------------------------+            | cancel (not in pending plan)
                (cancel -> CANCELLED)                       v
                                                        [CANCELLED]
```

- **State preservation on preemption**: When a running sequence (`PREFILLING` or `DECODING`) is preempted, all of its KV blocks are immediately returned to the free pool (`allocated_blocks = 0`), `computed_tokens` resets to `0` for recomputation, `preempt_count` increments by `1`, and `prompt_len`, `max_new_tokens`, `generated_tokens`, `first_token_step`, `priority`, and `arrival_seq` are preserved verbatim.
- **Re-admission after preemption**: The preempted sequence is placed at the front of `waiting_queue` (internal buffer capacity `256 + 64 = 320`, so preemption never fails even when external `waiting_count == max_waiting_seqs`). Upon re-admission, its recomputation prefill target is `prompt_len + generated_tokens`; completing that chunked prefill emits token `generated_tokens + 1` without overwriting its original `first_token_step`.

## Step planning, chunked prefill, and preemption rules

Two scheduling policies are supported:

1. **`ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL` (`0`)**:
   - **Phase 1 (Running decodes)**: Sort running `DECODING` sequences by `(priority DESC, arrival_seq ASC)`. Each requires `1` token of step budget and `new_blocks = (computed_tokens % block_size == 0) ? 1 : 0`.
   - **Phase 2 (Continuing prefills)**: Sort running `PREFILLING` sequences by `(priority DESC, arrival_seq ASC)`. With `slack = allocated_blocks * block_size - computed_tokens` and `max_fit = slack + free_kv_blocks * block_size`, allocate `chunk_tokens = min(remaining_prefill, remaining_budget, max_fit)` and `new_blocks = ceil((computed_tokens + chunk_tokens) / block_size) - allocated_blocks`.
   - **Phase 3 (Admit waiting)**: Skipped if any preemption occurred in this step (preventing eviction/re-admission thrashing). Otherwise pop from the front of `waiting_queue` while `running_count < max_running_seqs`, `remaining_budget > 0`, and `free_kv_blocks > 0`, computing `chunk_tokens = min(prompt_len + generated_tokens, remaining_budget, free_kv_blocks * block_size)` and `new_blocks = ceil(chunk_tokens / block_size)`.
2. **`ET_CB_POLICY_PREFILL_FIRST_CHUNKED` (`1`)**:
   - Executes continuing prefills first, waiting admission second (skipped if any preemption occurred this step), and running decodes third.

### Deterministic preemption and zero-deadlock guarantee

When a running sequence needs a new KV block (`new_blocks == 1` in decode or `max_fit == 0` in continuing prefill) while `free_kv_blocks == 0`:

1. Select the lowest-rank (`priority ASC`, tie-break `arrival_seq DESC`) running sequence `V` that is not already scheduled in the current step's pending plan.
2. If `V` is the requesting sequence itself and `plan.entry_count > 0`, defer the requesting sequence for this step rather than self-evicting while other work is already advancing.
3. Otherwise preempt `V`, releasing all `V.allocated_blocks >= 1` KV blocks back to the free pool and pushing `V` to the front of `waiting_queue`.
4. Suppress waiting-queue admission for the remainder of the current step (`preempted_count > 0`).

Because every admitted request individually fits in `total_kv_blocks`, the highest-rank running sequence is never evicted and strictly advances until completion, guaranteeing 100% completion with zero deadlock and zero livelock under both policies.

## Conservation and hot-path invariants

- **Zero heap allocations**: `native/continuous_batch.c` does not import `malloc`, `calloc`, `realloc`, or `free` (enforced by `nm -u` symbol audit in `scripts/build-continuous-batch.sh`), and `tests/continuous_batch/test_continuous_batch_native.c` links with `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free` to trap any dynamic allocation during `et_cb_init`, `et_cb_enqueue`, `et_cb_cancel`, `et_cb_step_plan`, `et_cb_step_commit`, stats/info queries, and all per-step AOT bridge calls.
- **Physical KV-block ownership conservation** (`et_cb_verify_invariants`):
  - `used_kv_blocks + free_kv_blocks == total_kv_blocks` at every step (post-plan and post-commit).
  - Every physical block `0 .. total_kv_blocks - 1` appears exactly once in either the free list (`owner == ET_CB_BLOCK_NONE`) or one running slot's block chain.
  - Non-running (`WAITING`, `PREEMPTED`, `FINISHED`, `CANCELLED`) slots hold `0` KV blocks; when all requests finish or cancel, `free_kv_blocks == total_kv_blocks` and `used_kv_blocks == 0`.
  - Every planned batch satisfies `sum(chunk_tokens) <= max_tokens_per_step` and `active_running_seqs <= max_running_seqs`.

## Measured verification results

Verified inside the pinned Ubuntu 22.04 container (`eshkol-r0-debug:20260928`,
Clang 21.1.8, pinned Eshkol `fe9dfd52`) via `bash scripts/test-continuous-batch.sh`,
`python3 -m unittest -v tests/q0/test_python_isolation.py`, and
`bash scripts/check-ci-topology.sh`:

| Check | Measured result |
|---|---|
| Core deterministic archive (`libeshkol_transformer_continuous_batch.a`) | SHA-256 `ae9fdda84886fc4ee5408700aecde39b98fc60b4f1258bb798ef6a5893380fc8` (zero heap imports) |
| Bridge deterministic archive (`libeshkol_transformer_continuous_batch_aot.a`) | SHA-256 `5900b652ef53a0523d297c50c77b3592e48abcc926b14bc0a63fe2455a6fbdcb` |
| C++17 header & archive linkage (`clang++-21 -std=c++17`) | PASS (normal + ASan/UBSan/LSan) |
| Native C11 suite (`-O2`, `--wrap=malloc/calloc/realloc/free`) | PASS (0 heap allocations on all hot-path ops) |
| Independent Python oracle (`tests/continuous_batch/test_continuous_batch_oracle.py`) | PASS (4 scenarios, 44 steps, 70 plan entries, 6 preemptions; exact step-by-step match) |
| Sanitizers (`ASan`, `UBSan`, `LSan detect_leaks=1:halt_on_error=1`) | PASS (C11 + C++17 suites) |
| Compiled Eshkol AOT (`tests/continuous_batch/test_continuous_batch_aot.esk`) | Byte-identical ELF & stdout across 2 fresh builds (`--strict-types --optimize 0 --no-stdlib`); ELF SHA-256 `5b59a757d8260a408b41953e9b8a4a8708b787747fb990f8921ba437b92f0ef4` |
| Python isolation & CI topology gates | PASS (`5/5` isolation tests, `23` shared CI suites) |

### Measured compiled Eshkol AOT workload trace

Configuration: `max_running_seqs=3`, `max_waiting_seqs=16`,
`max_tokens_per_step=8`, `total_kv_blocks=6`, `block_size=4` (`24` token capacity).
Arrivals: step `0` -> Req `101` (`prompt=15, max_new=3, prio=2`), Req `102`
(`prompt=3, max_new=5, prio=1`, EOS on 2nd generated token), Req `103`
(`prompt=7, max_new=4, prio=0`); step `2` -> Req `104` (`prompt=8, max_new=4, prio=0`,
cancelled at step `4`); step `3` -> Req `105` (`prompt=11, max_new=2, prio=1`).

#### `DECODE_FIRST_CHUNKED_PREFILL` step trace

| Step | Entries | Total toks | Prefill toks | Decode toks | Preemptions | Post-commit `used_kv` | Post-commit `free_kv` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 8 | 8 | 0 | 0 | 2 | 4 |
| 1 | 2 | 8 | 8 | 0 | 0 | 5 | 1 |
| 2 | 3 | 7 | 6 | 1 | 0 | 6 | 0 |
| 3 | 2 | 2 | 0 | 2 | 1 | 0 | 6 |
| 4 | 2 | 8 | 8 | 0 | 0 | 3 | 3 |
| 5 | 2 | 8 | 7 | 1 | 0 | 4 | 2 |
| 6 | 2 | 4 | 3 | 1 | 0 | 6 | 0 |
| 7 | 2 | 2 | 0 | 2 | 0 | 0 | 6 |

#### Per-request final accounting (`DECODE_FIRST_CHUNKED_PREFILL`)

| Request ID | Final state | Generated tokens | TTFT (steps) | Completion step | Preemptions | EOS stopped |
|---:|---|---:|---:|---:|---:|---:|
| `101` | `FINISHED` (`4`) | 3 | 2 | 3 | 0 | 0 |
| `102` | `FINISHED` (`4`) | 2 | 3 | 3 | 0 | 1 |
| `103` | `FINISHED` (`4`) | 4 | 5 | 7 | 1 | 0 |
| `104` | `CANCELLED` (`5`) | 0 | 0 | 4 | 0 | 0 |
| `105` | `FINISHED` (`4`) | 2 | 4 | 7 | 0 | 0 |

#### Policy comparison summary (identical workload)

| Policy | Total steps | Finished | Cancelled | Preemptions | Generated tokens | Scheduled tokens | Final `free_kv` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DECODE_FIRST_CHUNKED_PREFILL` | 8 | 4 | 1 | 1 | 11 | 47 | 6 / 6 |
| `PREFILL_FIRST_CHUNKED` | 9 | 4 | 1 | 1 | 11 | 47 | 6 / 6 |

## Limitations and scope

- Fixed-capacity single-threaded CPU scheduler supporting up to `64` concurrent running sequences, `256` external waiting requests (`320` internal queue slots including preempted sequences), `512` lifetime tracked request records per scheduler instance, `4096` KV blocks, and `block_size <= 64`.
- Preemption uses recomputation (evicting KV blocks and re-prefilling `prompt_len + generated_tokens` on re-admission); CPU-offload swap-out, prefix-cache block sharing (copy-on-write radix attention), speculative decoding multi-token decode commits, and GPU kernel dispatch are outside this standalone scheduler's scope.
- `tests/continuous_batch/test_continuous_batch_oracle.py` is a development-only verification oracle; the delivered scheduler and AOT driver are fully Eshkol/C-native.
