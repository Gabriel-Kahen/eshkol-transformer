# DIST5 data-parallel collectives

DIST5 ABI 1.0 is a synchronous C11 collective library. Build it with
`bash scripts/build-dist5.sh` and link `build/dist5/libeshkol_dist5.a` with
`-lm`. The public header is `include/eshkol_transformer/dist5_collectives_abi.h`
and supports C++17 linkage.

## Communicator and collectives

Create one connected `AF_UNIX` `SOCK_STREAM` socket pair per nonzero rank
before `fork()`, or provide equivalent pre-connected endpoints. Rank 0 takes
one endpoint for each rank in ascending peer-rank order. Other ranks take
their endpoint to rank 0. Size one uses zero descriptors. The constructor
duplicates descriptors with close-on-exec and borrows the originals. Close
the originals and unused inherited endpoints before entering a collective so
the kernel reports a crashed peer as soon as its endpoint closes. The
application controls process startup and shutdown.

Use one caller per communicator and call matching collectives in the same
order. Tags may repeat because the protocol also checks a sequence counter.
Payloads are packed native little-endian binary32 and signed 64-bit integers.
The operation tables below use `W` for the world size and `r` for the local rank.

| API | Caller buffer contract |
|---|---|
| `et_dist5_barrier` | No payload. All ranks rendezvous at the same sequence and tag. |
| `et_dist5_broadcast` | Each rank supplies a writable `count`-element buffer. The chosen root's bytes become every rank's result. |
| `et_dist5_allreduce` | Each rank supplies `count` elements and receives the same reduced `count` elements. Exact in-place input and output are supported. |
| `et_dist5_allgather` | Each rank supplies `send_count` elements and receives `W * send_count` elements concatenated in ascending rank order. Input and output must be disjoint. |
| `et_dist5_reduce_scatter` | Each rank supplies `W * recv_count` elements. After reduction, rank `r` receives slice `[r * recv_count, (r + 1) * recv_count)`. Input and output must be disjoint. |
| `et_dist5_sync_gradients_f32` | Each rank supplies `count` gradients and receives the equal-rank mean. Exact in-place input and output are supported. Apply optimizer updates only after success. With uneven local batch sizes, weight contributions explicitly before synchronizing. |

Reductions seed the accumulator with rank 0's operand, then combine ranks
`1..W-1` in order, regardless of arrival time. Binary32 SUM and MEAN round
each addition under the default floating-point environment. MEAN divides once
by `W` after the sum. MIN and MAX retain the earlier rank's bits on ties,
including signed zeros. The coordinator restores its incoming floating-point
environment and sends the computed bytes to each rank. I64 supports SUM, MIN,
and MAX. An intermediate I64 sum outside the signed range returns
`ET_DIST5_ERR_INVALID_ARG`. A nonfinite F32 reduction operand or sum returns
`ET_DIST5_ERR_NONFINITE`. Broadcast and allgather preserve raw payload bits.
Bitwise reproducibility applies to identical rank operands in the validated
floating-point environment. Different microbatch partitions can change f32
parenthesization.

## Wire format and failures

Frame words are consecutive host-order `uint64_t` values with no struct padding.
The protocol assumes trusted, homogeneous processes. Each operation starts at
sequence zero on a newly created communicator and advances the sequence after
each call that enters the protocol.

1. Workers send an 11-word request containing magic `0x4554444953540501`,
   sequence, sender rank, world size, kind, tag, count, dtype, op, broadcast
   root, and local status. Kind values 1 through 6 denote barrier, broadcast,
   allreduce, allgather, reduce-scatter, and gradient sync. The unused fields
   are zero. Gradient sync checks local finiteness and includes the verdict in
   this request, before any rank sends gradient payloads.
2. After matching requests, rank 0 sends each contributing worker a GRANT.
   That worker sends the exact number of bytes agreed in preflight. Rank 0
   consumes inputs in ascending rank order and stages the result.
3. Rank 0 sends each worker a RESULT followed by its result bytes. Even for a
   barrier, each worker acknowledges staging by sending the sequence word.
4. After receiving all acknowledgments, rank 0 sends COMMIT. Each worker
   publishes its local destination after consuming COMMIT. Rank 0 publishes
   after sending the final COMMIT.

Reply frames contain sequence, phase, and status. Phase values are GRANT (1),
RESULT (2), COMMIT (3), and ABORT (4). ABORT carries a nonzero rejection or
transport status. Transport errors during a partial outbound frame or payload
are signaled on that socket by shutdown because the peer cannot reliably parse
a control frame there.

- Invalid operands, allocation failures, and nonfinite inputs produce a
  coordinated rejection when all peers are participating. Destinations remain
  unchanged, and the communicator can be reused. A null communicator or
  nonpositive timeout is rejected locally before protocol entry.
- A kind or tag mismatch returns `ET_DIST5_ERR_MISMATCH` even when an operand
  is invalid. Other call parameters are compared when both ranks pass argument
  validation. Metadata mismatches, IO errors, timeouts, and peer disconnections
  make the communicator unusable. Future calls with a valid timeout return the
  stored failure. Destroy and bootstrap again.
- Transport waits share one monotonic deadline per call. Nonblocking stream
  I/O handles partial transfers and interruption. Rank 0 monitors all active
  peer sockets while waiting. It sends the failure status where a reply frame
  can still be delivered, then shuts down the streams. A local deadline
  reports `ET_DIST5_ERR_TIMEOUT`. EOF without a complete status frame reports
  `ET_DIST5_ERR_PEER_DISCONNECTED`, including when the coordinator itself
  exits. The application should allow for different transport statuses when
  ranks have different deadlines or when a control frame cannot be delivered.
- A rank copies to the caller only on local success. A death before all
  acknowledgments leaves surviving destinations untouched. If a process dies
  during final COMMIT delivery, some ranks may already have returned success.
  Restart the group from a known-good common training state after a transport
  failure. The protocol provides no persistent recovery or distributed commit
  guarantee across process death.

The transport is single-host Unix socket IPC using Linux socket flags. It
uses neither filesystem socket paths nor shared memory. The coordinator is a
serial bottleneck. The ABI caps the world at 64 ranks and each full operand or
result at 64 MiB, including expanded gather or scatter spans. Rank 0 may stage
up to two such buffers. Remote hosts, heterogeneous peers, concurrent
collectives on one communicator, RDMA, NCCL/RCCL, and GPU buffers are outside
this implementation. Throughput and scale beyond the smoke workloads have
not been measured.

## Validation

`CC=clang-21 CXX=clang++-21 DIST5_ASAN_DETECT_LEAKS=1 bash scripts/test-dist5.sh`
runs the archive and C++17 linkage checks, then runs the C suite twice, in an
optimized build and under ASan/UBSan/LSan. The suite runs 28 independent fork
groups per build, covering bitwise collective outputs under reversed rank
start order, 128 KiB broadcasts through constrained socket buffers, packed-byte
reductions, malformed inputs and mismatched calls, and process exits or stalls
during metadata, payload, or result acknowledgment. It checks descriptor counts
after rejected bootstrap calls and the call-wide timeout during a slow payload
transfer.

Training uses MOD5 K1 RMSNorm, causal multi-query attention, and SwiGLU
forward/backward on eight deterministic microbatches and checks 36 block
parameter derivatives by central finite differences. Over 16 SGD steps, every
rank's weights match the other ranks and the single-process rank-ordered
reference bitwise at every step. Repeated full traces also match bitwise.
Single-rank NaN and Inf injections preserve the weights, with training
resuming after the coordinated rejection.

| World size | Initial mean loss | Final mean loss | Maximum weight delta from W=1 |
|---|---:|---:|---:|
| 1 | 0.0384391658 | 0.0193505306 | 0 |
| 2 | 0.0384391658 | 0.0193505306 | 1.1920929e-7 |
| 4 | 0.0384391695 | 0.0193505287 | 0 |

The tested lane is x86-64 Ubuntu 22.04.5 with Clang 21.1.8 in
`eshkol-r0-debug:20260928`. The training harness uses a test-only block with
diagonal projections. TR3 integration and hosted CI admission are separate
gates.

## Eshkol AOT bridge and multi-process training

The Eshkol AOT bridge (`native/dist5_eshkol_bridge.h`,
`native/dist5_eshkol_bridge.c`) builds as deterministic static archive
`build/dist5_aot/libeshkol_transformer_dist5_aot.a` via
`bash scripts/build-dist5-aot.sh` (27 `et_dist5_aot_*` C11/C++17 symbols) and
links against `libeshkol_dist5.a`, `libeshkol_transformer_mod5.a`, and
`libeshkol_transformer_k1.a`.

`et_dist5_aot_open` bootstraps independent OS rank processes over an `AF_UNIX`
`SOCK_STREAM` rendezvous socket (`<sock_dir>/dist5_r0.sock`) with a shared
monotonic deadline, validates rank/world/ABI handshake frames, rejects duplicate
ranks and overlong paths, unlinks the listener socket after rendezvous, closes
all temporary descriptors after `et_dist5_comm_create` duplicates them, and
discovers the MOD5 K1 runtime on the same handle. Every collective wrapper
accepts `void *` pointers, checks 4-byte binary32 alignment without undefined
typed casts, and routes invalid operands through the DIST5 protocol so peers
reject cleanly without deadlock.

`bash scripts/test-dist5-aot.sh` verifies:

- deterministic byte-identical archive build, symbol allowlists, and C++17
  header/linkage (`tests/dist5/header_bridge_cpp.cpp`);
- native C11 bridge unit and multi-process tests
  (`tests/dist5/test_dist5_aot_bridge.c`) in both `-O2` and ASan/UBSan/LSan
  (`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1`)
  modes, covering FD-leak invariance, misaligned pointers, all six collectives,
  canonical rank-order accumulation, atomic non-finite abort, kind/tag
  mismatches, and duplicate-rank handshake rejection;
- two fresh `--strict-types --optimize 0 --no-stdlib` compilations of
  `tests/dist5/test_dist5_aot.esk` producing byte-identical ELFs (SHA-256
  `61bfe598e88b2d70f671ab011ff359fc99923702e16a742e9ee291988bf5c7b4`);
- live multi-process Eshkol AOT data-parallel training of a full pre-norm MQA
  decoder block (`tokens=8`, `width=16`, `hidden=32`, `H_q=4`, `H_kv=1`, 2,208
  binary32 parameters = 8,832 bytes) using Eshkol `matmul`/tensor ops, MOD5 K1
  RMSNorm/attention/SwiGLU kernels, central finite-difference gradient checks on
  `W=1`, and 10 SGD steps over 4 deterministic microbatches launched in reversed
  rank order.

| Configuration | Initial mean loss (step 0) | Final mean loss (step 10) | Final weights SHA-256 (8,832 bytes) | Bitwise parity |
|---|---:|---:|---|---|
| `W=1` (`accum=4`) | `0.007994377054274082` | `0.005709689110517502` | `e27fb42634d84f1bc53282959b167034e3ff0e53d9a798d919d937eb81de4ac0` | Reference for `W=4` |
| `W=4` (4 OS processes) | `0.007994377054274082` | `0.005709689110517502` | `e27fb42634d84f1bc53282959b167034e3ff0e53d9a798d919d937eb81de4ac0` | All 4 ranks & `W=1 (accum=4)` bitwise identical |
| `W=1` (`accum=2`) | `0.007994377054274082` | `0.005709689110517502` | `83191052e60bb42abc5ef8b2056b650bf2ea37149906699e9d7f63615963fad5` | Reference for `W=2` |
| `W=2` (2 OS processes) | `0.007994377054274082` | `0.005709689110517502` | `83191052e60bb42abc5ef8b2056b650bf2ea37149906699e9d7f63615963fad5` | Both ranks & `W=1 (accum=2)` bitwise identical |

Loss decreases strictly at every step (`0.007994377054274082` ->
`0.007644729223102331` -> `0.007338153198361397` -> `0.007065237034112215` ->
`0.006819297093898058` -> `0.006595486309379339` -> `0.006390227936208248` ->
`0.006200840231031179` -> `0.0060252845287323` -> `0.005861981306225061` ->
`0.005709689110517502` for `W=1 (accum=4)` and `W=4`). In `--mode nonfinite`
with `W=4`, injecting qNaN (`0x7fc00000`) and `+Inf` (`0x7f800000`) on rank 3
returns `ET_DIST5_ERR_NONFINITE` (`5`) on all four ranks, leaves all 8,832
weight bytes untouched (SHA-256
`28d8283acf4207882601d5f0619bd0f333a453ca1779ff87dd97b5a04004e484`), and recovers
cleanly on the subsequent finite `sync_gradients_f32`. In `--mode mismatch` with
`W=2`, a tag mismatch on rank 1 returns `ET_DIST5_ERR_MISMATCH` (`2`) on both
ranks and leaves the communicator permanently failed on subsequent calls.
