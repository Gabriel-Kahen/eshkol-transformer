# DIST5 data-parallel collectives

DIST5 ABI 1.1 is a synchronous C11 collective library. Build it with
`bash scripts/build-dist5.sh` and link `build/dist5/libeshkol_dist5.a` with
`-lm`. The public header is `include/eshkol_transformer/dist5_collectives_abi.h`
and supports C++17 linkage.

The original ABI 1.0 declarations, symbols, constructor request, and
collective frames are supported by the ABI 1.1 library. The constructor
checks the requested version against its local library. It does not
negotiate a version with peers. Every participant in a weighted call needs
the 1.1 entry point. If a participant instead issues a 1.0 collective at
that sequence, both sides get a fatal kind mismatch. If a participant fails
to enter the exchange, the participating ranks can time out.

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
| `et_dist5_sync_gradients_f32` | Each rank supplies `count` gradients and receives the equal-rank mean. Exact in-place input and output are supported. Apply optimizer updates only after success. |
| `et_dist5_weighted_sgd_f32` | Each rank supplies its effective count and packed objective and gradient means. The call stages the count-weighted global objective, gradient, and plain SGD update. |

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

## Count-weighted SGD

The opt-in `et_dist5_weighted_sgd_f32` entry point takes contiguous native
binary32 buffers. For `P` parameters, each packed mean has shape `[P + 1]`
with the scalar objective first and mean gradients in the same order as the
`[P]` parameter buffer. Local means use the same denominator as the local
effective count. For example, with token masks, divide the local objective
and gradient sum by the number of contributing tokens. Sample weights can be
represented in common integer units. A zero-count rank contributes zeros and
can omit its local mean. It still participates and applies the common update.

The participants agree on the parameter count and bit pattern of the
positive finite learning rate, and must start with identical layouts and
parameter values. Parameters, the packed output, and the I64 count output
occupy distinct spans. Exact in-place local and global means are supported.
Every active input value is checked for finiteness before reduction.

For nonnegative local counts `n_r`, the first exchange computes
`N = sum(n_r)` with checked I64 arithmetic. Zero total or overflow rejects
the step. The second exchange reduces binary32 contributions
`float32((double(n_r) / double(N)) * double(local_mean_r))` in ascending
rank order. Normalizing before the sum avoids multiplying a binary32 mean by
a large count. Count reporting remains exact. Fraction conversion and each
rank contribution can round, so serial arithmetic may differ by normal
binary32 rounding. The caller's floating-point environment is restored.

Each rank then stages the plain SGD update `p_i - learning_rate * g_i`
with separate binary32 multiply and subtract. The third exchange coordinates
update finiteness. The call publishes the new parameters, packed global mean,
and exact count after this exchange succeeds. A recoverable validation or
numeric error in any phase leaves those three destinations unchanged and
keeps the communicator reusable. All three phases share one deadline. This
step supports plain SGD with CPU binary32 arrays. Optimizer state,
checkpointing, canonical TR3 dispatch, and device buffers are outside its
contract.

## Wire format and failures

Frame words are consecutive host-order `uint64_t` values with no struct padding.
The protocol assumes trusted, homogeneous processes. Each operation starts at
sequence zero on a newly created communicator and advances the sequence after
each call that enters the protocol.

1. Workers send an 11-word request containing magic `0x4554444953540501`,
   sequence, sender rank, world size, kind, tag, count, dtype, op, broadcast
   root, and local status. ABI 1.0 kind values 1 through 6 denote barrier,
   broadcast, allreduce, allgather, reduce-scatter, and gradient sync. ABI 1.1
   adds count, weighted-sum, and update-vote phases as kinds 7 through 9.
   The count phase uses the count and root fields for parameter size and
   learning-rate bits, and sends one I64 effective count per rank. The
   weighted-sum phase carries `P + 1` binary32 contributions and the vote has
   no payload. The unused fields are zero. Gradient sync and the weighted
   step report local finiteness before sending their operands.
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

`CC=clang CXX=clang++ DIST5_ASAN_DETECT_LEAKS=1 bash scripts/test-dist5.sh`
builds the archive, audits its exports and imports, checks C++17 linkage, and
runs the native suite in optimized and ASan/UBSan/LSan builds. The suite runs
22 fork groups per build at world sizes 1, 2, and 4. It checks bitwise
collective results under opposite rank start orders, 128 KiB broadcasts
through constrained socket buffers, packed-byte reductions, gradient means,
and a coordinated nonfinite-gradient rejection followed by a successful
barrier. Negative cases cover malformed bootstrap and operands, metadata
mismatches, peer exits and stalls during metadata, payload, or result
acknowledgment, descriptor cleanup after rejected bootstrap, and the call-wide
deadline during a slow payload transfer.

`make test-dist5-mod5` is a separate development-only composition
witness. It reuses the accepted MOD5 GQA block backward fixture and its K1
provider on current `main`. Two forked ranks compute gradients from distinct,
equal-sized two-sequence minibatches. For each of three updates, the test
compares every active synchronized gradient and SGD parameter against the
serial mean of the two independently computed minibatch gradients, using
absolute tolerance `1e-7` and relative tolerance `1e-6`. On an intervening
nonfinite-gradient injection, both ranks observe rejection, destination buffers
and model parameters stay unchanged, and a later barrier and SGD step succeed.
This gate first links the existing MOD5 and DIST5 archives, then recompiles
the MOD5 provider, K1 runtime, DIST5 transport, and witness with
ASan/UBSan/LSan (`detect_leaks=1`). Both builds passed the focused local gate.
No runtime ABI was added. Rank processes exit normally so leak reports affect
their status; the parent reaps them and fails the gate on a child error.

`make test-dist5-weighted-sgd` builds ABI 1.1 and runs the MOD5 GQA composition
in optimized C11 and ASan/UBSan/LSan builds. Two ranks compare four updates
with a serial masked-token objective, checking global counts, objective,
every active gradient and parameter, and byte-identical rank replicas.
It checks nine central differences, switches which rank has no effective
tokens, and checks floating-point environment restoration across ranks.
The gate also covers recoverable malformed counts, buffers and numeric
inputs, one-rank overflow in the staged update, an exact `INT64_MAX` count,
and fatal collective-kind and learning-rate disagreements. The standalone
collective and equal-rank MOD5 gates remain available. Hosted validation and
TR3 integration for the ABI 1.1 step are pending.
