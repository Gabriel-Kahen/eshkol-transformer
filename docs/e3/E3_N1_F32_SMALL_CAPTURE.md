# E3 N1 f32 one-use small-metadata capture candidate

Status: **source-only HOLD**. `tests/e3_private/e3_n1_small_capture.py` is a
separate one-use wrapper for the existing small-metadata CLI. It changes
neither that CLI's stdout/HOLD API nor its `REVIEWED_CAPTURE_SHA256 = None`
production gate. No actual f32 request was executed and no OCI/index, manifest,
config or layer byte was read for this candidate.

The public wrapper has `CAPTURE_RUN_ENABLED = False` and returns
`HOLD: capture-procedure-not-admitted` before touching its request or
destination or starting a child. The `capture()` routine below is an internal
synthetic harness, not a production acquisition API. This additional gate
stays closed even if the small-metadata CLI's capture pin changes elsewhere.

When a separate source review admits the public wrapper, its fixed argv will
take an absolute request path and a fresh absolute capture-directory path and
invoke only the Python small-metadata CLI, without a shell or caller-selected
command. Internal tests now inject tiny synthetic child programs. Before
starting a synthetic child, the harness creates the
directory with exclusive `mkdir` through nofollow ancestor fds, writes and
syncs a canonical checksummed `intent.json` with `status=HOLD`, and rejects an
occupied destination without running the child. The directory and files are
mode 0700/0600. These modes do not protect them from the same UID or a writable
ancestor.

The wrapper streams raw stdout and stderr into exclusive `stdout.bin` and
`stderr.bin`, retaining at most 8 MiB and 1 MiB respectively. An extra byte
detects overflow, kills the child process group, and marks the attempt HOLD;
the retained prefix remains available for diagnosis. A 30-second in-process
timer starts before destination creation and covers child execution,
validation and result publication. The child is killed early to reserve
finalization time. Signal delivery cannot interrupt every blocked kernel
operation, including a stuck `fsync`, so this candidate does **not** prove
a strict 30-second wall-clock bound. A separately supervised execution with
an external hard deadline remains a real-read gate. All timeout,
overflow, child-failure, malformed-output and publication failures remain
HOLD, with any files already written left in the one-use directory. A timer or
sync fault can leave only `intent.json`, partial streams, or a partial result;
their visibility never proves checked completion.

On Linux the wrapper obtains the direct child's peak RSS from `wait4`
(`ru_maxrss`, KiB) and records it in bytes. Missing `wait4`, an active process
timer, non-main-thread use, missing child rusage, or an observed peak above
268,435,456 bytes fails closed. This is a post-run observation, not memory
containment. The wrapper's own RSS and a combined current-RSS cap are not
measured, and descendants are not counted. A complete 268,435,456-byte
whole-procedure resource proof remains **HOLD** before a real read. The
reviewed small-metadata CLI has no subprocess call, but deployment must prove
the exact executable and process tree before treating direct-child measurement
as complete. If the child cannot be reaped during the bounded attempt, a
written result records `child_reaped=false` and a `child-not-reaped` failure
reason; the child state is uncertain and no success is claimed.

A zero-exit child must emit one canonical, checksummed `e3-n1-small-metadata-v1`
HOLD record for the pinned index request. The wrapper checks the request and
index digests, bounded decoded raw bytes, their digest, and the selected
manifest claim. Even this returns `outcome=captured-hold`. A nonzero exit or
malformed output produces `failed-hold`. The wrapper writes a versioned,
canonical, checksummed, non-executable `result.json.partial`, fsyncs it,
no-replace links `result.json`, and syncs the directory. It never deletes
partial files after a seal failure. Every result has `status=HOLD`, exit
status, timestamps, elapsed time, stream hashes/sizes/truncation flags, and
observed direct-child peak RSS or its absence. A visible `result.json` after a failed sync
is uncertain, never a seal or receipt.

This wrapper is not the distinct-UID [broker](E3_N1_BROKER_CANDIDATE.md),
[durable receipt](E3_N1_BROKER_CANDIDATE.md#source-only-durable-receipt-successor-v2),
or [seal authority](E3_N1_SEAL_AUTHORITY_CONTRACT.md). Its same-UID home
directory and writable ancestors cannot establish protected custody or an
immutable snapshot. No result authorizes a manifest request, layer read,
export, N1/M1 admission, or Numerical F claim. Before a real index read,
independent and ROOT review must accept the exact capture source, destination,
failure behavior, child executable and resource observation; separately
preserve the complete stdout evidence under reviewed custody; and compile the
distinct capture-contract SHA-256 pin into the small-metadata source. Until
then, that pin remains `None` and the real CLI holds before opening the
request.

Synthetic tests cover valid HOLD output, stdout/stderr overflow, timeout with partial
output, nonzero child, malformed output, occupied destination, result seal
failure, and child RSS excess. No Docker or heavy gate is involved.
