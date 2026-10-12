# E3 N1 f32 synthetic external observation diagnostic

Status: **source-backed resource-proof blocker; diagnostic source only**.
`tests/e3_private/e3_n1_small_observation.py` models expected actors for one
metadata attempt with generated index bytes only. It never
opens the f32 request, Docker store, or any OCI blob. The existing public
`CAPTURE_RUN_ENABLED = False` and `REVIEWED_CAPTURE_SHA256 = None` gates remain
unchanged. Every observation and preservation state remains HOLD; a measured
candidate does not authorize a real read, manifest request, export, or N1/M1.

The frozen [index proposal](E3_N1_F32_INDEX_ACQUISITION_PROPOSAL.md) predeclares
30 elapsed seconds and 268,435,456 **current RSS** bytes, but reports no usage.
The stronger absolute-enforcement question remains blocked by the preceding
[feasibility review](E3_N1_F32_STRICT_SUPERVISOR_BLOCKER.md). This source
records whether the expected synthetic actors appear nominally inside the
original caps; it does **not** prove the complete procedure met them.

`observe_synthetic(destination)` is a test harness, not a production CLI. It
requires an absolute nofollow `/tmp/` destination, accepts only fixed synthetic
modes, and caps no larger than the frozen values.
The external witness starts `CLOCK_MONOTONIC` before exclusive destination
creation, intent sync, or actor launch. It ends after the observer has been
reaped and its bounded pipes have reached EOF and synced. That interval covers
the measured actors' setup, child capture, validation, raw-stream sync,
result seal, report and exit. Witness report validation and final result
construction/sealing occur **after** that endpoint. Thus the recorded interval
is not a completed-procedure elapsed proof, even when nominally <=30 seconds.

The **expected actor set** is observer, worker, and direct child;
the external test witness is measurement infrastructure outside that RSS set.
The observer launches only the fixed worker script, the worker calls the
existing capture harness with only the fixed synthetic child script, and the
child has no process-launch path. The accepted `valid` path has no other
process creation. This is a source-level assumption for these exact synthetic
scripts, not runtime confinement or a proof that an unexpected descendant
cannot escape observation. A changed interpreter, adversarial same-UID actor,
or future real CLI deployment invalidates the assumption. The witness's own
RSS is excluded, so the three-actor sum is not a whole-procedure RSS bound.

Linux `wait4` supplies the child's peak RSS to the worker, the worker's peak
to the observer, and the observer's peak to the external witness. Each actor
is measured **after exit**, so its late allocations are included. Each peak
is converted from Linux KiB to bytes. At every instant, aggregate current
RSS of those three **if they are the complete actor set** is at most the sum
of their per-process peaks, even when their lifetimes overlap. The result
records this as `expected_actor_peak_sum_bytes` and may mark
`nominal_expected_actor_caps_met`; neither is a whole-procedure proof. A missing
peak, failed/late reap, nonzero exit, malformed canonical report, child capture
failure, overflow, or elapsed/peak-sum excess makes that nominal field false.
Every result has `resource_proof=HOLD` and no custody/admission authority.

The witness bounds each pipe at 64 KiB, retains its prefix in exclusive files,
and kills the direct process group at timeout or overflow. The underlying
capture retains its 8-MiB stdout and 1-MiB stderr bounds. No timeout result
is a positive proof: uninterruptible work can stop late, and a failed
ancestor may leave descendant state uncertain. Intent and already-written
stream files are retained; result publication uses the same canonical,
checksummed, non-executable partial/no-replace/sync pattern as the capture
candidate. Missing or failed seal leaves partial evidence, never a receipt.

The `preservation` field is an explicit **unavailable interface**, not a
broker invocation: a separate-UID protected broker must establish a durable
HOLD intent before any real source open, retain complete raw stdout and
failure evidence under protected ancestors, authenticate a post-sync receipt,
and obtain an independently held read-only seal. The existing
[broker candidate](E3_N1_BROKER_CANDIDATE.md) models those concepts only on
synthetic data and is not deployed for this source. Same-UID checksums and
mode 0600 files remain mutable, so they cannot satisfy protected custody.
Independent and ROOT review of the exact real executable, closed process set,
resource observations, complete timing endpoint, preservation transport and
authority boundary is the next gate before either public real-read gate changes.
