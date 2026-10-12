# E3 N1 f32 protected attempt and process-scope prerequisite

Status: **engineering contract only; HOLD**. This is a new integration boundary
between the [strict supervisor blocker](E3_N1_F32_STRICT_SUPERVISOR_BLOCKER.md),
the [synthetic observation](E3_N1_F32_SMALL_OBSERVATION.md), and the
[broker](E3_N1_BROKER_CANDIDATE.md) and
[seal-authority](E3_N1_SEAL_AUTHORITY_CONTRACT.md) candidates. It adds no
deployed service, process confinement, resource proof, protected custody, or
permission to read the real f32 index. On the published `3357d4e4` source,
`CAPTURE_RUN_ENABLED = False` and `REVIEWED_CAPTURE_SHA256 = None` remain the
real-read gates. The existing 30-second and 268,435,456-byte **current RSS**
limits are unchanged.

## Frozen boundary before implementation

An implementation must freeze one exact executable/argv/environment, source
and destination identity, finite process graph, and list of every actor that
prepares the attempt, opens a source, transports output, validates, preserves,
issues a receipt, or seals it. The list must include the capture wrapper,
reader, broker, sealer, and active supervisor/witness when they perform those
actions; a helper cannot be excluded merely because it is called a monitor.
The privileged scope-creation/launch manager is an attempt actor: its setup
time and final peak RSS count even though it begins outside the private
cgroup. Broker, seal and supervisor actions must run in finite per-attempt
workers that exit for final `wait4` measurement. If a persistent service
dispatches or performs any attempt work, it is an actor that cannot meet this
finite-actor completion contract; the design remains HOLD. Freeze the boundary
of any passive, independent measurement authority for ROOT review; exclusion
is never implicit. An unlisted actor, child, thread/process conversion,
inherited writable fd, or mutable executable makes the attempt HOLD. The
existing three-actor synthetic list is not this frozen real graph.

The acquisition procedure starts **before** destination preparation or the
durable pre-read HOLD intent. Its completion endpoint is **after** the last
protected receipt/seal write, readback, directory sync and delivery check,
all stream fds and pipes have closed with expected EOF, every declared actor
has ceased and yielded its final resource usage, and the isolated process
scope is verified empty. A visible result file or a reaped leader alone is not
completion. The final resource attestation may be assembled afterward by an
independent verifier; if that verifier performs acquisition or custody work,
it belongs in the measured scope. The exact proof-writer boundary needs ROOT
acceptance before a real attempt.

Use one externally observed `CLOCK_BOOTTIME` nanosecond interval from start to
that completion endpoint, with a single clock domain and no reset on retry.
`CLOCK_BOOTTIME` includes suspended time, unlike `CLOCK_MONOTONIC`; the current
synthetic diagnostic uses monotonic time and stops before final sealing.
The completed-attempt predicate is `end - start <= 30_000_000_000` nanoseconds.
A timer may request cancellation before that deadline but cannot establish
cessation. If any actor remains live or any seal/sync/close is unresolved at
the deadline, the attempt is HOLD even if it terminates later. A completed
attempt over 30 seconds is HOLD. This is an **after-the-fact measurement**, not
an advance hard-completion guarantee: uninterruptible kernel work can outlast
`SIGKILL` or a systemd runtime limit.

## Linux process and RSS evidence to implement and test

1. A finite privileged setup worker creates one dedicated cgroup v2 scope
   after the external start timestamp and before acquisition launch. An
   independent parent obtains that worker's final `wait4` peak; its elapsed
   work and RSS are part of the whole procedure even though it is outside
   the private cgroup. The capture UID cannot write its `cgroup.procs`, the common ancestor
   needed to migrate out, or a cgroup namespace/mount escape. Spawn the first
   measured actor directly into it, for example with `clone3` and
   `CLONE_INTO_CGROUP`; fork-then-move leaves an unmeasured interval. Test exact
   kernel, service-manager, permission and syscall availability; failure is
   HOLD. `cgroup.events:populated=0` is a necessary final empty-scope check,
   **not** a proof if migration was possible. `memory.max`, `memory.current`
   and `memory.peak` may be containment/diagnostics only, never substitutes
   for current RSS or the frozen 256-MiB cap.
2. Freeze the allowed spawn graph. A reviewed launcher installs `no_new_privs`
   and seccomp restrictions before executing a no-spawn actor; cover `fork`,
   `vfork`, `clone`, `clone3` and relevant namespace/migration routes. For an
   actor that must spawn, review that exact edge and install the descendant's
   restrictions before untrusted work. Seccomp inheritance can support this
   design, but a source assertion or a late filter cannot prove closure.
   A subreaper and pidfds may help identify/reap orphaned and direct children;
   neither alone proves there was no escaped process or pipe holder. Any
   unsupported syscall, unexpected actor, missing EOF, failed reap, or
   nonempty/ambiguous scope is HOLD.
3. Obtain Linux `wait4` `ru_maxrss` for **each** exited process, including
   late allocations through exit; a parent that measures a nested child must
   return an authenticated report before it exits and is itself measured.
   The finite setup, broker and sealer workers must exit and be measured too;
   a still-running persistent authority is not a complete peak observation.
   Convert the Linux KiB values to bytes and conservatively sum individual
   peaks. Only after the finite process set and interval are proven closed
   does `sum(all process peaks) <= 268435456` bound aggregate **current RSS**
   at every instant. `RUSAGE_CHILDREN.ru_maxrss` reports the largest child,
   not the sum of a tree. Missing usage, a reused PID without stable identity,
   an unmeasured active actor, or a sum over the cap is HOLD. `/proc` samples
   cannot fill a gap between samples.

These are candidate mechanisms, not a claim that `clone3`, seccomp, pidfd,
subreaper, cgroup permissions, or `wait4` chaining are configured or supported
for this attempt. The current login is under a non-delegated `app.slice`; no
isolated 256-MiB capture scope or distinct-UID broker has been deployed.

## Protected pre-read and final-seal order

Before **any** real source open, a deployed distinct-UID broker must durably
write and sync a protected `HOLD` intent under ancestors inaccessible for
caller write/rename/unlink, then establish a bounded stream channel. The
reviewed reader/capture path must send complete raw stdout, stderr and failure
evidence through that channel with exact byte counts and hashes; overflow,
channel break, missing bytes, unknown exit, or partial sync stays HOLD with
the protected prefix retained. A same-UID `0700` capture directory and
checksums do not supply custody. Review inherited fds, `/proc`, ACLs, mounts,
UIDs and all writable ancestors live, not just file modes.

The broker must verify the frozen source and output identities, durably publish
and read back its receipt, and deliver the receipt through a protected channel.
A distinct seal authority must independently verify the receipt and protected
snapshot, publish a no-replace seal, read it back, sync its own protected
directory, and return a protected seal reference. Neither authority may hold
the other's signing capability. The receipt, seal, token and reference must
match the same attempt and be checked before the completion timestamp. Any
uncertain publication, delivery, custody transfer or post-seal mutability is
HOLD; recovery does not manufacture authority. The existing broker and sealer
documents specify synthetic/export contracts, not this deployed small-index
transport or real-read admission.

## Next finite gate

Implement and independently review **only** a generated-byte, network-disabled
prototype of the protected attempt boundary first. Its fixed tests must prove
pre-read intent order; timed and RSS-accounted setup-worker activity; finite
broker/sealer exit and final usage; direct launch into a private scope or fail-closed lack
of support; denied spawn and migration; a detached pipe holder; complete
per-actor peak collection; `CLOCK_BOOTTIME` timing through seal and closure;
separate UID/ancestor/fd/key custody; and fault cuts at every write, sync,
readback, return and deadline. Make all uncertainties HOLD. Record exact
binary/config/UID/cgroup identities and test results, then seek independent
source/tests and ROOT review of the finite process and proof-writer boundary.
Only a separately approved real-index procedure may later attempt a completed
30-second/256-MiB proof. Neither this contract nor a green synthetic run
changes either public gate or admits §9, N1 or M1.

## Linux contract sources

- [Linux cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html): process migration permissions, recursive `populated`, `memory.max` semantics.
- [Linux `clone3`](https://man7.org/linux/man-pages/man2/clone.2.html): `CLONE_INTO_CGROUP` and direct placement.
- [Linux `seccomp`](https://man7.org/linux/man-pages/man2/seccomp.2.html): filter inheritance across fork/clone and exec.
- [Linux `pidfd_open`](https://man7.org/linux/man-pages/man2/pidfd_open.2.html) and [child subreaper](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html): stable task identity and orphan reparenting limits.
- [Linux `wait4`](https://man7.org/linux/man-pages/man2/wait4.2.html) and [`getrusage`](https://man7.org/linux/man-pages/man2/getrusage.2.html): individual child usage and `ru_maxrss` semantics.
- [Linux `clock_gettime`](https://man7.org/linux/man-pages/man2/clock_gettime.2.html): monotonic and suspend-inclusive boot clocks.
