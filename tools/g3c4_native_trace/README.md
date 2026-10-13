# G3/P2G2 C7g native trace prototype — UNADMITTED

This directory contains a source prototype with host tests. It is not part of
the sealed C7d bundle or an enabled G3 gate. Container calibration and live
launcher integration remain pending.

An opt-in source-only validator is now available through
`check-g3c4-proof-receipts.py ROOT --native-log EVENTS`. It validates the
native task ledger and correlates each C7f command receipt with a distinct
native `exec` by exact argv and monotonic interval, reporting extra native
images. This does not enable a live gate or replace the exact-image canary.

## Contract under test

`trace --log EVENTS [--fault-after-exec N] -- PROGRAM ARG...` creates a new
private event log (`0600`, exclusive create), starts `PROGRAM` as its own
tracee, and follows kernel ptrace fork/vfork/clone/exec/exit events through
all observed descendants. It never attaches to preexisting processes. It
returns the root tracee's exit status after every observed task exits. A
supervisor error returns `86`; `PTRACE_O_EXITKILL` kills its tracees if the
supervisor exits. The gate must treat any nonzero status or incomplete log as
HOLD and independently kill and reap its private cgroup.

The tab-separated `g3c4-native-exec-v2` log has:

- `spawn`: monotonic nanoseconds, parent PID, child PID, fork/vfork/clone,
  observed child thread-group ID and `/proc` start ticks. Missing or malformed
  child process identity fails closed. Fork/vfork must create a new group;
  clone may create a process or a member of its parent's group.
- `exec`: monotonic nanoseconds, PID, traced parent PID, `/proc` start ticks,
  executable device and inode, hex-encoded NUL-separated argv, hex-encoded
  `/proc` executable path, hex-encoded cwd, SHA-256 of raw `/proc/PID/environ`.
- `exit`: monotonic nanoseconds, PID, terminating signal (0 for normal exit),
  status (`128 + signal` for signaled exit).
- `merge`: monotonic nanoseconds, former thread TID, new process PID after a
  nonleader thread execs and Linux changes its TID to the thread-group leader.
  The former TID must be a live same-group clone, and the next event must be
  the destination PID's exec. Its start ticks must follow the former TID's
  observed identity. Version 1 logs are rejected by the receipt postflight.
- `complete`: monotonic nanoseconds, root PID, root status (zero), then
  spawn/exec/exit/merge counts. It is
  written only after all traced tasks exit and the root status is zero. A
  validator must require this as the last record and reconcile its counts,
  including `spawn + 1 = exit + merge`.

`exec` is recorded while that tracee is stopped at `PTRACE_EVENT_EXEC`, before
its new program runs. The log is capped at 32 MiB, each `/proc` blob at 4 MiB,
and concurrent tasks at 4096; exceeding a bound fails closed. The log records
argv and environment digest, not environment contents. Keep the log private:
argv can contain sensitive data.

The root tracee installs an inherited x86_64 seccomp filter before its first
exec. A legacy `clone` with `CLONE_UNTRACED` or any x32 syscall triggers a
ptrace seccomp stop and supervisor HOLD. `clone3` returns `ENOSYS` without
creating a child, so libc can retry with legacy `clone`. This changes the
tracee's syscall environment and must be part of the reviewed launch contract.
If the filter cannot be installed, the root exits nonzero and no completion
footer is written.

## Offline host canaries

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools/g3c4_native_trace/tests -v
```

The ten host canaries previously passed on the v1 prototype. They compile
from source with strict warnings and cover 32 rapid
fork/exec children, `posix_spawn`/vfork, `system()` descendants, a thread
clone, nonleader thread exec/TID reassignment, an ignored attempt at an
untraced clone, an injected supervisor failure that kills a stopped root and
subtree before either can write a marker, an existing-log collision, and exact
argv/time receipt correlation with tampered-log negatives. They run on the host
only and create no Docker container. The v2 tracer has passed source-only
syntax and postflight tests; its ptrace behavior has not been rerun.

## Remaining gate

This records successful process images and their descendants; it does not
record failed `execve`/`execveat` attempts or their errno. That remains a
separate syscall-entry/exit receipt requirement if the gate claims every
native command attempt. An own-child TRACEME/CONT canary passed in the pinned
f31 image; this does **not** prove the supervisor's fork/clone/seccomp options
or complete descendant capture under the launcher's rootless Docker profile.
It does not attribute raw stdout/stderr to each native child, record successful
`open` resolution of external resources, or prove compiled binary equivalence,
address/provider identity, resource bounds, and lifecycle in the live recipe.
An exact-image micro-canary should first run the same supervisor with fast
children and injected failure inside the actual cgroup/container profile,
validate event ordering/lineage and cgroup cleanup, then review the integration
into the C7f receipt/validator contracts. Until that calibration, native
descendant receipt completeness and gate enablement remain HOLD.

The [source-prepared one-use v2 calibration recipe](calibration/README.md)
pins the C7i source and f31 image for that micro-canary. It is launch-disabled
and unrun pending its own review and the shared heavy-slot handoff.
