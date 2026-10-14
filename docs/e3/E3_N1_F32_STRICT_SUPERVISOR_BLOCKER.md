# E3 N1 f32 strict capture-supervisor feasibility HOLD

Status: **source-backed blocker for an absolute enforcement claim; no supervisor
admitted**. This bounded review started from the accepted one-use capture
source at `aa583f7a6e50a8c1d0dae8dddc6321e455becf28` (tree
`cc890a96d88512b06d4aa35a25c0b43da445ac7e`). The frozen
[acquisition proposal](E3_N1_F32_INDEX_ACQUISITION_PROPOSAL.md) says: “The
proposed caps are 4,096 source bytes, 30 elapsed seconds, and 268,435,456
current RSS bytes. They are predeclared limits, not measured acquisition
usage.” This review evaluates the **stronger successor request** for a hard
30-second end-to-end completion guarantee and an instantaneous 268,435,456-byte
whole-procedure RSS/process-tree ceiling. The frozen proposal does not itself
claim those guarantees or report a measurement. No real request, index,
manifest, config, layer, Docker or capture was run. The public
`CAPTURE_RUN_ENABLED` remains `False` and the small-metadata
`REVIEWED_CAPTURE_SHA256` remains `None`.

## Host and source observations

- The host has unified cgroup v2. `/proc/self/cgroup` placed this task under
  `user@1000.service/app.slice/...`; the user service and `app.slice` expose
  `memory` and `pids` controllers. The user manager reports `Delegate=no` for
  `app.slice`, `MemoryMax=infinity` and an effective ancestor limit of
  33,561,227,264 bytes. `systemd-run` is version 262. These are inspected
  capabilities, not a configured 256-MiB capture unit.
- The accepted capture source uses an in-process signal timer and Linux
  `wait4` for the direct child's peak RSS. Its documentation already marks
  external deadline, whole-procedure RSS and same-UID custody as open.
- No transient service, cgroup, Docker process or synthetic capture was
  started for this review. Read-only commands were `cat /proc/self/cgroup`,
  `cat` of the user-service and app-slice controller files,
  `systemctl --user show app.slice -p ControlGroup -p Delegate -p MemoryMax
  -p EffectiveMemoryMax -p TasksMax`, and `systemd-run --version`.

## Why the stronger absolute proof cannot be issued

| Candidate primitive | Source-backed limit against the stronger guarantee |
| --- | --- |
| cgroup v2 `memory.max` / systemd `MemoryMax=` | The [kernel cgroup v2 contract](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory-interface-files) says usage may temporarily exceed `memory.max`; `memory.current` includes the cgroup and descendants but tracks accounted memory, not an exact sum of process RSS. systemd's installed `systemd.resource-control(5)` maps `MemoryMax=` to `memory.max`. Thus a 268,435,456-byte setting cannot prove an instantaneous <=268,435,456-byte whole-procedure RSS ceiling. Lowering the setting by an unproved margin is still an approximation. |
| `wait4`, `/proc` samples, `RLIMIT_RSS` or `RLIMIT_AS` | `wait4` reports one child's peak after exit; samples miss peaks, and per-process address-space limits do not bound a process tree's aggregate RSS. They can be diagnostics or conservative rejects, not this proof. |
| systemd `RuntimeMaxSec=` and `cgroup.kill` | Installed `systemd.service(5)` says runtime termination begins after the service has been active longer than the configured limit; activation and capture preparation are outside that interval. The [kernel cgroup v2 contract](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#core-interface-files) defines `cgroup.kill` as SIGKILL delivery, not a guaranteed completion time. It says cgroup freezing may take time before `frozen=1`. |
| In-process or external timer plus SIGKILL | An external timer can record a deadline and request termination, but cannot prove that a task has stopped by that instant if it is in an uninterruptible kernel wait. The [kernel scheduler documentation](https://docs.kernel.org/scheduler/completion.html) describes a default uninterruptible wait without a timeout; the [kernel delay documentation](https://docs.kernel.org/timers/delay_sleep_functions.html) distinguishes uninterruptible sleep from waits a signal can end early. A blocked `fsync` or read therefore remains an explicit uncertainty, never a checked 30-second completion. |

The cgroup memory controller would materially improve containment, and a
separate supervisor could improve timeout detection. Neither supplies the
**absolute** guarantees evaluated in this successor. This does not establish
that a completed attempt could never satisfy the frozen proposal's predeclared
caps. It means a wrapper cannot claim those caps from these primitives alone.
No source or run gate was changed in this successor.

## Next gate

Under separate ROOT review, an external observer and post-run evidence might
establish that **one completed attempt** met the frozen caps: define the
monotonic interval from before preparation through verified process-tree
cessation, and prove it was at most 30 seconds; define the complete process
set and a sound whole-procedure **current RSS** maximum, then prove it never
exceeded 268,435,456 bytes. Missing samples, unobserved descendants, delayed
termination, or uncertain measurement must HOLD. `memory.peak` measures
cgroup-accounted memory and is **not** silently interchangeable with current
RSS. A conservative accounting bound or a revised metric needs its own proof
and ROOT review; neither may silently relax the frozen cap. The procedure
must also receive separate source, nofollow destination, partial-failure
retention, and protected-custody review before either public gate changes or
any real index read. Same-UID home files still do not establish custody.
