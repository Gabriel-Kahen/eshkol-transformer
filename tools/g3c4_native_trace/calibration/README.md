# C7i v2 native tracer calibration — SOURCE PREPARED, RUNTIME HOLD

This one-use micro-canary uses the exact C7i tracer/source commit
`2f867bed5ec4f2667bc5cd4cd68f7b47dbea37af` (tree
`f4ff635c25caef7452a1bbc01cdc58cb3385e16f`) in a detached, clean
worktree. It pins the f31 image ID
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`.
The read-only source mount and individual trace, SHA, canary, and parser hashes
are checked before dispatch. `plan.json` has `launch_enabled=false`; source
preparation has not invoked ptrace, Docker, or the shared heavy slot.

The recipe carries the already prepared C7e lifecycle launcher by exact
SHA-256. That launcher owns the one-use claim, sole-slot lock, private cgroup,
guardian on parent loss, daemon-issued container ID, direct wait, both stream
EOFs, labeled cleanup, quarantine, and `SHA256SUMS-RUN`. The imported vessel
is read-only; the C7e prepared bundle and failed evidence are untouched. This
micro-canary is a distinct bundle and does not run the terminal proof runner.

Inside the pinned image, `entry.sh` builds the exact v2 tracer and canary with
Clang 21. It exercises rapid fork/exec children, `posix_spawn`, a same-group
clone, and nonleader thread exec. It then injects supervisor failure after a
child exec and attempts `CLONE_UNTRACED`; both must exit 86, lack a completion
footer, and leave no marker. `postflight.py` checks the v2 ledger and event
relationships. Its JSON status is calibration-only and unadmitted. A nonzero
guest exit or uncertain outer lifecycle remains a sealed failure or quarantine.

After independent source review, ROOT pin review, and release of the shared
heavy slot, the reviewed operator can run:

```sh
/usr/bin/python3.14 -B launcher.py --check-source
/usr/bin/python3.14 -B launcher.py --prepare-enabled /absolute/private/new-bundle
/usr/bin/python3.14 -B /absolute/private/new-bundle/launcher.py --launch
```

The destination must not exist. It receives its own prepared manifest and
one-use claim; a failed or partial run is retained. Recovery uses that same
bundle's `launcher.py --recover` path only after reviewing its quarantine
state. Never reuse a claimed bundle. The inherited launcher is pinned to this
host and worktree; any drift is a prelaunch HOLD and requires a fresh reviewed
bundle.

This calibration cannot establish failed `execve` attempt coverage, native
child stream attribution, Eshkol terminal behavior, public generation, or
full G3 acceptance.
