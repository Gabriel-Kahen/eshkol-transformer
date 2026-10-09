# Inactive copied-F loader outer source

This test-only launcher prepares the exact f31 image-loader observation. It is
**inactive**: direct module execution exits before any Docker command. A later
immutable ROOT wrapper must pin the exact canonical external decision SHA and
call `run_once`; the SHA argument is not self-authenticating. Neither a
caller-selected decision file nor this source can grant launch. Keeping the
pin in the external wrapper avoids a circular source-HEAD/decision hash.
Its input basis is the accepted collector `fb752684cfd65e20fbc1b8526709fabbf898ea91`
(tree `5ce3532ad0e1384be129408aa48afa18a19c1827`), not the earlier
conditional compiled-gate design's historical source checkout.

The decision must bind a clean exact compiler checkout, a clean public source
checkout authenticated by its independent and ROOT source-admission reviews,
the historical failed F output, a fresh stage/output pair, the exact
external 6803 lock, accepted f31 tool/runtime inventories, the accepted
offline native-controls-only review, and the ordinary Docker client. The
guest binds the empty stage root at `/stage`; the opt-in, exact
`--outer-stage-child` attester route creates previously absent
`/stage/candidate`. The attester's original `/candidate` default remains
unchanged, and arbitrary destination paths still fail. The guest runs the
accepted copied-F `prepare` phase and bounded loader collector at their fixed
paths. It does not produce bootstrap provenance or execute the
public G2 package. The four selected F artifacts are fresh ordinary copies;
the ten generated LLVM links remain only in the original read-only inventory.
The single canonical source-admission JSON names exactly the independent and
ROOT seal files under `/bundle/reviews/`. In the guest those paths are real;
the outer host verifier maps only those two logical paths into its selected
ordinary bundle directory and checks the same file bytes and hashes. The
default attester route and admission schema remain unchanged.

The outer profile uses the exact f31 image ID, read-only root, no network,
4 GiB memory/swap, 128 PIDs, two CPUs, 128 MiB file limit and a 256 MiB
writable `/tmp`. Six selected binds distinguish read-only original compiler,
public checkout, original F output and bundle from fresh writable candidate
and output. Every Docker client phase has owned process-session, 16 MiB
per-stream acquisition caps, actual `wait4` status and bounded writer closure.
The 128 MiB `fsize` limit covers the measured largest preserved F file,
`libeshkol-static.a` at 85,874,424 bytes; the historical 16 MiB profile
would fail during that copy. This is a source-level correction, not a run.
An optional ROOT-supplied monitor runs inside each owned client capture loop,
including the guest attach loop, roughly once per 50 ms drain. Its exception
denies the phase and terminates only the authenticated owned client session;
the outer `finally` still inspects, kills if needed, genuinely waits for,
removes and checks absence of the selected CID without invoking the failed
monitor again. The future wrapper must measure its selected stage/output/control
aggregate and emergency disk floor through this hook. Sampling may overshoot
the threshold between checks, so it is not a hard disk quota.
The monitor is checked immediately before Docker `create`, then suspended
through that one bounded client receipt. Monitoring resumes only after its
genuine CID has been parsed and retained for owned cleanup. If `create` itself
fails without a CID, an unmonitored exact-name census may recover a single
candidate only after full CID, label, image, effective profile and mount
authentication. That CID then enters the same owned cleanup. An absent name,
multiple candidates or foreign profile remains a quarantined failed attempt;
one absent census does not prove physical closure.
The container is identified by the exact created CID, label, name, image,
effective entrypoint/command, user, environment, workdir, resource limits,
host mount options and six bind rows before any cleanup; stopped inspect,
an actual `docker wait` decimal result matching the stopped `ExitCode`, owned
removal and absence are required. Stopped state must be exited, not running,
PID zero and not OOM-killed; a genuine never-started state is cleanup-only.
Failed and partial receipts remain in the consumed
attempt. Any incomplete closure is a failed result, never permission to retry.

Current tests exercise command shape, direct CLI denial, real owned child
status/overflow/timeout/monitor denial, inspect mismatch, monitor-failed
Docker closure, failed-create recovery/absent/foreign denial, and matching two-surface review
bytes with wrong-path and drift denials. Actual Docker image,
rootless daemon, loader/library observation, source admission and physical
container closure are **unrun**. The original F guest remains failed; full
public G2, sanitizer/LSan, math, current-main union and Wave 3 remain open.

The first actual ROOT-owned collector attempt created the selected container
but stopped before its guest. The pinned image's actual `Config.Labels` contains
`org.opencontainers.image.version=22.04`; the created container correctly
inherited it and added `eshkol.gate`. The original sole-label inspect check
rejected that genuine union. The source successor requires the exact image
label dictionary plus the one selected gate label and rejects any image that
already reserves the gate key. The exact first image and created-inspect raw
receipts are pinned by SHA-256 in the test closure; owned tests deny missing,
extra, malformed and conflicting labels. The original attempt remains failed,
and this source change grants no retry or loader/native/public acceptance.
