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
the historical failed F output, a fresh candidate/output pair, the exact
external 6803 lock, accepted f31 tool/runtime inventories, the accepted
offline native-controls-only review, and the ordinary Docker client. The
guest runs the accepted copied-F `prepare` phase and bounded loader collector
at their fixed paths. It does not produce bootstrap provenance or execute the
public G2 package. The four selected F artifacts are fresh ordinary copies;
the ten generated LLVM links remain only in the original read-only inventory.

The outer profile uses the exact f31 image ID, read-only root, no network,
4 GiB memory/swap, 128 PIDs, two CPUs, 16 MiB file limit and a 256 MiB
writable `/tmp`. Six selected binds distinguish read-only original compiler,
public checkout, original F output and bundle from fresh writable candidate
and output. Every Docker client phase has owned process-session, 16 MiB
per-stream acquisition caps, actual `wait4` status and bounded writer closure.
The container is identified by the exact created CID, label, name, image,
host profile and six binds before any cleanup; stopped inspect, owned removal
and absence are required. Failed and partial receipts remain in the consumed
attempt. Any incomplete closure is a failed result, never permission to retry.

Current tests exercise command shape, direct CLI denial, real owned child
status/overflow/timeout, and inspect mismatch. Actual Docker image,
rootless daemon, loader/library observation, source admission and physical
container closure are **unrun**. The original F guest remains failed; full
public G2, sanitizer/LSan, math, current-main union and Wave 3 remain open.
