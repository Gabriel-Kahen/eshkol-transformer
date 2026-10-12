# Private P2/G2 first-frame leaf command receipts

Status: source-only leaf instrumentation on a `codex/*` successor of
`8f82cad1`. The native first-frame contract and fixture assertions are
unchanged. No compiled, sanitizer or Docker gate has run for this successor.

`scripts/test-g3c4-p2g2-first-frame.sh` records the direct toolchain
check, static checker, owner compiles, individual `nm`/`awk`/`sort`
symbol stages, parity check, expected invalid compile and current diagnostic,
feature-off and normal/sanitize compiles and runs, oracle, repeat, stderr
predicates and parity checks. Each listed direct shell-site command has a fresh numbered directory
under the path in `commands-path.txt`, with physical cwd, NUL-separated
argv, numeric exit, and distinct raw stdout/stderr. Named outputs such as
`normal.stdout`, `invalid.stderr` and `off.symbols` hardlink to the
current raw stream. Positive compiler and symbol-stage output is retained
there even on failure; it is no longer echoed to aggregate predecessor
stderr. Failed toolchain/static stages retain a receipt root; after `tmp`
creation, failures also retain the build directory.

The invalid isolated compile must return an ordinary 1–123 rejection,
and its fresh stderr must contain the required feature-boundary diagnostic.
The opt-in first-frame mode of `common.sh` now records the toolchain
verifier and compiler resolver's child processes in `nested-path.txt`;
the static checker records its two Python children in
`static-nested-path.txt`. `nested-profile.tsv` binds whether CC/CXX came
from the lock or an environment override, the selected executable basenames,
and whether the resolver used its relative-path branch. Default `common.sh` callers
still execute their original commands directly. Each nested receipt has
physical cwd, NUL argv, numeric exit and distinct raw streams. The
wrapper replays the child streams and returns its status; a replay error
also fails the wrapper. The Eshkol version command's original merged
`2>&1` stream may have different stdout/stderr interleaving in opt-in
mode because capture is separate; its exact expected one-line value is
still checked by `verify_toolchain`.

The final audit requires the exact ordered direct-command labels, safe
receipt shape, executable basenames, ordinary zero exits, same-file named
aliases, one physical cwd, and the expected nested child labels and
multiplicity. It writes `SHA256SUMS-RECEIPTS` over all 30 preceding
direct receipts, sorted nested receipts, root pointers and override
profile, so a later audit rejects changed raw streams or metadata. The
source pin binds the script and the receipts capture argv bytes, but the
auditor does not validate every compiler flag or variable file path. The running audit
cannot hash its own final receipt; a future outer evidence seal must cover
that receipt, the manifest and gate files.

On shell failure, the EXIT trap preserves the original nonzero gate exit
and runs a prefix auditor. `postfailure-inventory.json` reports
`HOLD/INCOMPLETE`, missing child counts and hashes of successfully
inspected regular files. The active audit receipt is volatile and excluded.
`postfailure-audit.exit=0` means that snapshot ran; it
never means gate PASS. Failures before build creation retain the receipt
roots only; later failures retain the build directory too. Hard kills
still require outer lifecycle postmortem and seal.

This is a leaf shell-site inventory. The parent prefix/carrier
predecessor commands remain aggregate. `common.sh`'s source-time path
bootstrap, its receipt housekeeping, and the Python checker children's
own subprocesses are outside this nested inventory. The composer
source list includes the changed leaf files here but is not a complete
transitive runner-input closure. Full transitive per-command receipts,
compiled gate and outer lifecycle closure
remain HOLD.
