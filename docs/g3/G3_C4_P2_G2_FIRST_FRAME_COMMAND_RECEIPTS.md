# Private P2/G2 first-frame leaf command receipts

Status: source-only leaf instrumentation on a `codex/*` successor of
`945b681f`. The native first-frame contract and fixture assertions are
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
The final audit requires the exact ordered direct-command labels, safe
receipt shape, executable basenames, ordinary zero exits, same-file named
aliases, and one physical cwd. It writes `SHA256SUMS-RECEIPTS` over all
30 preceding receipts, so a later audit rejects changed raw streams or
metadata. The source pin binds the script and the receipts capture argv
bytes, but the auditor does not validate every compiler flag. The running audit
cannot hash its own final receipt; a future outer evidence seal must cover
that receipt, the manifest and gate files.

This is a leaf-only direct-command inventory. The parent prefix/carrier
predecessor commands, the toolchain checker's nested subprocesses, the
compiler resolver's version commands and the leaf static checker's two
nested predecessor static-checker subprocesses still have aggregate or
no individual evidence. The composer
source list includes the changed leaf files here but is not a complete
transitive runner-input closure. Full transitive per-command receipts,
postfailure inventory audit, compiled gate and outer lifecycle closure
remain HOLD.
