# P2/G2 terminal-close proof instrumentation

This c7e-derived branch adds an opt-in capture path. Native and Eshkol
production source files are unchanged. An external one-use launcher must create
an empty private receipt directory and export `G3C4_PROOF_RECEIPTS` to enable
per-command capture in the terminal-close runner and its eight nested runners.
The default path calls each original command directly. In capture mode each
wrapped external command keeps its argv and cwd, records numeric exit and
monotonic wall interval, hashes argv/environment/raw streams, and relays both
streams. A raw stream over 16 MiB terminates the command and fails with exit
86. Shell builtins are outside per-process capture. The `common.sh` toolchain
probes are individually wrapped. In capture mode a Python startup hook wraps
supported `subprocess.run` calls and rejects direct spawn routes and unsupported
options. The required Q0, clone, and predecessor checker child sites are
validated. Generated C++ link wrappers are wrapped. Native Eshkol children
have no exhaustive exec trace, so total per-process completeness remains on
HOLD.
`check-g3c4-proof-source.py` rejects known unwrapped external forms in the
nine runner scripts and known `common.sh` probes. Each receipt carries its
Bash file:line site, and
`check-g3c4-proof-receipts.py` requires every listed wrapper site and every
nested predecessor invocation plus known Python child sites after a full run.
This is a conservative shell audit, not an exhaustive Bash parser. The known
Eshkol linker child is captured through the generated C++ wrapper, but any
other native internal child requires an independently verified exec inventory.

The optional `--native-log EVENTS` postflight input accepts the C7g
`g3c4-native-exec-v1` log. It requires a complete, ordered root-zero process
ledger, validates spawn/exec/exit/merge lineage and footer counts, and matches
each command receipt to a distinct native `exec` with exact NUL argv and an
exec timestamp inside the receipt interval. Its JSON result reports unmatched
native exec images so they can be reviewed. This is an offline source seam;
the launcher does not yet produce a calibrated native log inside the pinned
container/cgroup. A passing postflight does not establish failed-exec attempt
coverage or per-native-child output attribution.

Set `G3C4_PROOF_EMIT=1` for emitted artifacts. Every normal and sanitizer
fixture, including feature-off and overlays, receives a separate
`--emit-object --dump-ir --emit-depfile` compile, raw object/bitcode/IR/depfile,
linked map, and relinked executable. The runner compares each relinked
executable bytewise with its direct-compile baseline. A mismatch fails closed.
This extra path changes the compiler invocation and cannot by itself prove the
direct build's emitted tail. Independent review must establish that the IR,
depfile, link map, symbol closure, and address trace support the compiled claim.

With receipt capture enabled, successful `build.*` trees are retained. The
external launcher must cap total evidence, own the private cgroup and daemon
container identity, establish both stream EOFs and a numeric direct wait, seal
the run, and quarantine any failure. No Docker run is part of this source change.
