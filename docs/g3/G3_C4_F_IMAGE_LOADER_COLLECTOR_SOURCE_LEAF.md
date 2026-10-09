# F image-loader collector — source only

This test-only leaf observes the copied F compiler in a future ROOT-pinned f31
image. It does not start Docker, build Eshkol, or authorize the historical F
gate. F's original guest result remains failed; the separate offline C audit
supports native controls only.

The collector requires the immutable 33-row f31 runtime inventory (SHA
`94a7c39142c213681da20ba640996d14504d4c697dcc88c925af35577dedb53c`)
and tool inventory (SHA
`18764bbd7e634c0c01c3ca8f0337fa34f73d92b0631bffb23e25c3a1ef850cc7`)
at fixed external mount paths. It rechecks all 33 resolved ordinary runtime
files, the exact readelf and loader tools, and all four ordinary copied F
artifacts before and after observation. Readelf must report the accepted twelve
direct ELF dependencies and RUNPATH. The loader's complete observed closure,
including transitive libraries and the loader itself, must resolve to members
of the accepted runtime inventory with exact path and bytes; unknown libraries
are denied. The resulting image-libraries TSV is new test-only evidence and
must be pinned by a future immutable ROOT launcher before the F attester uses
it. The collector cannot authenticate its own Docker image or grant launch.

The first real label-corrected collector attempt failed at readelf before
loader observation. The pinned GNU readelf followed separate DWARF links while
handling `-d`, emitted diagnostics, exceeded the unchanged 64 KiB stderr
limit, and was terminated with its actual wait and partial raw streams retained.
For dynamic-section observation only, the next source route specifies
`readelf --debug-dump=no-follow-links -d /out/build-off/eshkol-run`. This
disables automatic separate-debug following; it does not suppress stderr,
weaken error handling, strip the binary, or prove DWARF correctness. The
pinned guest tool's actual behavior remains to be measured in a fresh gate.
The separate test-only attester must match this exact recorded CLI before it
can consume a future successful collector receipt.

Readelf and loader run as owned child sessions with 120-second deadlines,
2 MiB stdout and 64 KiB stderr acquisition limits, genuine `wait4` status,
unreaped leader identity and bounded descendant/writer closure. Each phase
retains argv, raw streams and status even on failure. No stdout hash or
successful verdict is inferred when closure is uncertain. Tests use owned
synthetic children for normal completion, overflow, timeout, nonzero exit,
realistic direct/transitive loader rows, unknown and mutated libraries, and
the two-phase receipt route. Python 3.10 runtime and actual f31 observation
remain unrun; source grammar is checked separately.
