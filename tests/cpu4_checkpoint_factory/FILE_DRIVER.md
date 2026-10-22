# Direct FILE driver, normal/repeat only

Run only through the independently reviewed one-use supported-image launcher
under the shared heavy lock. This driver does not acquire another lock or launch
parallel compiler jobs. The wrapper verifies the pinned fe9 toolchain and invokes
`python3 -B -m tests.cpu4_checkpoint_factory.file_driver` with exact runner/compiler
paths. Production source root and per-TU roster are unchanged.

`file_runtime.esk` loads the real aggregate once, then the byte-pinned unchanged
nonregional and positive64ea fixtures, and makes the actual data-movement call.
`file_native_overlay.json` adds existing test hooks to five selected production
TUs and links the existing header-only child observer once. All original flags
and candidate ON remain. Test objects replace production object builds; no
second native owner is linked. No new numerical/reader/writer/provider function,
generation handoff, installed policy or factory implementation is added.

The deterministic rawV256 N1/T4 corpus reuses `tests.d2.test_resources` and the
exact child/control tokens and fingerprint. Resources, both genuine saved C2
files, compiler stdout/stderr/status/argv, object/IR/bitcode/ELF, native/compiler
depfiles, archive membership, nm symbols/imports and complete hashes are retained.
The actual compiler dependency inventory is discovered during the build. It is
not an allowlist or a claim that compiled closure has already passed review.
The real region fixture checks NULL baseline, distinct nested regions, negative
entry rejection and held-ticket custody. The positive fixture checks exact data
movement plus actual partial failures and cleanup/retry. Two fresh process runs
must have identical stdout and saved C2 bytes.

This driver supports normal mode only. No native or generated-IR/runtime sanitizer
instrumentation is claimed. Keep the current checksum-pinned IR sanitizer helper
and proof unchanged for a separately reviewed future SAN recipe. Direct driver
status separates program failure from post-exit custody failure; the outer
launcher independently preserves source/toolchain/recipe custody and cleanup.

Prepare-only checks:
`python3 -B -m unittest -v tests.cpu4_checkpoint_factory.test_file_driver`.
Actual normal/repeat FILE/AOT/native/region and all SAN/end-to-end gates are
NOT RUN until root schedules the one reviewed execution. Artificial coherent
FILE data movement does not prove training or persisted generation continuation.
