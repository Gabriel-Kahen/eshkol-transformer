# E3 §9 N5 private one-over preflight candidate

This source-only candidate targets the exact 8,193-row private preflight on
main `09f7ed52ebd30ec9d2ec2b663a1e6e90e930ebe6`. It has no compiled
verdict yet. Run `scripts/test-e3-n5-private.sh` only after the orchestrator
reviews the exact source/toolchain pins and the active hosted run is complete.

## Corpus and call contract

`tests/e3_n5/generate.py` writes canonical checksummed raw-T1/V256 D1 bytes
from the six-byte sequence `03 c5 00 ff 07 07`, repeated to 16,387 tokens.
Eight shards contain 2,048 tokens and the final shard contains three. The
independent packed D2 equation `ceil((16,387-1)/2)` gives exactly 8,193
full `[1,2]` rows and 16,386 active target positions. The generator also
creates the original six-byte three-row corpus for a later retry. Its
versioned manifest records tokenizer identity, every file length/SHA-256,
D2 configuration, starting cursor, token count and expected row count.
It also records every expected input, target and mask row from direct token
slices, and compares each row against the independent D2 reference.
Generated resources stay outside Git; two materializations must be byte equal.

The compiled call opens genuine D2 using the private driver configuration and
uses the existing seed-1729 M3 model, four I2 destinations and bound E3 frame.
It snapshots D2 cursor bytes, all 17 P1 modes, four scalar bits, two counters,
model value/gradient image, RNG, graph events and owner controls. The real
`e3-evaluate-into!` must raise E1 `unsupported` for operation
`e3-evaluate-into!`, message `evaluation dataset preparation failed` and
source-code 2. The specified byte and owner invariants must match afterward.
The same model/frame then
evaluates the three-row corpus and must publish the six accepted words
`(1085366897 1084227584 1132430055 0 5 3)` with cursor restored.

## Package and observer manifest

The exact noninstalled package tuple is
`tests/e3_mode_graph/{driver_root.esk,bridge.c,package_private_renames.txt,package_public_exports.txt}`.
The existing source, native-source, defined-symbol, public-string, import,
native-object and archive-member manifests under
`tests/e3_mode_graph/package_*.txt` remain the build comparisons. No root,
export or source-closure member is added. The N5 scenario is selected only by
the boxed `(vector 'n5-8193)` scenario; the existing mode/graph scenario
remains available. The observer is built only under `ET_E3_RETENTION_TESTING`
in this noninstalled tuple. Its existing probe symbol now admits selectors:

| Selector | Entry counter |
| --- | --- |
| 29 | native frame acquire |
| 30 | native batch staging |
| 31 | native forward role (provider routing) |
| 32 | native cross entropy |

All four must remain byte-identical across the one-over failure. The three-row
retry is their positive control: deltas `1,3,63,3`. The stage-2 error and
pinned source order in `internal/e3/lib/e3_d2_restore.esk` and
`native/e3_private_extension.esk` place this rejection before native acquire,
P1 mode enter, D2 traversal and provider routing. The counter is test-only;
the production ABI and private evaluator interface do not change.

## Deferred compiled gate

The gate script requires `N5_SOURCE_COMMIT` equal to the clean checkout HEAD
and a fresh output directory. It checks the source contracts, generates two
byte-equal corpora, builds two fresh normal packages and one sanitized package,
compares exact manifests, objects, archives and depfiles, compiles normal and
sanitized optimized AOT callers, then runs normal, poisoned and
ASan/UBSan/LSan cases. It requires identical stdout and empty runtime stderr
and records elapsed time/peak RSS for each mode. The generated output must be
sealed with SHA-256 and reviewed before any acceptance claim. The accepted
8,192 single-invocation horizon and five-layout final-output evidence remain
separate; neither proves this N5 case.
