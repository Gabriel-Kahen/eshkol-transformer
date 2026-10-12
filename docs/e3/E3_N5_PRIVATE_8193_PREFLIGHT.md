# E3 §9 N5 private one-over preflight candidate

This source-only candidate targets the exact 8,193-row private preflight on
main `09f7ed52ebd30ec9d2ec2b663a1e6e90e930ebe6`. It has no compiled
verdict yet. Run `scripts/test-e3-n5-private.sh` only after the orchestrator
reviews the exact source/toolchain pins and the active hosted run is complete.

## Corpus and call contract

`tests/e3_n5/generate.py` writes canonical checksummed raw-T1/V256 D1 bytes
from the six-byte sequence `03 c5 00 ff 07 07`, repeated to 16,387 tokens.
Eight shards contain 2,048 tokens and the final shard contains three. A separate
16,384-token exact-limit case emits 8,192 rows, with one valid position in its
last row. The independent packed D2 equation
`ceil((16,387-1)/2)` gives exactly 8,193
full `[1,2]` rows and 16,386 active target positions. The generator also
creates the original six-byte three-row corpus for a later retry. Its
zero-row successors are an empty D1 manifest and a one-token D1 shard. Both
emit no D2 rows. The existing six-byte corpus also supplies an entry-EOS case
after its three rows are consumed, yielding a nonzero current cursor. The
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
Finally, the exact-limit dataset first has an injected first-read failure; the
previous six output words and counters, cursor, model, modes, graph state,
native staging and last-row observation must remain unchanged. Its clean retry
must publish exact-f32 mask weight 16,383 (`1182792704`), 16,383 active
positions and 8,192 batches. A test-only native
observer must report the final staged targets `(255 0)` and mask `(1 0)`.
No compiled verdict has been obtained for this successor.
After that success, the same model/frame/output owners evaluate each zero-row
case. The current E3 finalizer requires positive token and batch counts, so
each must raise E1 `invalid-state` at stage 12. The witness checks the exact
error envelope, unchanged six outputs, current cursor bytes, model, RNG,
modes, graph, last staged row and owner controls; native acquire increments
once while staging/forward/CE do not. The entry-EOS cursor is proved different
from the initial cursor before evaluation. These are source-wired expectations,
not compiled results.

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
Selectors 33--36 record the last successfully staged target pair and mask pair
under the same test-only guard. The exact-limit first-read fault must leave
those four values unchanged; the clean retry must end at `(255 0 1 0)`.
The mode/graph source generator permits more than four in-call notes only while
this N5 exact-limit case is armed; the ordinary four-note limit remains intact.

## Deferred compiled gate

The gate script requires an absolute canonical fresh output directory outside
the checkout, with no symlink ancestor. It rejects the unsupported-host
compatibility override. The orchestrator must supply and review these exact
launch pins before running it:

| Variable | Required identity |
| --- | --- |
| `N5_SOURCE_COMMIT`, `N5_SOURCE_TREE` | Full Git commit and tree IDs of the clean source checkout |
| `N5_LOCK_SHA256` | SHA-256 of `toolchain/eshkol.lock`; this source tree yields `8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe` |
| `N5_PROVENANCE_SHA256` | SHA-256 of the supported Eshkol build provenance TSV |
| `N5_RUNNER_SHA256` | SHA-256 of the pinned `eshkol-run` executable |
| `N5_CC_SHA256`, `N5_CXX_SHA256` | SHA-256 of the compilers named by that provenance |
| `N5_CORPUS_INVENTORY_SHA256` | SHA-256 of sorted `sha256  ./relative-path` rows for every generated corpus file; current generator yields `97a5359116173246a205b980842917889fbd8ac6b52676c92f525494569d541e` |

The script validates source and toolchain pins before builds and at exit. It
validates both generated fixture inventories before builds and at exit. The
toolchain lock/provenance are copied into the output before compilation. It
checks the source contracts, generates two
byte-equal corpora, builds two fresh normal packages and one sanitized package,
compares exact manifests, objects, archives and depfiles, compiles normal and
sanitized optimized AOT callers, then runs normal, poisoned and
ASan/UBSan/LSan cases. Each runtime is bounded at 3,600 seconds and must emit
exactly one expected success line. It requires identical stdout and empty runtime stderr
and records elapsed time/peak RSS for each mode. Its exit handler preserves
and seals partial output on failure, recording the failing stage and status.
`INVENTORY.jsonl` lists every directory, regular file, symlink and nonregular
entry; symlinks and nonregular entries reject success. `SHA256SUMS` seals the
regular files and inventory. The four seal files are created once with exclusive
creation; any pre-existing reserved file or symlink stops sealing without
altering it. `SEAL-COLLISION.json` records the collision when it can be created
exclusively; it is diagnostic, not a valid seal. A bounded post-seal strict
`sha256sum -c` plus complete inventory reconciliation checks the seal without
writing into the output tree; failure prevents gate success. `exit.status`
records the pre-verification gate verdict; the shell exit is authoritative if
post-seal verification fails. The sealed output needs independent review before
any acceptance claim. The accepted
8,192 single-invocation horizon and five-layout final-output evidence remain
separate; neither proves this N5 case.
