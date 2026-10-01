# E3 complete-call retention candidate

This test-only successor addresses the separate repeated-complete-call horizon
in [E3 §9](../E3_EVALUATION_PROPOSAL.md). The merged single-invocation witness
counts 1,024/8,192 batches inside one call; it does not exercise this path.
This candidate requires independent source review and a supported exact-tree
normal, poisoned and ASan/UBSan/LSan gate before acceptance.

`tests/e3_private/retention_driver_root.esk` loads the unchanged canonical
private driver, then adds one boxed retention entry. The exact retention tuple
has its own policy inputs, source/native closures, export/string/undefined
inventories and archive. The canonical private, diagnostic and installed tuples
do not compile the retention observers. The runner rebuilds and checks the
canonical private package and rejects any retention symbol in its object.

For each 1,024/8,192 horizon and absent/present gradient state, the fixture
creates one genuine fixed M3 model, T1 tokenizer, packed D2 dataset, E3 frame
and four I2 destinations. It sets and records mixed modes on all 17 P1 nodes,
seeds one nonzero present gradient when selected, and snapshots every word of
all 14 parameter values and gradients, gradient metadata, the four initializer
successor words, destination bits, counters and D2 cursor. One complete call
warms lazy setup and a second sets the steady-state caller-region baseline.
The `execute` phase makes the specified number of complete evaluator calls on
those same authorities. At the
midpoint it injects one native D2 read fault, checks the typed error, original
cursor/output/mode/model/RNG bytes and idle owner counts, then retries on the
same dataset/frame as that iteration's successful call.

The caller reads exact root-arena used/total/block counts immediately around
`execute`; the private driver independently checks retained root bytes after
the failure and horizon. A test-only shim reads `region_current()->arena` used
bytes before and after each complete success, fault and retry. It requires
every successful call and retry to match the exact calibrated caller-region
delta, and records the fault delta within the fixed region budget. Per-call
temporary work stays inside a region. The native probe reads actual I2 live and
retired registries, borrow begin/view/end
events, E3 live/dead/busy frame controls, M3 graph owners, and D2 live dataset,
batch/borrow, owned-allocation and FD controls. Live controls must return to
baseline; graph owners, busy frames and active batch/borrow/FD controls must be
zero. The I2 retired-borrow count must equal completed borrow ends, and its
retained bytes must equal that count times the measured borrow-control size;
other retired I2/frame controls and the D2 owned-allocation live balance remain
fixed. FD peak growth is measured and reconciled with before/after probes.
These native controls are distinct from root-arena bytes.

`scripts/test-e3-complete-call-retention.sh` builds two fresh exact private test
tuples, compares their complete objects, archives and normalized raw Eshkol/C
depfiles, then compiles and runs each in a separate fresh optimized AOT cache.
It compares their byte-exact stdout for every required horizon and gradient
state under normal and poisoned execution. The sanitized
aggregate instruments generated Eshkol IR, the private bridge, E3/I2/D2/native
providers and the caller shim; its executable links ASan/UBSan with
`detect_leaks=1` and fails on sanitizer diagnostics. Its noninstrumentation
undefined imports stay within the normal test tuple's exact allowlist. The
runner records elapsed time and peak RSS for every process, retained RSS before
and after the measured horizon in a separate file, exact generated-corpus
payload bytes and the counter transcripts; no measured limit or full E3 acceptance
is claimed before the supported gate passes.
