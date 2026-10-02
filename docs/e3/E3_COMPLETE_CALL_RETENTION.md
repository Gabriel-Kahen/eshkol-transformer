# E3 complete-call retention candidate

This test-only successor addresses the separate repeated-complete-call horizon
in [E3 §9](../E3_EVALUATION_PROPOSAL.md). The merged single-invocation witness
counts 1,024/8,192 batches inside one call; it does not exercise this path.
The final closure-order source repair at `4771e408` has independent source-and-
test approval. A supported exact-tree normal, poisoned and ASan/UBSan/LSan
gate is still required before acceptance.

The composed `c78682ae` fe9/f31 gate passed canonical packaging but stopped
at the retention test tuple's exact undefined-symbol inventory, before any
retention runtime. The retained generated Eshkol `private.o` has five
actual-only imports and no expected-only imports:
`eshkol_builtin_arena_used`, `eshkol_display_value`,
`eshkol_runtime_current_output_fp`, `eshkol_tail_transfer_slot`, and `fputc`.
The first is emitted for the retention-only `__arena-used` reading; the
display/output imports come from its test failure reporting, and the tail
transfer is compiler-emitted for the test source. The four `eshkol_*` symbols
are defined by both pinned fe9 runtime archives; `fputc` is a libc import.
The scoped repair admits these five only in the retention tuple and checks
their actual providers before the long build. Canonical, diagnostic, installed,
and numerical manifests are unchanged. The 110-file failed evidence is sealed
at `/home/gabe/e3-complete-call-retention-c78682a-fe9-f31-20261001-prepared-e/SHA256SUMS-FAILED`
(SHA-256 `569b8e2e31a3cda148c8520d77dff8ca6d4b9ce012aeea96b5ad95efe1a63f53`).
This is inventory evidence, not a compiled retention result; a fresh pinned
run of the repaired tuple remains required.

The first supported attempt at `d724c537` stopped in its canonical predecessor
package before compiling the retention tuple: fe9 emitted exactly two
actual-only undefined runtime imports, `eshkol_format_float32_bits` and
`eshkol_runtime_fatal`. The failed 13-file evidence is sealed at
`/home/gabe/e3-complete-call-retention-d724c53-fe9-f31-20261001-prepared-a/SHA256SUMS-FAILED`.
Both symbols are defined in the pinned runtime archives and already admitted by
the installed E3 diagnostic package. This repair adds only those observed
imports to the exact canonical and retention test manifests. A separate
diagnostic-private fe9 object inventory found the same exact two actual-only
imports and no expected-only difference; its 69-file evidence is sealed at
`/home/gabe/e3-diagnostic-private-inventory-d724c53-fe9-f31-20261001-probe-a/SHA256SUMS-INVENTORY`.
The diagnostic-private manifest adds that independently observed pair, while
the installed diagnostic manifest and production sources remain unchanged. No
retention runtime or sanitizer result follows from that failed attempt; a fresh
pinned gate must compile and execute the repaired tree.

The exact `0c358d5` supported rerun passed canonical packaging, then stopped
before retention runtime at the strict native-closure comparison. Its emitted
56 dependencies exactly match the expected set: only `native/d2_native.h`
appears earlier, because the retention-only observer includes it before
`m3_model.c`. The test manifest now preserves that measured first-seen order;
the production manifest and native sources remain unchanged. All 103 failed
evidence files, including the raw native depfiles, are sealed at
`/home/gabe/e3-complete-call-retention-0c358d5-fe9-f31-20261001-prepared-b/SHA256SUMS-FAILED`
(`579aa559c2651a121bb146702f833ce6fa30eedda66c8b513b4d116e9627bdf4`).
No dependency or check was removed, and compiled retention/sanitizer acceptance
still requires a fresh exact-head gate.

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
