# E3 N5 authentic edge and 8,193 preflight preparation

## Control-check diagnostic successor after `2a39341`

The sealed `2a39341` attempt2 completed six AOT builds and returned from the
normal-A 8,192-row control evaluation, then exited 134 before
`control-after-checks` (independent review:
`e3/n5-stage-2a39341-full-attempt2-independent-review-20261025/REVIEW.md`,
raw seal `d1fb1653...`). Its trace cannot distinguish the batch-count,
token-count and cursor checks. This test-only successor places a flushed
`before-check` and `after-check` marker around each existing assertion. The
last emitted marker can identify the first check interval that aborts, while
the assertions, authentic D2/M3 evaluation, cap/edge AOTs, resource limits and
final gate result remain unchanged. The resulting package/IR/AOT must be
rebuilt and independently reviewed. No control count, stage-2 outcome, edge
case, resource ceiling or E3 §9 admission follows from this source change.

## Diagnostic successor after the `943b745` runtime abort

The sealed `943b745` attempt completed its private builds and AOTs but its
first normal cap process aborted before the final result, with empty stdout.
That observation does not locate the failure within the seed, 8,192 control,
or 8,193 preflight. This test-only successor emits and flushes nine ordered
`E3-N5-STAGE` lines: before evaluation, after evaluation, and after checks for
each of those three phases. The final cap result still follows all original
assertions. The gate requires the exact ordered lines and result for every
mode; an abort retains the last completed marker in raw stdout. The genuine D2
corpora, model evaluator, limits, and stage-2 atomicity checks are unchanged.

This successor has source/host-test evidence only. Its changed driver and gate
require fresh normal A/B/sanitized private packages and cap/edge AOT builds;
`943b745` binaries cannot validate it. The resulting undefined import
inventories, including any flush-related import, must be measured and reviewed
against the exact new source before a compiled gate. No N5 runtime or resource
admission follows from these markers.

Status: **source/host tuple preparation only** on base `44c0418d`. The
noninstalled private candidate includes a genuine 8,192-row success control
before the 8,193-row rejection, plus one-token, empty, singleton and entry-EOS
edges with six-output and cursor checks. Its sanitizer
import inventories still require measurement and pinning, so the compiled gate
has **NOT RUN**. No N5 resource acceptance follows from this file. The distinct
single-invocation 8,192 horizon and 1,024/8,192 complete-call retention gates
retain their own evidence and scope.

`python3 -B -m tests.e3_n5.prepare --output FRESH_DIRECTORY` writes a raw V256
T1 artifact, six canonical checksummed D1 resources, and a deterministic
`preflight.json`. The record binds the exact D1 file sizes and SHA-256, manifest
trailer, D2 entry/end cursor bytes, total and remaining row counts, active
tokens, and every emitted input/target/mask row. The generator reads every D1 resource
back through the accepted D1 parser and derives rows/cursors through the
development-only D2 reference. It does not feed rows or expected metrics into
the compiled evaluator. The fixture is generated outside Git for each run.

| Case | Authentic D2 state | Required private outcome |
|---|---|---|
| `empty` | zero shards, zero rows | Current finalizer cannot publish `W=0`; capture typed stage-12 rejection and six-output/cursor atomicity. |
| `singleton` | one token, zero rows | Same typed stage-12 rejection, independently checks a present D1 shard. |
| `entry-eos` | two tokens, one row, entry cursor already at ordinal 1 | Same stage-12 rejection with exact nonzero entry cursor. |
| `one-token` | two tokens, one row, `[(3,0)→(197,0)]`, mask `10` | Genuine one-active-token success; compare six outputs against the pinned independent M3/CE/rational-metrics oracle and unchanged entry cursor against D2 reference bytes. |
| `cap-8192` | 16,385 tokens, 8,192 full-mask rows | Existing single-invocation predecessor is the successful horizon control; a new private gate must still bind its own exact fixture, six outputs and resources. |
| `cap-8193` | 16,387 tokens, 8,193 full-mask rows | Stage-2 `unsupported` before native acquire or traversal; unchanged entry cursor, all six outputs, modes and owners. No 8,193rd batch or publication. |

## Compiled-private stage-2 candidate and remaining N5 gate

The current **separate noninstalled E3 test tuple** extends the canonical
private root and bridge in `tests/e3_n5/`; it does not edit the accepted
retention tuple or installed library. It reuses the fixed seed-1729 M3 fixture,
real D2 open, frame bind and evaluator. Its driver first evaluates one active
token, then succeeds on the authentic 8,192-row fixture and checks 8,192
batches, 16,384 active tokens and cursor stability. Only then does it evaluate
8,193 rows and require the stage-2 `unsupported` table identity, idle frame,
unchanged cursor, mode-token identity, frame/model owner state and all six published outputs. This successful control rules
out a shared profile mismatch as the cause of the one-over rejection. The
stage-2 runner declares a 524,288 KiB peak runtime RSS ceiling for each
normal, repeated, poisoned and sanitized process. The closed package has exact
source, native, defined/export and normal undefined
manifests. The sanitized import inventories are deliberately absent until
measured from this exact compiled tuple; the full runner refuses to start
without them. Under `heavy.lock`, one direct
`scripts/build-e3-n5-test.sh FRESH_PATH sanitize` call can retain a rejected
builder stage with `noninstrumentation-undefined.txt` and
`sanitizer-undefined.txt` for independent review and pinning. This discovery
call is not admission. No source-only result establishes a compiled outcome.

The same test tuple now has a second boxed edge entry. It opens each D2
resource, checks the genuine total/next ordinal, evaluates one token, then
checks empty, singleton and entry-EOS for exact stage-12 `invalid-state`, idle
frame, unchanged six outputs and byte-identical cursor. Entry-EOS advances
and releases its sole batch before evaluation. The runtime emits all four
cursor bytevectors as decimal bytes so the development verifier can compare
them to the signed D2 preflight. The pinned Python 3.14.6/PyTorch 2.13.0+cpu
M3 reference computes one-token logits and CE; the rational metrics oracle
checks the frozen `40b28797/3f800000/43846900/00000000` words. The compiled
loss and perplexity are compared using the existing E3 mathematical tolerance;
weight, accuracy and counters remain exact. This is an independent oracle,
not a compiled observation.

The six D1 manifest/shard payloads total exactly 269,104 bytes: 224, 496,
504, 504, 133,680 and 133,696 bytes in the table's order. The runner checks
these sizes and declares a 524,288 KiB peak RSS ceiling for each normal,
repeated, poisoned and sanitized runtime process. No RSS value has been
measured for this tuple. N5 still needs a genuine P1 mode-bit snapshot seam,
full native-owner/FD/borrow/pin/root-region accounting and reviewed compiled
outcomes. The requirements below describe those remaining checks.

For each case, open D2 with the current driver config
`N=1,T=2,V=256,packing=#t,shuffle=#f,window=1`, save the bytevector cursor,
four I2 scalar bit patterns and two counters, and snapshot the 17 P1 mode bits,
model/gradient/RNG image and native owner probes. For `entry-eos`, call
`d2-token-dataset-next-batch`, release the batch, and require the entry cursor
to match the frozen ordinal-1 bytes. Run exactly one evaluator call inside a
bounded region. On rejection, compare the caught E1 category and operation
with `e3-error-table-ref` for its exact stage, then compare all pre-call bytes
and controls. On the 8,193 case also assert the frame diagnostic stage is 2,
the pre-acquire owner counters are unchanged and D2's next ordinal remains 0.
Close the dataset after the checks. A fresh 8,192 control uses the same driver
and checks its independent row count and mask-weight word; it is not a
substitute for the separately accepted predecessor.

The source ordering to preserve is `e3-stage-set! entry 2` →
`e3-d2-prepare!` → `e3-stage-set! entry 3` → `e3-native-acquire` in
`native/e3_private_extension.esk`. `e3-d2-prepare!` computes
`core[14]-core[15]` from the authentic D2 state and rejects `>8192` before
publishing its prepared record. The later traversal cap cannot prove this
one-over row. A gate that merely checks the Python row count or directly
invokes the provider must remain labelled structural or provider-only.

Run two fresh optimized AOT builds at stable generated-source path, compare
objects/archives/executables and exact manifests, then normal, poisoned and
ASan/UBSan/LSan runtimes with empty success stderr. Measure payload, peak RSS,
FDs, D2 borrow/batch liveness, I2 views/pins and root-region bytes against
the predeclared ceilings. Seal toolchain/source hashes, generated resources,
raw output, `/usr/bin/time` receipts and post-exit SHA-256. Route heavy builds
through `heavy.lock`; do not infer this gate from a previous source tree.

The current generator and host tests establish authentic D1 bytes, D2 rows,
signed cursor bytes, deterministic payload sizes, the one-token oracle and
source closure. Model-forward values, stage/category observations, rollback
and runtime resource ceilings remain **NOT RUN**.
