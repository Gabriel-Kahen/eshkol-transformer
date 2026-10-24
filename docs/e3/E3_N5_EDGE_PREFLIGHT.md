# E3 N5 authentic edge and 8,193 preflight preparation

Status: **source/host input preparation only** on base `44c0418d`. No N5
compiled-private or resource acceptance follows from this file. The distinct
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
| `one-token` | two tokens, one row, `[(3,0)→(197,0)]`, mask `10` | Genuine one-active-token success; compare six outputs and end cursor to a separately frozen independent M3/L2/L3S oracle. |
| `cap-8192` | 16,385 tokens, 8,192 full-mask rows | Existing single-invocation predecessor is the successful horizon control; a new private gate must still bind its own exact fixture, six outputs and resources. |
| `cap-8193` | 16,387 tokens, 8,193 full-mask rows | Stage-2 `unsupported` before native acquire or traversal; unchanged entry cursor, all six outputs, modes and owners. No 8,193rd batch or publication. |

## Smallest genuine compiled-private gate to add

Use a **separate noninstalled E3 test tuple** derived from the already admitted
retention package. Do not edit `retention_driver.esk`: its exact first-evaluation
mutant and source hashes depend on those bytes. Reuse its fixed seed-1729 M3
constructor, real `d2-token-dataset-open`, `e3-frame-bind!`, and the real
`e3-evaluate-into!`; do not supply logits, CE, masks or row values through a
hook. The new closed driver needs only one case selector and an entry skip
count. Its build needs its own exact source, native, defined/undefined/export
and sanitized-import manifests under `scripts/e3-private-package-policy.sh`;
the canonical private object must remain observer-free.

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

The current generator and host tests establish only authentic D1 bytes, D2
rows, signed cursor bytes and deterministic output. Model-forward values,
stage/category observations, rollback and resource ceilings remain **NOT RUN**.
