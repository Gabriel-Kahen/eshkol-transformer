# TR3 deterministic training reference

This development-only toolkit freezes the numerical reference required by section
13 of the TR3 trainer proposal at commit `3e28db6`. It does not implement a trainer,
checkpoint, evaluator, sampler, production API, or native acceptance test.

The reference uses the accepted direct M3 mathematical model and canonical seed
1729. It trains one authentic T1/D1/D2 batch with inputs `[3,197]`, targets
`[197,41]`, and two active positions for 32 updates. The fixed optimizer is CPU-f32
AdamW with f32 learning-rate bits `3d4ccccd`, beta bits
`(3f666666,3f7fbe77)`, epsilon bits `322bcc77`, zero weight decay, and constant
schedule-factor bits `3f800000`. Gradients are divided by the accumulated
mask weight before the independent PyTorch AdamW step, matching O2's
numerator/weight contract.

`reference_manifest.json` freezes only compact configuration, provenance, artifact
digests, and endpoint measurements. The generator writes the complete 33-point
loss, 14-parameter, and 28-moment trajectory to `trajectory.json` in a fresh output
directory. That roughly 1.4 MB bulk artifact and the physical corpora stay out of
Git.

Run the pinned reference with:

```sh
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  /home/gabe/.codex/worktrees/552e/eshkol-transformer/.tmp/o2-venv/bin/python \
  -m tests.tr3_reference.generate_reference --output /tmp/tr3-reference
```

Run the stdlib/schema tests with `python3 -m unittest -v
tests.tr3_reference.test_schema tests.tr3_reference.test_isolation`. Run the pinned
mathematical tests with the same environment and Python executable as the generator:
`-m unittest -v tests.tr3_reference.test_reference`.

The held-out dataset has two batches with mask weights `[2,1]` and active
transitions disjoint from training. The cursor records also derive an `A=1` case
and an `A=2` unequal-mask case whose first resumed update crosses dataset EOS.
The validator reconstructs the authentic D1 resources and D2 datasets, then
replays the saved position, EOS reset, contribution order, epochs, and end cursor.
It also requires the exact frozen shards, packing modes, batch weights, and active
training targets, so a different internally consistent corpus is not this profile.
Every summary and trajectory evaluation is bound to those authenticated batches;
masked per-token f32 losses reconstruct each exact numerator word. Final manifest
measurements are derived from the validated summary, including the unique active
argmax and exact train/held-out transition sets.
Frozen configuration, shape, counter, ordinal, version, and measurement records
use type-sensitive comparison, so JSON booleans and floats cannot stand in for
required integers and booleans cannot stand in for required floats.
The generated summary retains all three 14-parameter gradient records used to
recompute the unequal-mask tolerance result, its bit-identity flag, and the wrong
per-batch-mean control. Those records establish internal evidence consistency;
the pinned PyTorch regeneration remains the independent mathematical computation.
These are mathematical cursor/replay references. They do not prove native restore,
checkpoint equivalence, mode restoration, failure atomicity, or compiled execution.
Future native TR3 acceptance must compare its complete trajectory within the frozen
tolerances, then separately prove the proposal's runtime, evaluation, failure,
memory, and cross-process resume requirements.
