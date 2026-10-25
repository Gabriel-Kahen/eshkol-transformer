# E3 physical-capacity denominator mutant source

This is a noninstalled §9 M2 source candidate on exact T2 source `44c0418`.
The authentic packed three-batch D2 witness has masks `11,11,10`. The new
`capacity-denominator` numerical tuple keeps the final `10` row, actual M3
roles and L2 CE words. For that final row only, the test-guarded L3S provider
reports physical weight 2 and computes mean from the unchanged masked CE
numerator. The test-guarded E3 metrics provider reports active count 2 while
retaining the real masked correct count. Earlier batches and every canonical,
diagnostic, installed and public package use the unmodified source path.

The private gate requires two independent normal and mutant builds, package
closure checks, normal/repeat/poison/sanitizer executions, and complete real
model traversal. The frozen normal transcript must pass first. The mutant
verifier checks unchanged roles, rows, logits, CE, correct count and first two
batches, then independently derives the final reduction, serial state,
publication and counters. Its first divergence must be final-row
`reduction_f32[1]` (weight); the rejected mutant must publish no transcript.
The runtime's separate capacity case expects `(tokens,batches)=(6,3)` and
retains cursor-byte equality. The accepted padded-position mutant differs:
it includes the padded CE in the numerator, so it does not prove this row.

Status: source and host tests pass; supported compiled-private execution and
independent review are **NOT RUN**. The prior padded-position and sum-order
receipts are predecessor evidence only. The full E3 §9 matrix remains open.
