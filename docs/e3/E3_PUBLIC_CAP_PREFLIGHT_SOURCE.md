# Public E3 8,193-row preflight source candidate

This noninstalled test caller uses the unchanged installed
`diagnostic-evaluate-fixed!` package. Its generator writes a checksummed
canonical-byte D1 resource with eight full 2,048-token shards and a final
three-token shard. Independent D2 reference equations give exactly 8,193
packed `[1,2]` rows and 16,386 true target positions. The 8,192-row horizon
fixture remains separate and unchanged.

The caller opens the real D2 dataset, snapshots its cursor, and hides the
**first** shard using the already reviewed horizon test hook. A traversal or
shard read would yield `io`; the expected result is instead the exact typed
`unsupported` error from `e3-evaluate-into!` with message
`D2 dataset exceeds E3 traversal bounds`. It requires the original cursor,
restores the shard, and repeats the denial on the same model and dataset. The
source order in `e3_d2_restore.esk` and `e3_private_extension.esk` places this
cap check at stage 2 before native acquire, P1 mode enter and traversal.
The public API exposes the error and cursor, not E3's numeric stage field.
Offline source controls reject moving native acquire before D2 prepare or
raising the one-over threshold to 8,193.

`scripts/test-e3-cap-preflight.sh` is for a resource-gated, exclusive one-use
launch after source review. It builds a fresh installed package, compares
its declared symbols and archive object, checks the caller's public-facade
closure, and compiles an optimized AOT caller with only the existing test
filesystem hook. It runs normal and arena-poisoned cases with 120-second
runtime timeouts and a 524,288 KiB RSS ceiling, preserving stdout, stderr and
time records; a strict parser rejects extra output, missing restoration and
over-limit measurements. The script and source are **prepared only**; no compiled result
or resource measurement is claimed. Run it once on reviewed exact source
after the heavy-slot resource floor is met, then seal and independently review
the raw result.

This public witness cannot inspect the four private I2 destinations, two
counters, native owner claims, in-call P1 modes, gradients, or diagnostic stage
number. A separate authentic private frame/destination gate with normal,
poison and sanitizer evidence is still required for the N5 §9 row. Neither
this source candidate nor a future public-only pass closes N5.
