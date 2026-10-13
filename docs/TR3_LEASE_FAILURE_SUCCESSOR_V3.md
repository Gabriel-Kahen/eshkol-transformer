# TR3 lease failure successor gate v3

`scripts/test-tr3-lease-failures-successor.sh` is a separate, fail-closed
successor to the historical failure gate. The historical script, manifest, and
sealed evidence remain unchanged. The v3 manifest pins the exact 41cbf6c3
source baseline and tree, the current 23-slot successor contract, failure
fixture, production-OFF de0b249 runner/archive, provenance, build profile,
container, and exact O2 compiler warning bytes.

The default invocation compiles the same failure witness at O0 and O2 and
enumerates every `all`, promotion, and target allocation prefix through its
first authenticated `END`. It then checks the handler failure and post-staging
success cuts in fresh processes. Each mode verifies its complete marker and
case-status sequence, exit statuses, check total, source and artifact hashes.
The suite accepts only when both modes pass and their case-status files match;
it seals the combined evidence with `SHA256SUMS`. Probe and diagnostic modes
are not accepted. The source checkout must be a clean descendant of the pinned
baseline with no production or native-witness path difference.

`--source-check` performs only the source preflight. It cannot establish
compiled behavior. This source candidate has not run the supported Docker
O0/O2 suite; compiled admission and independent review remain pending.
