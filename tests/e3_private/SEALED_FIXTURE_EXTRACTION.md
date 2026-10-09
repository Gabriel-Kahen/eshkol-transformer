# Sealed-fixture emitted inventory extraction (source only)

`scripts/extract-e3-sealed-fixture-inventories.sh FRESH_DIR normal|sanitize`
is a nonpublishing observation route for the exact noninstalled fixture tuple.
It uses the existing E1B toolchain provenance check and the selected compiler,
private Eshkol generator, native compile flags, and relocatable link. Its
`E3_SEALED_FIXTURE_EXTRACT=1` switch is rejected for every other tuple. The
normal `build-e3-sealed-fixture-test.sh` still rejects absent emitted manifests
before compiling. The extraction switch never writes that ordinary `.o`, its
`.evidence` directory, an accepted archive, or the missing manifest files.

The fresh extraction keeps the complete builder stdout, stderr (including the
shell-expanded command/previous-status trace), numeric exit, invocation argv,
and selected input hashes before and after emission. The builder retains raw
private IR, depfile, generated P1/I2/M3/D2 source, compiled native objects and
depfiles, normalized source closures, raw and localized combined objects,
actual defined/undefined symbols, readelf local visibility, strings, link map,
and a clearly marked **probe-only** archive/member listing. Sanitizer mode also
retains rewritten private IR, rewrite provenance, and actual instrumentation
imports. A failed extraction preserves its stage and raw streams. These bytes
are inputs to independent review; their presence is not package acceptance.
The wrapper creates a private `tmp` inside that same persistent stage and
passes it as `TMPDIR` to the builder. On failure, partially emitted IR and
objects stay there with the raw streams and exit. On success, the builder must
copy its selected emissions and remove its scratch; the wrapper requires
`tmp` empty before publishing the extraction bundle. The guest's ephemeral
`/tmp` is never relied on for failure evidence.

Actual normal and sanitizer emissions, exact source/symbol/object/ABI manifest
derivation, two fresh reproducible builds, private-link denial, compiled fixture
execution, stale-handle behavior, and sanitizer runtime remain open. A later
separate immutable admission must pin the supported F compiler checkout, tool
provenance, image/runtime and heavy slot before invoking this route. This source
candidate has not invoked a compiler or Docker.
The staged source's tracked `toolchain/eshkol.lock` still pins `fe9dfd52…`,
whereas the reviewed native-F diagnostic compiler is `6803dbe2…`. The
accepted copied-F attestation/staged-common route uses a separately pinned
external `public-g2-6803.lock` and three exact source transformations; this
extractor does not create or bypass that route. A fresh E3 full-source
admission, the genuine copied-F loader result, compatible provenance TSV and
separate heavy-slot decision are prerequisites to running it under F.
