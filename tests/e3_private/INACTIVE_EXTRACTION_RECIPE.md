# Inactive E3 sealed-fixture emitted-inventory recipe

This source has no CLI or Docker entry point. `inactive_extraction_recipe.prepare`
requires a later
ROOT-fixed complete pin set: an actual sealed copied-F loader manifest, its
accepted ROOT and independent result reviews, the actual F-copy attestation,
compatibility TSV and readelf receipt, plus a distinct E3 full-source admission
and both accepted extraction-C source reviews. Missing or drifted material
fails before the source/compiler copy. The raw loader manifest SHA is checked
here; its complete member/topology proof belongs to that later outer review.
No current loader result SHA, E3 admission SHA, or one-shot decision is guessed.

The E3 package admission is `e3-sealed-fixture-source-admission-v1`, not the public-G2
schema. It pins clean E3 commit `e7db77c6`, tree `fe3822da`, 2,391 tracked
members and their full recorded hash. The reviewed F utility is loaded only
from `/bundle/f_artifact_attest.py` at its exact code digest. Its three exact
`scripts/common.sh` transformations point at the separately pinned 477-byte
`/bundle/public-g2-6803.lock` and match the copied-F CMake cache types. It
authenticates the historical F failed output and compiler source, stages a
fresh full E3 tracked copy under `/stage/candidate` from the read-only
`/source` checkout, copies only the reviewed F build
artifacts into `/out/build-off`, and checks the genuine copied binary/ELF and
the externally reviewed compatibility TSV. This is test-only provenance:
historical F remains FAILED and no public installed package is produced.
Preparation and both extraction phases share the same persistent `/out` mount;
`ESHKOL_BUILD_DIR=/out/build-off` therefore identifies those exact copied
artifacts. `/stage` is a fresh writable parent, so `/stage/candidate` is absent
until the authenticated tracked-source copy creates it. The historical F output
is read-only at `/f-output` and is never modified.

The module itself is a separate test-only source successor; the package source
mounted at `/source` remains the accepted extraction-C commit above.
The module must be separately pinned by the later outer wrapper. After
separate live/source/heavy/one-shot admission, that reviewed outer owner
would run `guest_command("normal")` to a fresh persistent `/out/emissions-normal`
and then `guest_command("sanitize")` to `/out/emissions-sanitize`. The existing
extraction-C wrapper retains each raw command, stream, numeric child result,
partial builder scratch, emitted IR/object/depfile/symbol rows and source input
hashes. Each phase needs its own genuine wait, stream closure, source/loader
recheck and selected-container absence before any result claim. The normal
fixture build still denies missing expected emitted manifests. Observations
cannot become expected manifests without a new independent source review.

This capsule only ran pure source-backed tests. No copied compiler, extraction,
sanitizer, Docker, native runtime or fixture test ran here. A later ROOT-pinned
wrapper must authenticate the complete loader closure, image/tool/runtime,
rootless daemon, source lease, exclusive heavy slot and output absence before
calling any preparation or extraction action.
