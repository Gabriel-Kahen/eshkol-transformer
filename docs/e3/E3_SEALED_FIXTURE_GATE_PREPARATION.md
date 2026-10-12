# Sealed-M3 fixture: bounded compiled gate preparation

This is a preparation recipe, not a Section 9 acceptance. The admitted source ancestor is
`6f3b1e8140ccc36588b7403c7ed7af0bdc33b276`, tree
`a2ee99faf964d1b362de6763e19d4be7bc0dd563`, with root source-review
seal `fb546e7ffbbdbd214a867d2668f43118d6559f4d929d55293820714e08ede0b9`.
The supported runtime is `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`;
the supported image ID is
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`.
Pin the exact candidate commit/tree after this source preparation is reviewed;
do not substitute the ancestor pin for the new tree. The accepted f31/fe9
runner/provenance SHA-256 values are respectively
`7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa`
and `4b2e87315b91dd27f6453862a4c4ebcd9fa1f577619d3bfa6ceca92964a9824c`.
The consumed J source capture on `fb31fac32530e0ff9b00ed3458e9080eaba9643a`, tree
`5ca77e78a6f798f677a0b23a14e763d957f830bd`, proved the 28-row
compiled source closure only. Its ROOT result is
`e3-section9-j-orchestrator-root-result-fb31-20261012-a/SHA256SUMS-ROOT`
(`b09b7992...`); the ROOT-reviewed run manifest is
`e3-section9-gate-prep-f31-fe9-run-fb31-20261012-j.claim/attempt/RUN-SHA256SUMS`
(`a5240c3e...`). J is consumed and must not be rerun without a material change.

## Acquisition order

1. Use a clean immutable source checkout and the pinned network-none Ubuntu
   22.04/LLVM 21 image. Mount source and `.deps` read-only; mount one fresh
   evidence directory writable. Verify source commit/tree, runtime commit,
   runner/provenance hashes and image ID before and after every container call.
   Retain command, environment, exit status, raw stdout/stderr, peak RSS and
   checksums. Never reuse an output directory or delete a rejected build.
2. The consumed J invocation ran `scripts/capture-e3-sealed-fixture-source.sh
   /out/source-once`. It staged the same four generated sources and ordered
   include roots as `build-e1b-consumer.sh`, then makes exactly one
   `--shared-lib --dump-ir --emit-depfile` compiler invocation. Preserve
   `private.d`, generated sources, `private.ll`, generation records and
   compiler streams, including on failure. This is a compile-only depfile:
   an executable-mode depfile is not assumed. It is not package admission.
3. J's `observed-source-closure.txt` matches bytewise with
   `tests/e3_private/sealed_fixture_package_source_closure.txt`. The checked-in
   28 rows; the J depfile is the compiled source observation. A future difference
   is a source gate failure: retain the raw
   depfile and diagnose actual loaded paths before editing any expected list.
4. A separate one-use preparation may compile J's retained `private.ll` without
   re-invoking Eshkol. The admitted entrypoint is a sealed ROOT launcher that
   fixes `J_RUN`, `J_ROOT_REVIEW`, `CLAIM_ROOT`, output, source/tree and image
   pins, then invokes `scripts/capture-e3-sealed-fixture-ir-object.sh
   J_RUN J_ROOT_REVIEW CLAIM_ROOT FRESH_OUTPUT` in the supported image after
   ROOT prelaunch admission. The generic script alone does not enforce one-use
   across caller-selected claim roots. It verifies the ROOT seal, ROOT-reviewed run manifest,
   every J file, source/runtime pins, runner/provenance hashes and active
   toolchain before compiling, then rechecks J inputs afterward. It uses the
   E1B `cc -c -x ir` command to emit an **IR-derived, pre-rename** `private.o`,
   and captures raw `nm`/`readelf` records for that object and J's shared
   library. It records compiler argv/version/status/streams and a fresh output
   `SHA256SUMS`, retaining failures. The launcher's `CLAIM_ROOT` must be a fixed,
   writable persistent
   directory outside the read-only J capture; the script atomically reserves
   `e3-section9-j-ir-object-fb31-fe9` there immediately before LLVM and never
   removes it, including on compiler failure. A fresh output name does not
   authorize a second attempt. This preliminary object is not E1B's
   localized `combined.o`: it has no bridge or native providers, and its
   symbols cannot be promoted to final package manifests.
5. Obtain native and final object inventories from a separate **retained**
   exact-tuple `build-e1b-consumer.sh` attempt after the first source result.
   The wrapper `build-e3-sealed-fixture-test.sh` currently stops on absent
   emitted manifests; it cannot be called an inventory run. The builder
   retains `e1b_tmp` on rejection. Capture each rejected temp directory and
   raw stream before changing one provisional manifest in an isolated gate
   checkout. Do not copy a guessed value into the accepted repository. Once
   the builder emits complete `OUTPUT.o.evidence`, run
   `scripts/extract-e3-sealed-fixture-inventory.py objects
   OUTPUT.o.evidence /out/observed-object` to copy the raw depfile, source and
   native closures, native depfiles/object list, global definitions, imports,
   public strings/exports, `nm`, `readelf` and link map into a fresh directory.
   This `OUTPUT.o.evidence` from the localized E1B object is the final symbol
   authority. Derive candidate checked-in manifests only from these observed files;
   independently review exact symbols, native translation-unit flags,
   localization, archive members and canonical observer absence before
   admission. Record every provisional attempt and its rejection.

## Execution gates after inventory review

Use two fresh stable generated-source gates. Compare their raw and normalized
depfiles, objects, archives, executable and stdout. Run normal, repeat and
poisoned runtime cases; assert the expected typed stale model/head-handle
errors, live-workspace rejection, unchanged original, reconstruction and
second retirement. Successful stderr must be empty. Keep all failure stderr
and rejected stages. Run the same source with `ET_G3R_CANDIDATE_RETIRE_PRIVATE`
off in a separate exact tuple and compare the affected public/trusted object
bytes and symbol inventories to the accepted default-off baseline; any drift
blocks admission. Do not infer feature-off parity from the feature-on object.

The builder presently excludes `sealed-fixture` from its admitted ASan IR
rewrite tuple list. Sanitizer remains a **held** gate until a separately
reviewed exact source/IR/import policy admits it. Then run ASan/UBSan/LSan
with leak detection and compare the actual source, object and runtime records
to the normal gate. A rejected sanitizer build is preserved evidence, not a
sanitizer pass. No row of the broader §9 numerical/failure matrix is closed
by this fixture source gate alone.
