# CLI3 public D1 link seam

Status: source-only candidate. No compiled CLI3 archive, app, D1 validator,
or hosted CI result has been produced from this exact tree.

The production aggregate already includes `native/d1_e1b_package_bridge.c`
once through its TR3 predecessor. This seam adds no second D1 archive or
bridge. The production CLI3 export allowlist now retains exactly the existing
CLI3 dispatch pair plus D1 `token-corpus-validate` and its six summary
accessors. `token-corpus-write!` remains local. No D2 public API is exported.
The expected global-defined manifest is the six E1B error symbols plus those
nine allowed exports. The private Eshkol/native source closures, public-string
inventory, and undefined-symbol allowlist are unchanged; the compiled gate
must confirm that they still match the exact manifests.

The app build stages a read-only `transformer.data` projection from the marked
production D1 facade component. It removes both writer definitions, retains
the validator and six accessors byte-for-byte, and rejects changed markers,
wrapper names, or foreign-symbol inventory. The app's module search path
contains that projection, the public error consumer, and a copy of the shared
`transformer.cli3_corpus_validate` handler. The app depfile must name all three
staged modules and must not name the full D1/D2 module. The builder retains the
projected facade, handler, and app depfile with the artifact and rejects
unresolved public D1-writer/D2 references. These are source and build-time
controls, not a compiled result.

The pinned Eshkol runner writes `--emit-depfile` only for object or shared-library
output. The builder therefore checks the app import graph with an object-mode
compile, then compiles the executable from the same source and import settings.

The installed entry intercepts only the `corpus validate` argument prefix
before its existing native dispatch and fallback guard. The standalone command
and installed entry call the same Eshkol handler, which uses the public D1
validator, six summary accessors, and public error accessors. All other
arguments continue through the existing CLI3 dispatch. The required CLI3
production-generate gate now runs installed validator positives and negatives
against a corpus built by that binary before its existing command suite.
The next gate is a fresh exact-tree pinned-toolchain build: compare archive
globals, public exports, undefined symbols, public strings, both source
closures, and app depfile; inspect linked symbols and run the installed
validator plus existing CLI3 command suite. Hosted exact-head CI remains
pending after that gate.
