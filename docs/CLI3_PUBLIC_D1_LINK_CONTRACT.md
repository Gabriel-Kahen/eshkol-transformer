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
wrapper counts, or foreign-symbol inventory. Only that staged module and a
copy of the public error consumer are on the app's module search path. The app
entry imports `transformer.data`; its depfile must name the staged path and
must not name the full D1/D2 module. The builder retains the projected facade
and app depfile with the artifact and rejects unresolved public D1-writer/D2
references. These are source and build-time controls, not a compiled result.

This seam does not add `corpus validate` dispatch. Existing commands continue
through the same CLI3 entry and private dispatch source. The next gate is a
fresh exact-tree pinned-toolchain build: compare archive globals, public
exports, undefined symbols, public strings, both source closures, and app
depfile; inspect linked symbols and run the existing CLI3 command suite.
After that, implement the installed command against these seven public D1
calls and validate its exact positional arguments, output, error, and exit
contract through required CI on the installed binary.
