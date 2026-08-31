# E1B separately compiled consumer boundary

E1B lets an installed public Eshkol package raise E1 errors without placing
`transformer.error_internal` or `transformer.error_core` in its source closure.
The public package is a source stub plus an already-localized relocatable object.
Application source is compiled only after that object has been completed.

This is a private package-building ABI, not an A0 API, K1/I1 ABI, or persistent
format. It is fixed to canonical `tsotchke/eshkol` commit
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`, x86-64 LP64 SysV, and the supported
Ubuntu 22.04 / LLVM-Clang 21 lane.

## Raise-only contract

The build-only Eshkol seam is exactly:

```scheme
(et-e1b-private-raise category operation message details cause)
```

It is fixed-arity, marked `:no-return`, and only calls
`transformer-error-raise`. It has no opcode, list dispatcher, arbitrary callback,
constructor result, wrap-foreign operation, or representation accessor. E1 alone
owns validation, category admission, canonical details, cause copying, identity,
metadata, bootstrap failures, and the six accessor results.

A trusted package root calls this seam from narrow package-specific functions. A
package C wrapper transports the package function's fixed arguments and exports only
reviewed names matching `et_e1b_public_*_v1`. The generic seam, its generated C-ABI
thunk, E1 constructors, `e1-internal-dispatch`, initialization helpers, and generated
companions are resolved inside one relocatable object and changed to `STB_LOCAL`.
They cannot satisfy an undefined reference from an application linked afterward.
`STV_HIDDEN` without local binding is insufficient and is not used as the security
boundary.

The six unchanged unary A0 accessors use one-input/one-output one-element vector
boxes and are the only base bridge exports. Package-specific public wrappers use the
same box transport. Each payload is the pinned layout:

```text
offset 0: signed vector length, exactly 1
offset 8: one 16-byte, 8-byte-aligned eshkol_tagged_value_t
box size: 24 bytes
```

Native glue checks vector subtype and length, copies a tagged value, initializes the
single private E1 registry, and crosses the generated call boundary. It does not
interpret categories, operations, messages, details, causes, or error identities.

## Package construction

`scripts/build-e1b-consumer.sh` takes a reviewed build-only Eshkol package root, a
reviewed package C bridge, a compiler-symbol rename list, an exact package export
allowlist, an output object, and optional public include directories. The export
allowlist and repository-owned runtime-undefined-symbol manifest are canonical
C-sorted, unique, one-symbol-per-line build inputs, not installed data formats.
Package export names must match `et_e1b_public_*_v1`. After localization the builder
compares both the global definitions and sorted `nm -u` output byte-for-byte with
their reviewed allowlists before atomically publishing the object and textual
depfile, link-map, symbol-table, allowlist-provenance, undefined-symbol, export, and
strings evidence. A differently named unresolved function, data, TLS, or weak symbol
is therefore rejected before an application can supply it at the later link.

The undefined-symbol policy is not caller-selectable. The base fixture uses
`native/e1b_undefined_symbols.txt`; the checked-in X1 trusted root is selected by
its exact repository root, bridge, rename, export, and include identities and uses
the independently reviewed X1-specific `native/x1_undefined_symbols.txt`. P1's
never-installed trusted root is selected only by its exact repository root, bridge,
rename, and export identities with no optional include directories, and uses
`native/p1_package_undefined_symbols.txt`. D1's
normal and test-fault policies accept only its exact repository root, bridge,
rename, export, and sole `src` include identities; copied roots, partial tuples,
include shadowing, and policy overrides on generic packages are rejected before a
temporary or published artifact is created. The test-fault policy is derived from
an exact never-installed test root used by `scripts/build-d1.sh test-faults`, is
forbidden from targeting the canonical production artifact directory, and uses its
separate reviewed undefined manifest.
The compiler subprocess clears caller `ESHKOL_PATH` and pins `ESHKOL_LIB_DIR` to the
repository public library, so environment module paths cannot replace the classified
source closure. No command-line manifest argument exists. The
trusted native consumer bridge is compiled with `-fstack-protector-all`, so both
supported Clang 21.1.8 and the explicitly unsupported Clang 22.1.6 compatibility
lane retain `__stack_chk_fail` in the base fixture's frozen 80-name manifest and
the reviewed package-specific manifests. The builder validates the resulting localized
object, not a compiler-dependent optional subset. Unsorted or
duplicate public-export inputs are likewise rejected rather than normalized.

The existing `transformer.error_public` facade and every installed public Eshkol
package stub import `transformer.error_consumer`, never
`transformer.error_internal` or `transformer.error_core`. Thus either import order
resolves the unchanged six accessors to the one artifact registry; no public source
closure creates the source `error_core` registry. Package stubs declare only their
narrow package-specific public wrappers. The completed object is archived or linked
before arbitrary application source is compiled. Never pass untrusted application
source as a trusted package root: a reference present before localization could be
resolved inside the partial link.

Pinned Eshkol crashes while compiling a direct function whose body contains a
lexical `extern`. Public stubs therefore use a top-level safe-only closure followed
by a direct fixed-arity public function. Because `provide` is informational, the
safe alias is technically source-reachable; it conveys exactly the same accessor or
package-specific capability and no generic E1 privilege. Privileged bridge helpers
remain absent from the public closure and local in the artifact.

Exactly one completed E1B artifact may own the E1 registry in a process. Combining
independently built E1B artifacts would create distinct identity registries and is
unsupported.

## Threat boundary and limitations

The in-scope attacker is arbitrary compiled Eshkol using only installed public
stubs and the completed public artifact, including code that guesses every private
Eshkol, C, generated-companion, and external-variable name. The security property
does not rely on symbol-name secrecy; private names may remain visible in `strings`.

Arbitrary malicious native object injection into the trusted partial link, a
modified compiler, debugger/process-memory mutation, and use of private build
intermediates are outside scope. Only the localized object, installed public stubs,
and evidence directory belong in a public consumer package. Reviewed private source
remains in the project repository but is not installed; LLVM IR/bitcode,
raw/private/bridge objects, and the unlocalized combined object are build temporaries.

E1B is AOT-only. Publishing bitcode would bypass ELF local binding, so no E1B JIT
artifact is provided or claimed. E1's independent AOT/JIT semantic suite remains in
force. The boundary is process-local and inherits E1's process-lifetime registry,
linear lookup, monotonic memory use, and lack of verified thread safety. Compiler,
tagged-value, vector-layout, calling-convention, platform, JIT, or module-opacity
changes require re-review.

Run the focused evidence gate with:

```sh
/usr/bin/bash -c 'make test-e1b'
```

D1, P1, and X1 must each build one trusted package artifact with only their reviewed narrow
public operations, compile callers after localization, and repeat closure, import-
both, malformed-error, symbol, link, and full-suite gates. A larger trusted closure
requires a reviewed repository-owned manifest/policy update, never a package argument
or build-derived optional set. They must not expose the generic five-value seam or
combine independent E1 registries.
