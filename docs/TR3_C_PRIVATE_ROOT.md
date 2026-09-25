# TR3-C private root closure

`native/tr3_c_private_package_root.esk` composes the accepted D2/O2/P1/I2/C2/K2/TR3 lease, snapshot, checkpoint SAVE/LOAD, joint restore and private result-cell restore in one Eshkol identity universe. It adds no trainer algorithm, public facade, or installed package. The pinned f31/LLVM 21 internal-convention `--emit-object` compilation has a fixed 46-file Eshkol source closure and 334-symbol undefined boundary. `scripts/test-tr3-c-private-package.sh` checks both complete manifests and the five private seam definitions, then compiles a genuine construct/snapshot/SAVE/LOAD/result-cell restore witness against that root.

The internal-convention object is not C-callable and retains 3,753 unlocalized global definitions. The canonical `--shared-lib --dump-ir` package path on the same pinned compiler fails before producing IR with 52 `Tail transfer: no public entry` diagnostics, beginning with `tr3-lease-vectors-overlap?__eshkol_tail_body` and extending into accepted M3T, O2 and D2 functions. This is a specific compiler/package prerequisite. No public export or private rename list is asserted for this raw object; E1B bridge, symbol localization, exact native package closure, and the public `trainer-create` facade remain downstream. The accepted private `tr3-lease-create-internal` still constructs the fixed-profile receiver in the runtime witness.

The documented `--shared-lib -c` object flavor now compiles that same root in
library mode without creating C ABI export thunks. Its distinct exact boundary
is the unchanged 46-source closure, 2,593 defined symbols and 330 undefined
symbols; the predecessor 334-symbol ordinary-object manifest is unchanged.
`scripts/test-tr3-c-private-package-mode.sh` checks the full symbol lists,
the private lease/snapshot/SAVE/restore entries and `__eshkol_lib_init__`.
It rejects a program `main`, any public trainer C symbol or decorated C ABI
implementation, then records the exact 52-error linked-mode failure.
This is a private relocatable object, not an installed or C-callable package.

The remaining compiler contract is specific: when linked `--shared-lib` renames
an Eshkol implementation `f` to `f__eshkol_internal_abi` and places a C ABI
thunk at `f`, tail-transfer finalization must resolve `f__eshkol_tail_body`
against that internal implementation with the same Eshkol function type. The
pinned compiler instead compares it with the differently typed C thunk and
rejects before IR publication. A reviewed upstream fix and a genuine linked
package, bridge, localization and hostile-link gate must precede public TR3
exports.
