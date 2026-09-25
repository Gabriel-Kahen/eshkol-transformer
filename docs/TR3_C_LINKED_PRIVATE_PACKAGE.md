# TR3-C linked private package boundary

The reviewed fixed Eshkol compiler `97c40c9d` can compile the accepted
46-source TR3 root in linked shared-library mode. This bounded gate compiles
the 28 accepted native translation units with their required private-feature
defines, checks their exact 64-path source/header closure, and combines them
with the Eshkol object. It localizes every combined definition except
`__eshkol_lib_init__`, then links one private archive and a versioned shared
library with `--no-undefined`. No predecessor package or runtime pin changes.

The exact boundary is 4,110 raw global definitions, 166 unresolved
runtime/system symbols in the localized object, one archive member, and two
dynamic entries: the version marker and `__eshkol_lib_init__`. The linked
library has no unresolved trusted `et_*`, TR3 or C2 references. A whole-archive
control link and `dlopen` succeed; external static links to the private TR3
lease and native restore symbols fail by name. `dlsym` sees the initializer but
cannot see those private symbols or the result-cell restore entry.

The shared library does not expose `trainer-create`, `trainer-state`,
`trainer-load-state!` or a C-callable trainer bridge. The `dlopen` witness does
not invoke the initializer, construct a trainer or exercise package runtime
semantics. The accepted source-compiled private 151-check witness remains the
only construct/snapshot/SAVE/LOAD/live-restore runtime evidence. Next work
needs a reviewed initializer/exception boundary and package-facing trainer
entry contract, followed by an actual C ABI runtime witness and ownership
tests before any public facade or resume claim.
