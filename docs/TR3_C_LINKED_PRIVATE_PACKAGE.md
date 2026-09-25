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

The fixed compiler's `createLibraryInitFunction` emits
`void __eshkol_lib_init__(void *arena)`. Its AOT entry initializes the hosted
runtime, obtains a global arena, and writes `__repl_shared_arena` before
executing package code. The existing E1B bridge calls the initializer with
`get_global_arena_shared()` inside a parallel scope and exception handler. The
linked private archive witness now follows those exact steps once. On the
pinned runtime it completes without stderr, restores the prior exception
handler, leaves the runtime's shared arena slot intact, and raises root-arena
used bytes from 0 to 7,234,848. The root arena remains runtime-owned; the
witness does not destroy or roll it back.

The linked `.so` still exports only the initializer and version marker.
`dlsym` explicitly rejects its arena getter and shared-arena slot, so a host
cannot supply the **same runtime's** initialized arena through this dynamic
boundary. The private archive witness proves initialization, not a callable
dynamic-host bridge. The real initializer has no accepted deterministic
failure injection here; its exception rollback and partial-root retention on
failure remain unmeasured. E1B's separate injected-initializer test covers
handler unwinding and retry only. The shared library does not expose
`trainer-create`, `trainer-state`, `trainer-load-state!` or a C-callable trainer
bridge. The accepted source-compiled private 151-check witness remains the
construct/snapshot/SAVE/LOAD/live-restore runtime evidence. A reviewed
same-runtime arena/exception bridge and package-facing trainer entry contract
are still required before a dynamic C ABI witness or public resume claim.
