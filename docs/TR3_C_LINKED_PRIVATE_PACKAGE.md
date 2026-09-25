# TR3-C linked private package boundary

The reviewed fixed Eshkol compiler `97c40c9d` compiles the accepted 46-source
TR3 root in linked shared-library mode. The gate builds 29 native objects with
their required private-feature defines, checks their exact 66-path
source/header closure, and combines them with the Eshkol object. It localizes
every combined definition except the versioned private initializer bridge,
then links one private archive and a shared library with `--no-undefined`.
No predecessor package or runtime pin changes.

The exact boundary is 4,111 raw global definitions, 172 unresolved
runtime/system symbols in the localized object, one archive member, and two
dynamic entries: the version marker and `et_tr3_c_private_initialize_v1`.
The linked library has no unresolved trusted `et_*`, TR3 or C2 references.
Whole-archive linking and `dlopen` succeed; external static links to the raw
initializer, private TR3 lease and native restore symbols fail by name.
`dlsym` sees the bridge but cannot see those private symbols or the result-cell
restore entry.

The fixed compiler's `createLibraryInitFunction` emits
`void __eshkol_lib_init__(void *arena)`. Its AOT entry initializes the hosted
runtime, obtains a global arena, and writes `__repl_shared_arena` before
executing package code. The existing E1B bridge calls the initializer with
`get_global_arena_shared()` inside a parallel scope and exception handler.
The private bridge performs those same steps using the library's own linked
runtime. Its no-argument C ABI returns `READY` on first success and repeats,
`BUSY` during reentrant/concurrent entry, and distinct runtime, arena,
handler and caught-exception errors. Failure clears only its atomic readiness
latch for retry; it preserves E1B's raised tagged value across handler
cleanup. The runtime owns the shared arena. The bridge neither destroys it
nor claims to roll back allocations made before an exception.

Actual `dlopen` invocation and repeat complete without stderr. The static
witness verifies restored handler and shared-arena identity, unchanged root
use on repeat, and root use rising from 0 to 7,234,848 bytes. An injected
initializer fixture verifies every error status, raised-value retention,
handler and scope cleanup, reentrant busy behavior, failure retry and repeat
without rerunning initialization. The real TR3 initializer has no accepted
deterministic failure injection here; its partial-root retention on failure
remains unmeasured.

The linked `.so` exports only the versioned bridge and version marker.
`dlsym` rejects the raw initializer, arena getter and shared-arena slot, so a
host cannot bypass the library's same-runtime setup. The shared library does
not unload in the witness after initialization: root-owned objects can retain
package code pointers, so callers must keep the library loaded for their
runtime lifetime. The bridge initializes the process-wide hosted runtime and
does not shut it down. It does
not expose `trainer-create`, `trainer-state`, `trainer-load-state!` or a
C-callable trainer bridge. The accepted source-compiled private 151-check
witness remains the construct/snapshot/SAVE/LOAD/live-restore runtime
evidence. A package-facing trainer entry contract and real C ABI trainer
ownership tests are still required before any public facade or resume claim.
