# G3-R post-adoption compiled gate: source contract and HOLD

The post-adoption fixture has two isolated source modes: a successful abandon
after a rooted-shell longjmp, and a test-only O2 release fault inside the real
abandon path. Neither mode has been compiled or run. The private capture
source consumes the accepted 17-word current-main held image; this fixture
does not admit a public SAVE or a continuation.

`tests/g3r/post_adoption_fe9_compiler.tsv` is the dedicated **source-only**
compiler identity. It pins the supported fe9 source/tree, runner/runtime
hashes, Clang/LLVM 21.1.8 and f31 image, and names the accepted C7b release
digest. The preparer checks its exact bytes and the toolchain lock, copies it
from the exact committed Git object to `COMPILER.tsv`, rechecks staged bytes
against that object and pinned SHA, then atomically publishes the prepared
bundle and checksums. The recovered
81298 TR3 candidate is not compiler authority for this gate. These hashes do
not prove that a capture executable was built.

The current `--verify-runtime` mode always exits 74. The former option to
choose any executable and corpus could pass status/marker checks without
proving that either came from the prepared source. An explicit release flag
does not override this HOLD.

Before replacing that denial, an independently reviewed builder must pin the
exact capture-native C units, private macros, headers and link inputs; compile
the **prepared** Eshkol root with a checked 96-leaf dependency closure; and
record argv, environment, working directory, native and Eshkol depfiles,
objects, archive and final ELF hashes. It must also define and review the
deterministic V256 corpus generator, its arguments and every output byte.
The existing private-union and TR3 restoration recipes are different targets
and cannot be used as this build recipe.

The builder must seal these outputs with the prepared source and compiler
identities in a fresh attestation. An independent receipt must pin that seal.
Only then may a verifier derive the executable and corpus paths from the
receipt, rehash them before both isolated processes, assert normal success
and fail-stop exit 134, and retain checked post-run evidence. A one-heavy,
one-use supported-host resource gate and measured disk/memory headroom are
also required. Until those contracts and independent reviews exist, compiled
capture admission remains HOLD.
