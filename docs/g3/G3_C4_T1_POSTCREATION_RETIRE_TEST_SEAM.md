# Private T1 postcreation retirement test seam

Status: **source-only candidate** based on F `80ae4ac5`; no native, compiler,
loader, sanitizer, or Docker result is claimed.

The production T1 contract retains sealed encoded shells for process lifetime.
`t1-shell-abort` remains construction rollback only. A separate
`ET_T1_SEALED_RELEASE_TEST_ONLY` build, together with
`ET_T1_I64_SHELL_TESTING` and `ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE`, exposes a
private retirement hook. It admits only an authentic sealed rank-1 length-two
T1 shell with its unchanged, live I1 tensor/borrow/view pair. Successful
retirement ends the borrow, destroys the tensor, removes the native live
admission, and keeps the shell allocation on a private tombstone list. The
old raw pointer cannot authenticate a later allocation. Denied preflight
does not change either owner. An injected refusal after borrow end terminates
the child process before unlinking or publishing success.

The one-root private fixture
`tests/g3g_manual/private_t1_retirement_runtime.esk` stages an exact one-entry
Eshkol registry removal before calling the native hook. Only its successful
return permits the allocation-free registry swap. It creates a genuine C4 P2
input from `[0,255]` before retirement, then checks that native/T1 access
fails and C4-owned history still captures and admits EOS and non-EOS G2
branches with their original IDs, terminal words, cache, and RNG. The C test
`tests/t1/test_sealed_retire.c` covers native refusal, live I1 storage removal,
raw-pointer ABA, and a separate fail-stop child at the post-borrow cut.

The next gate is independent source/tests review, then an immutable bounded
normal/sanitizer launch plan that pins exact package topology, toolchain,
objects, sources, and public/private symbol closure. The existing installed
public fixture and public T1 ABI are unchanged. Compiled T1-release evidence
and production reclamation remain **HOLD**.
