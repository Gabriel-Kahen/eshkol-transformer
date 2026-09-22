# M3 call cleanup-handler ordering follow-up

Status: implementation candidate for
[issue #117](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/117).
This is a narrow follow-up to the accepted shared-call implementation in PR #105;
it does not revoke or broaden that implementation's recorded evidence.

`m3-call` retains one outer `m3t-boundary`, its pre-mutation reentry rejection,
one inner cleanup `guard`, and the existing normal and exceptional clears. The
only runtime-source change moves the busy-state publication into the inner guard:
the cleanup handler is installed first, then `m3-call-state` becomes true, then
the body executes. Under the repaired runtime's checked allocation-failure
behavior, failure to allocate that handler transfers to the already installed
outer boundary while the busy state is still false. Once the handler exists,
body failure still clears the state before raw rethrow, and normal return still
clears it before returning the answer.

The focused source-contract test parses the exact macro and rejects six unsafe
mutations: publishing before handler installation, removing the exceptional
clear, removing the normal clear, moving the reentry check after the guard,
dropping its fixed rejection, and bypassing the exact body binding.
The predecessor-source manifest records the intentional new source digest. All
38 checked M3T adapters, eight M3 entries, public/private inventories, fixed14 pin
code, numerical schedules, compiler pin, and CI topology remain unchanged.

Development checks cover the structural contract, shared closure, Python
isolation, ordinary nested/reentry/failure cleanup behavior, genuine M3
forward/logits, and repeated fixed14 pin cycles. Local package evidence uses the
explicitly unsupported CachyOS / Clang-LLVM 22.1.6 compatibility host with pinned
Eshkol dependencies read-only; it is not supported-runtime evidence.

At source commit `d3be9d33b79d5c2c0728971acd0cc545c6a9d774`, the ordinary
package gate passed against pinned Eshkol `90cbd7130f47b8184bcc77b8d5c1b0026da980de`.
The normal artifact and both instrumented artifacts ran with empty stderr. The
two fresh instrumented builds were byte-identical: private IR SHA-256
`21633b3c0506033599bed4411bf5bdd8ae34b02484ea4fdda4ca1274b2b3a24e`,
combined object `c0d601548a3e999cc8867ed58cd478de5074dafd80390784f9f3d8642d974b4d`,
and caller `a65235e60b3a59a6275282b42986aaa9070116b71fb1257d58ef869339f72bf5`.
This proves ordinary package compatibility and determinism on that local lane;
it does not inject exception-handler allocation failure.

Acceptance additionally requires a compiled witness using the repaired runtime.
That witness must fail allocation of this exact inner exception handler while an
outer boundary is live, prove the model-call body did not mutate state, prove the
busy flag stayed false, then execute a subsequent normal M3/shared call
successfully. Static source order or IR inspection cannot replace that witness.
The upstream instrumented-runtime revision and artifacts are still pending, so
this candidate does not yet claim acceptance, supported execution, or coverage of
every allocation/error path.

No E3/P1/D2 consumer implementation, G3 runtime, public ABI, numerical behavior,
full-training capability, or new pin is part of this correction. Root owns the
runtime-pin union, the single full-CI dispatch, acceptance, and merge.
