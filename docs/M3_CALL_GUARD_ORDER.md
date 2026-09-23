# M3 call cleanup-handler ordering follow-up

Status: implementation candidate for
[issue #117](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/117).
This is a narrow follow-up to the accepted shared-call implementation in PR #105;
it does not revoke or broaden that implementation's recorded evidence.

`m3-call` retains one outer `m3t-boundary`, its pre-mutation reentry rejection,
one inner cleanup `guard`, and the existing normal and exceptional clears. The
two runtime-source changes move busy-state publication into the inner guard and
annotate `m3t-rethrow-raw` with the private emergency-rethrow parameter modifier.
The cleanup handler is installed first, then `m3-call-state` becomes true, then
the body executes. Under the repaired runtime's checked allocation-failure
behavior, failure to allocate that handler transfers to the already installed
outer boundary while the busy state is still false. Once the handler exists,
body failure still clears the state before raw rethrow, and normal return still
clears it before returning the answer. Every physical raw-rethrow entry checks
the exact caught value before Scheme predicates or error construction can run.

The focused source-contract test parses the exact macro and rethrow definition.
It rejects the six ordering/cleanup mutations plus a missing modifier, a wrong
selected formal and a modifier moved ahead of the rethrow body.
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
The explicit repaired-runtime gate is
`scripts/test-m3cg-repaired-handler.sh RUNTIME_SOURCE RUNTIME_BUILD OUTPUT`.
It requires clean frozen runtime source commit
`81298b4a9608fb92eb6f351a2eabd8392da7d9ef` (tree
`7669312845a9d8d372006af52271045e69505813`), runner SHA-256
`4a0e6303f7b85ed06fb753b52b62155235a3a77bca6c32aeb17241a28ed80be1`
and runtime archive SHA-256
`c32bb593ac1f365f3cbeaefd581704c4be029a4aa8877db29463d0e12356c168`,
plus Clang 21.1.8. It changes no dependency pin and builds no runtime.

The fixture source-loads the actual M3 package and common adapters, retaining the
ordinary 46-entry witness. GNU `--wrap=malloc` fails exactly
`sizeof(eshkol_exception_handler_t)` persistently once the selected defect is
reached. Existing recycled handlers are occupied until an actual allocation is
observed, rather than assuming a free-list size. The prepublication experiment
allows the outer `m3t-boundary` push, fails the second `m3-call` cleanup push and
proves exact condition 5 crosses the outer `m3t-rethrow-raw` entry with the body
untouched and the busy flag false. The body experiment allows both production
guards, then fails a real one-frame runtime reservation; inner cleanup clears
busy before exact condition 5 crosses both `m3t-rethrow-raw` entries. Handler
allocation stays failed through the final outer catch in both experiments.

`m3t-rethrow-raw` uses the compiler-private
`:runtime-emergency-rethrow-param caught` modifier, so canonicalization occurs at
each physical entry before Scheme predicates or E1 construction. The shim obtains
condition 5's static identity from the frozen runtime and checks that identity at
every caught boundary, exact guard-stack progression, body entry counts and idle
retry. A real checked input create/release after each experiment proves reuse.

The test records raw depfiles, a sorted SHA-256 manifest of every compiler-reported
Eshkol input plus all manually compiled and reviewed-delta sources, the exact
inherited-plus-two-runner global inventory, link map, runtime/source/executable
hashes, stdout and stderr. This is a focused allocation-failure witness, not
evidence for every failure path, sanitizer coverage or full CI. Root still owns
the immutable-runtime adoption and acceptance union.

No E3/P1/D2 consumer implementation, G3 runtime, public ABI, numerical behavior,
full-training capability, or new pin is part of this correction. Root owns the
runtime-pin union, the single full-CI dispatch, acceptance, and merge.
