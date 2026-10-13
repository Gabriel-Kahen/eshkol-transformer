# O2 issued-state to C2 compiled-provider gate

**Noninstalled test source; integration HOLD.** This gate is based on the
independently source-reviewed O2 adoption candidate `0d2e31e8` (tree
`ce68555b`) and the design contract `cd766f0b`. It compiles the actual
`o2_wave2_extension.esk`, private issued-state adoption extension, and
`c2_training_state_extension.esk` in one Eshkol registry universe. It does
not install a package, add an API, invoke C2 SAVE, or publish a G3-R pair.

The test-only C bridge in `tests/g3r/o2_issued_c2_provider_bridge.c` reads
the fourteen roster entries of one registered optimizer and calls the
accepted native same-owner O2 hold/issue functions. This is deliberately a
**seam proof**, not M3 roster authentication: the eventual coordinator must
obtain and verify those fourteen parameter owners and identities from its
sealed M3 model. The bridge includes the native O2 provider exactly once;
the private Eshkol wrapper registers precisely the state pointer issued by
that provider. Existing C2 compose then claims the resulting shell.

`scripts/test-g3r-o2-issued-c2-provider.sh /fresh/absolute/output` requires
the repository's pinned supported Eshkol/Clang toolchain and writes all
artifacts under the supplied fresh path. The runtime test uses deterministic
D2 resources and authenticated C2 cursor/config controls. It checks an
equal-byte different native state, a forged shell, and a second registered
equal-bit optimizer on independent parameter owners. The registered foreign
case compares held O2 control shells and ledgers byte for byte before and
after rejection, checks the held moment invariant and live counts, then
retries adoption on the original owner. The gate also checks exact
issued pointer registration and duplicate rejection, counter and P1 alias
preflight failures, preclaim coordinator release, injected second-claim
rollback, same-owner retry, C2 claim consumption, exclusion of public O2
release while claimed, and postclaim C2-only retirement. It compares two AOT
artifacts and runtime outputs. The native bridge has no production build
reference.

Admission still needs a supported-host run with sealed logs, independent
source/test review, then the exact G3/M3/F32/P1/O2/A2/C2 joint provider and
its fault matrix. This gate does not prove writer outcomes, tokenizer-control
issuance by a live trainer, companion pairing, restore, or continuation.
