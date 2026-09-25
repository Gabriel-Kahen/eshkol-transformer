# G3-T private RNG owner mapping

This leaf adds the accepted source-private Eshkol `g3t-generation-output-rng`
and `g3t-generation-rng-release!` operations. The output accessor first
authenticates the exact live G3-T output shell through the same-aggregate
registry. It preallocates and roots a new shell, pending 12-slot `rng` entry,
and registry cons cell before requesting the native kind-8 clone. It then
attaches the native owner and publishes the already allocated entry with no
further allocation. If anything raises after native clone creation, its guard
releases that exact kind-8 owner and tombstones the pending entry.

The RNG shell holds no output, generator, model, or tokenizer root. It survives
output release and generator close. Typed release accepts only an authentic
`rng` entry, calls native `rng_release`, and leaves an idempotent tombstone;
generic tensor and output release reject the wrong kind. The native clone
copies inline words and is independent of an ID-tensor borrow. A native
allocation cut leaves the source output and both live registries unchanged.

The pinned, network-disabled Ubuntu 22.04/LLVM 21 gate passes 46,508 checks
each in normal, repeat, and ASan/UBSan/LSan modes with identical stdout and
empty compile/runtime stderr. Eight G3-T source contracts and Q0 4/4 pass;
the production object excludes test hooks. Sanitizer settings include
`detect_leaks=1`, `halt_on_error=1`, and arena poisoning. The corrected test
and loaded extension also passed strict Eshkol parse/type preflight.

The focused witness covers P2/G0 detached lifetime, P1/G0 exact words,
forged/wrong-kind/dead identities, wrong typed release, allocation failure,
borrowed output IDs, zeroization, and idempotence. P1/G1 retains a native
snapshot witness; no Eshkol G1 output shell is produced by the current private
test route. The seeded `generator-create` still rejects `:rng`; adding RNG
input admission to that constructor is a separate accepted seam. No public
facade, package export, CLI, or generation claim is added.
