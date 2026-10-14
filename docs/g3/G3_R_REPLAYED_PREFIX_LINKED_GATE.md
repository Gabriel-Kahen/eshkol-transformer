# Private replayed-P1 linked runtime gate

This one-use gate is a **candidate**, not a linked-Eshkol admission result. It
tests only the source-private, in-memory `diagnostic-c2` P1 continuation in
`g3r-generate-replayed-p1!`. It does not authenticate a C2 persisted pair,
load a fresh process, or establish full G3-R continuation.

`scripts/test-g3r-replayed-prefix-linked.sh --check` works without a built
toolchain. Its ordered closure and SHA-256 manifest pin the actual Eshkol
source commit `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`, compiler
versions and provenance fields, native providers, Eshkol fixture, bridge, and
runner. It checks the exact linker-wrapper command, the replayed-prefix feature
flag as a token, and independent fixed SHA-256 values for the full runner and
its `build_mode` command and receipt body. Source-only semantic mutations repin
the runner SHA in the manifest; the checker still rejects them, including an
early return before the PASS receipt. It also rejects omitted manifest entries
and critical build commands.
These checks do not compile Eshkol.

For a separately released run, start from a clean immutable gate commit and
set `G3R_LINKED_RUNTIME_RELEASE=1` plus
`G3R_LINKED_EXPECTED_COMMIT=<that exact 40-character SHA>`, then invoke
`scripts/test-g3r-replayed-prefix-linked.sh --run`. The runner checks the
existing toolchain lock, clean Eshkol checkout, compiler binary/provenance,
and full Eshkol SHA before compiling. It records raw shell-escaped commands
and exit codes in `raw-commands-and-exits.log`, copies the compiler provenance,
and runs normal, repeat, and sanitizer linked fixtures. A missing dependency,
dirty checkout, changed source, failed command, stderr, or changed output
closes the gate. This document grants no release to execute it.

The fixture uses a real evaluated M3T model, exact same-registry byte
tokenizer, and two independently imported identical typed RNG owners. Manual
P1 prefill consumes no RNG block. At the uninterrupted reference's actual
G3-S dispatch, the link interposer checks all 256 logits, the eight P1 K/V
words, and all four RNG words against the committed replay prefix before
delegating to the real provider. The replayed generated call then traverses
the private wrapper's output enrollment, sample, 21-role decode and final
commit. The fixture compares emitted ID/text, output words, all generated
logits, both cache positions, committed history, and RNG. A fresh second
scenario gives both owners the reference ID as EOS and checks terminal history.
Budget-zero, authentic full-P2, and exhausted-counter prefixes must reject
without a sample; a fault injected at the real sampler boundary must preserve
the exact P1 prefix and permit the same generator to retry successfully.

The C interposer exists only in this test archive, behind test/private flags.
It neither substitutes positive provider output nor changes the public G2
path. A source-only PASS records readiness for a later linked runtime attempt,
not success of that attempt.

The first one-use compiled attempt at `5aff31c` stopped during Eshkol fixture
compilation: `g3t-generation-output-text` was referenced without loading its
already pinned `g3t_text_result_extension.esk` definition. This source repair
adds that load after final publication. A new exact-head linked run is required
before any runtime or numerical claim.
