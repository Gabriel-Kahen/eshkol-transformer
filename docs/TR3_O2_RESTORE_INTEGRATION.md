# TR3-C native O2 restore integration

Status: accepted bounded native participant; full trainer restore remains pending.

The accepted source is `2aeefad282468bb8dab2aa9b5b203f95c61b468d`, tree
`854b585187a8313e08751b111183a6e6e34db16a`, on the accepted I2/G3 bridge
base `fc924df9`. It preserves the six O2 implementation, manifest and test files
from accepted leaf `6412b9e`, adds the four reviewed shared-file changes, and
extends the test runner with simultaneous I2-restore/G3-construction symbol checks.
All other base paths are preserved. Independent review approved this exact source.

Root integrated it as `5dbac5303191bfde9c131b18e3260c39fe7f66cc`, tree
`a58782c029d1205ff1b6146c649c29647d26c284`. Every production subtree and all ten
changed paths are identical to the tested source. The preceding integration adds
only accepted P1 tests and acceptance documentation. This is source-equivalent
native evidence, not a new full-suite CI or merged-main claim.

## Private contract

The feature-gated participant adds prepare, append, check, commit-native, finalize
and abort operations under `et_o2_tr3_c_restore_*_v1`. The LP64 request is 728
bytes, with fourteen 48-byte parameter entries. Receiver update count and target
checkpoint update count are distinct inputs. The participant shares the existing
optimizer busy authority and I2 owned-clone authorizer; it introduces no second
optimizer or tensor registry. Commit/finalize consume the checked I2 transaction
in the order required by the live restore contract.

There is no public trainer binding in this change. A joint restore still needs
the accepted lease, model/optimizer/dataset/configuration state composition and
its complete failure and resumed-trajectory proof.

## Supported evidence

Root ran the original O2 assertions and commands against read-only source in
immutable image `sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`
(Ubuntu 22.04, Clang 21.1.8 and GCC 11). The wrapper changes only the location of
`common.sh` and preservation of temporary evidence after execution. It completed
with exit 0 in 118.82 seconds, including the original I2 restore gate.

- GCC and Clang repeated runs and sanitizer run: 19,696 checks each.
- One process per horizon: 1,024 abort cycles and 8,192 commit cycles passed.
- Existing optimizer update/clear coexistence: 37,264 adversarial checks passed.
- Ordinary object parity, exact manifests, hidden symbols, feature-negative cases
  and public isolation passed. Joint I2/G3 bridge compilation adds exactly the
  two accepted construction definitions and one owner-preflight dependency.
- The supported native G3 bridge/owner/pins/C2 command set then completed with
  exit 0 in 68.84 seconds. Normal/repeat/sanitizer outputs match, runtime stderr
  is empty, and ordinary/private C2 transcripts are identical.

Evidence directory: `tr3-o2-union-2aeefad-supported` under the orchestrator's
September 22, 2026 artifact directory. Its 1,380-file checksum manifest is
`102a84805e2c4fc81d06f8888d1a4b53267e2947df4df66dbea69aae5ba3ff9e`.
It includes exact commands, source hashes, raw output, the preservation-only
patch and root merge-source audit. No Eshkol compiler rebuild was needed.

## Limits and remaining dependencies

O2 retains 1,464 bytes of terminal control state per restore: total retention is
linear. The corrected tests release all fourteen temporary model clones after
I2 commit and check genuine tensor/clone counters after every cycle. No flat total
memory, natural host-OOM, public trainer, joint restore, full package, full CI or
exact resumed training claim is made. The separate runtime allocation-failure
repair, lease acceptance and full native trajectory remain required.
