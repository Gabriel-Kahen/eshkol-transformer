# G3-R/C2 one-use launcher contract

The development-only controller is `tests/g3r/one_use_gate.py`. It runs one
reviewed replay or authentic-SAVE preparation:

```bash
flock -x /home/gabe/.cache/eshkol-transformer-orchestration-build.lock \
  python3 -B -m tests.g3r.one_use_gate /absolute/fresh/package EXPECTED_SEAL_SHA256
```

`source_pin` and `seal_preparation` are preparation helpers. The caller supplies
the independently checked SHA-256 of the versioned `preparation/seal.json`;
the package cannot choose its own expected identity at launch. JSON is data,
not executable checkpoint content. Each preparation contains one closed gate
profile, its guest recipe and exact required evidence inventory.

## Input and one-use checks

- Bind the committed source HEAD/tree, clean tracked/untracked status, complete
  source file inventory (including rejection of ignored extras), dependency
  inventories, recipe files, Docker client, socket endpoint and image identity.
  Changed or extra inputs fail.
- Reject traversal and symlink components with no-follow filesystem operations;
  the fixed Docker vector mounts inputs read-only. Only `out/guest` is mounted
  writable at `/out`; host CID, inspection logs, Docker config and result records
  remain outside that writable subtree.
- Hold `/tmp/eshkol-transformer-heavy-build.lock` and
  `/home/gabe/.codex/eshkol-heavy-docker.lock`; recheck inputs and the unchanged
  16 GiB available-RAM, 8 GiB free-swap and 32 GiB disk floors under the locks.
  Unmet resources return status 75 before Docker launch.
- An exclusive `CLAIM.json` and fresh output inventory prevent a second use.
  A failed claimed attempt remains consumed; its evidence is preserved.
  Scratch and `TMPDIR` belong on the `/home`-backed output mount.

## Custody and results

The controller binds container name, CID, image and source/preparation labels.
It disables Docker signal proxying and retains both locks during client
interruption or uncertain inspection until the owned container is terminal,
or two spaced exact absence observations agree. It never kills a guessed
container. The actual Docker missing-container dialect (`[]` stdout with the
exact named error and status 1) is handled by the same strict classifier in
prelaunch and custody checks; contradictory output remains uncertain.

A successful container exit alone is insufficient. Required per-step command,
stdout, stderr and exit records must be complete; the controller checks runtime
oracles, normal/repeat/sanitizer parity and artifact identities before sealing
a PASS result. It requires nonempty per-step argv records; independent review
of the sealed recipe and resulting command records establishes command identity.
SAVE additionally checks the no-SAVE control, authentic producer
result, canonical seed-to-saved byte identity, companion digest and unchanged pair.
Passing output inventories allow only the accepted `link.g3r -> valid.g3r`
and `fifo.g3r` negative fixtures. Failure seals retain unexpected links or
special nodes as metadata without dereferencing them; those nodes cannot pass.

Focused control/oracle and Q0 tests exercise these controls
with fake Docker, resources and clocks, including dirty/stale/reused inputs,
path escapes, missing logs, artifact substitutions, interrupted clients,
inspection uncertainty and logging failures. A real read-only Docker absence
probe supplies the compatibility regression. This is control evidence, not
generated Eshkol execution. Custody evidence covers handled interrupts and
daemon uncertainty, not uncatchable process or host failure.

## Current admission bound

The November 15 replay/SAVE preparations and preflight receipts are retained
under `build/evidence/g3r-launcher-hardening-20261115/concrete-aece425d/` in
the historical integration worktree. Their independently recorded seals are
`1646205b598774c3ec6fa0a0d842aceae52b118bee836cfe132defe0116c8e0f`
and `d2e511e35304ca0fdc8c60bbfa3380604c93c5a659351c13005fbb1511378abe`,
respectively. Both bind source head `aece425db480ffb1e16ccea7a4e393b2dfb9e974`
and tree `4be3eff7d6678f3ad6eb5545e3807c83797e89a7`; they are ineligible
for this composed tree. Earlier failed and unlaunched bundles remain preserved.

The [current source reconciliation](CPU_V1_SOURCE_RECONCILIATION_20261125.md)
requires new preparations pinned to the final committed head/tree, copied
dependencies with exact inventories, independent recipe/config/seal review and
a caller-supplied seal SHA-256. Keep preparations and outputs outside the mounted
source tree so ignored files cannot become unpinned source inputs. The launch
host must satisfy 16 GiB available
RAM, 8 GiB free swap and 32 GiB free on both scratch/output and Docker storage,
rechecked under the locks. No heavy replay/SAVE execution is established here.
Planned sanitizer modes instrument native providers and link sanitizer runtimes;
they do not instrument generated Eshkol IR. Independent runtime-result review
and E3 §9 remain separate admission gates.
