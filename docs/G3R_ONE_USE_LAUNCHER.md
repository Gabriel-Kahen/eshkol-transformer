# G3-R/C2 one-use launcher contract

The development-only controller is `tests/g3r/one_use_gate.py`. It runs one
reviewed replay or authentic-SAVE preparation:

```bash
python3 -B -m tests.g3r.one_use_gate /absolute/fresh/package EXPECTED_SEAL_SHA256
```

`source_pin` and `seal_preparation` are preparation helpers. The caller supplies
the independently checked SHA-256 of the versioned `preparation/seal.json`;
the package cannot choose its own expected identity at launch. JSON is data,
not executable checkpoint content. Each preparation contains one closed gate
profile, its guest recipe and exact required evidence inventory.

## Input and one-use checks

- Bind the committed source HEAD/tree, clean tracked/untracked status, tracked
  file inventory, dependency inventories, recipe files, Docker client, socket
  endpoint and image identity. Changed or extra preparation/dependency files fail.
- Reject traversal and symlink components with no-follow filesystem operations;
  the fixed Docker vector mounts inputs read-only and writes only gate output.
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
a PASS result. SAVE additionally checks the no-SAVE control, authentic producer
result, canonical seed-to-saved byte identity, companion digest and unchanged pair.
Passing output inventories allow only the accepted `link.g3r -> valid.g3r`
and `fifo.g3r` negative fixtures. Failure seals retain unexpected links or
special nodes as metadata without dereferencing them; those nodes cannot pass.

Thirty focused tests (26 control/oracle and four Q0) exercise these controls
with fake Docker, resources and clocks, including dirty/stale/reused inputs,
path escapes, missing logs, artifact substitutions, interrupted clients,
inspection uncertainty and logging failures. A real read-only Docker absence
probe supplies the compatibility regression. This is control evidence, not
generated Eshkol execution. Custody evidence covers handled interrupts and
daemon uncertainty, not uncatchable process or host failure.

## Current admission bound

Fresh concrete replay/SAVE preparations and source/toolchain preflight receipts
are retained under `build/evidence/g3r-launcher-hardening-20261115/` in the
integration worktree. Earlier failed and unlaunched bundles remain untouched;
this controller does not relabel or release them. Each fresh package still
needs its exact recipe/config/seal reviewed before a resource-qualified launch.

The [measured standard hosted runner](G3R_HOSTED_RESOURCE_QUALIFICATION_20261115.md)
fails the same RAM/swap floors. No heavy replay/SAVE execution is established
by this change. Planned sanitizer modes instrument native providers and link
sanitizer runtimes; they do not instrument generated Eshkol IR. Independent
runtime-result review and E3 §9 remain separate admission gates.
