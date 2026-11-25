# G3-R/C2 frozen Wave 3 preparations — 2026-11-25

Status: **prepared, independently reviewed, unclaimed, unrun; runtime HOLD**.

The source mount is the separate clean detached worktree
`/home/gabe/.cache/eshkol-g3r-frozen-source-20261125`, fixed at HEAD
`4d31964456212f84837387c2ba2bb23803438f69` and tree
`19506579245818f08d93c673411053d24690282e`. The official `source_pin`
helper bound all 2,631 tracked files, and its complete source inventory found no
ignored or untracked extras. This documentation branch is not the pinned source.

The fresh evidence root is
`/home/gabe/.codex/evidence/eshkol-transformer/g3r-c2-prep-4d319644-20261125`.
It contains independent `replay` and `save` packages, each with only a sealed
`preparation/` directory. The copied fe9 Eshkol source and build dependencies
have exact 4,118-file and 6-file inventories; both match the retained November
15 originals. The historical guest recipes were adapted only for the new source
mount. Replay retains its sealed independent G3-S observer and oracle. Both
configs pin the Docker client, UID-owned socket, root and Ubuntu 22.04 image ID.

The independently recomputed, caller-supplied `preparation/seal.json` SHA-256s
are:

| Profile | Seal SHA-256 |
| --- | --- |
| Replay | `333ceb497bb29bb5e5b107444c9298d7ea98c2ed0a0f742e2dccf365fa62456f` |
| Authentic SAVE | `5332260a17411eeb2684e7be0aa6cd28af948b46b1752e79b0af931c6419ea1c` |

The exact commands are recorded as argv arrays in the external `handoff.json`.
From the frozen source worktree, the replay command is:

```bash
flock -x /home/gabe/.cache/eshkol-transformer-orchestration-build.lock \
  python3 -B -m tests.g3r.one_use_gate \
  /home/gabe/.codex/evidence/eshkol-transformer/g3r-c2-prep-4d319644-20261125/replay \
  333ceb497bb29bb5e5b107444c9298d7ea98c2ed0a0f742e2dccf365fa62456f
```

For authentic SAVE, replace `replay` with `save` and the seal with
`5332260a17411eeb2684e7be0aa6cd28af948b46b1752e79b0af931c6419ea1c`.
These are future one-use commands, not evidence of execution. The controller
must recheck at launch at least **16 GiB available RAM, 8 GiB free swap, and
32 GiB free on both package/output and Docker storage** under its locks. The
preparation-time host sample had 15,511,440 KiB available RAM and 3,536,420
KiB free swap, below both memory floors; no heavy gate was attempted.

Preparation checks passed: official `Inputs.verify()` for both packages,
byte-for-byte dependency inventory comparison, `bash -n` on both guest scripts,
Python syntax parsing for the guest helpers, and read-only Docker
image/socket/client/root inspection. An independent reviewer checked each
exact recipe, profile, config inventory and both SHA-256 seals. The retained
November 15 seals and all historical results were left intact. Generated
replay, SAVE/reader, runtime result review and E3 section 9 remain open.
