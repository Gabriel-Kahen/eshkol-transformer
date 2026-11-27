# CLI3 fixed CPU runtime baseline

The development runner uses the retained [PR #255 CLI3 artifact](CLI3_FIXED_BINARY_ARTIFACT.md)
and the CLI3-B six-byte corpus, byte tokenizer and CPU-f32 context-two fixture
from `443dad4b`. It verifies the pinned archive and executable hashes and the
artifact seals against a clean checkout of the artifact source tree. On Ubuntu
22.04 x86-64 with the artifact libraries, Python 3.10+ and a C compiler, run
from the repository root with a fresh output directory.

```bash
python3 -B -m tests.cli3.fixed_runtime \
  /absolute/path/to/download \
  /absolute/path/to/clean-513d2ede-checkout \
  /absolute/path/to/new-output
```

The only compilation is the existing B0 Linux child measurement helper. The
runner prepares the tokenizer and corpus once, then creates a one-update seed
and checks all four checkpoints in an untimed `1+1+1+1` chain. Five serial trial
pairs alternate the order of a four-update pretrain and a three-update resume
from the same verified seed. Every measured command writes to a new path.
Each final checkpoint must match the chain byte for byte and match the pinned
25,408-byte digest `67ee14d747dfbaafeba1218eb03522b839c3d3339cd07290dc54eeefe12846ab`.
C2 cumulative counters are 14 tokens, four updates and two epochs.

`report.json` records deterministic checkpoint and JSON checks, then provides
SHA-256 and size receipts for each generated file. Raw stdout includes the
checkpoint path. The deterministic summaries check that path and omit it
before comparing repeated invocations.

The observations cover the whole child process from launch to completion,
including the loader, setup, checkpoint load on resume, updates and save.
The helper uses `CLOCK_MONOTONIC`, pidfd completion and direct-child
`wait4` peak RSS in KiB. Each pair reuses the warmed fixture, with OpenBLAS
and OpenMP thread limits set to one. The report records host affinity and load
averages. Shared caches and host activity can affect the observed values.
These measurements describe this fixed CPU CLI3 invocation on one host.

The recorded replay used a network-disabled Ubuntu 22.04.5 container image
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`
on a Ryzen 7 3700X with a Btrfs output mount and CPUs 0–15 available. The host
one-minute load average went from 8.74 to 8.76 across the sample window.

| Invocation | Elapsed ms, median (min–max) | Peak RSS KiB, median (min–max) |
| --- | ---: | ---: |
| Pretrain, four updates | 270.3 (266.3–278.1) | 242,748 (241,908–244,428) |
| Resume, three updates | 403.0 (396.5–408.7) | 248,892 (247,840–249,580) |

The replay report SHA-256 is
`5dd80fbbdfff1acbdea0b8921da114bd30c69d359056922f487f4113a227e2a9`.
