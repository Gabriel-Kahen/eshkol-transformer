# CLI3 corpus validate slice

Status: shared standalone/installed source and command tests added;
pinned-toolchain compilation and runtime verification of this exact tree remain
pending. This is the D1-only part of CLI3, not acceptance of the full CLI3
workstream.

## Command contract

```text
eshkol-transformer corpus validate DIR [--max-manifest-bytes N] [--max-shard-bytes N] [--max-total-tokens N]
```

`DIR` is a nonempty path. Options may appear once each, in any order after `DIR`.
Values are unsigned decimal text in the signed 64-bit range. Byte limits must be
positive; the token limit may be zero. Defaults are 67,108,864 bytes for the
manifest, 67,108,864 bytes per shard, and 1,000,000,000 total tokens. A caller
must raise these limits explicitly for a larger corpus.

Success exits 0, writes no stderr, and writes exactly one stdout line:

```text
shards=N tokens=N vocab=N limit=N bytes=N fingerprint_hex=LOWERCASE_HEX
```

Fields come from the six public D1 summary accessors; `bytes` is the sum of shard
file bytes, not the manifest size. `fingerprint_hex` encodes the opaque UTF-8
fingerprint bytes, so even a fingerprint containing newline or control characters
remains on one output line. No summary is printed until every shard validates.

Argument errors exit 64 with empty stdout and two stderr lines: `error: MESSAGE`
and the usage line. A structured D1 failure exits 2 with empty stdout and one
stderr line: `error: category=CATEGORY operation=OPERATION message=MESSAGE`.
Missing or unreadable manifest files have category `io`; corrupt shards and stored
files exceeding a supplied policy limit have category `corrupt-data`. Unknown
manifest versions have category `version-mismatch`, and unknown required features
have category `unsupported`. An unexpected non-transformer condition exits 70
with empty stdout and one stderr line.

The shared `transformer.cli3_corpus_validate` handler calls only public
`token-corpus-validate`, its summary accessors, and public error accessors.
The standalone command requires the reviewed D1 archive. Its D1-only staged
facade in `scripts/test-cli3-corpus-validate.sh` is extracted from the same
markers as `scripts/test-d1.sh`; no D2 wrapper or private D1 API is linked.
The installed entry intercepts only `corpus validate` before existing CLI3
dispatch and imports the shared handler beside its read-only staged D1 facade.
The command is Eshkol-native. Python is used only for command-level tests and
fixture mutation.

Compiled gate, once the exact pinned toolchain is available:

```text
scripts/test-cli3-corpus-validate.sh /absolute/new/evidence-directory
```

The evidence path must be absolute, outside the project checkout (including paths
through symlinked parent directories), and must not already exist. The gate checks
these conditions before creating the directory or verifying the toolchain. It
refuses to overwrite an occupied path and
never removes it. `metadata.tsv` records the initial project HEAD/tree/clean state,
lock hash, compiler hash, and compiler provenance identity. The gate rejects a
dirty or changed project checkout and validates the compiler hash and provenance
against the lock before building D1. Final HEAD/tree/status files expose source
drift during a long gate and turn that run into a failure. Metadata remains in the
evidence directory even when toolchain verification fails. `overall.status`
records the gate exit. Each build, compile,
and command-test step writes `steps/NAME.stdout`, `steps/NAME.stderr`, and
`steps/NAME.status` on success or failure. `artifacts/` retains the command and
fixture executables, their depfiles, staged public facade, cache, and a copy of
the rebuilt D1 archive with its evidence. An early failure retains all completed
step records; later steps have no record because they were not run.

The gate rebuilds the D1 archive from the exact tree, checks its actual exported
symbols and evidence, and compiles from a staged public facade with depfile
boundary checks. It covers a valid corpus, opaque newline fingerprint, missing
manifest, corrupt shard, version and feature categories, all three exceeded
limits, and invalid command arguments.

The required `scripts/test-cli3-production-generate.sh` gate also runs
`tests/cli3/test_installed_corpus_validate.py` against the installed CLI3 binary
and a corpus built by that binary, then runs the existing CLI3 command suite.
It covers exact plain output, bounded options, invalid and duplicate options,
signed-64 overflow, missing/corrupt/version/feature corpus cases, policy limits,
and old-command dispatch. This installed gate has not yet run on this tree.

The lightweight `tests/cli3/test_gate_evidence.py` checks occupied-path rejection
and retained metadata, logs, and status on toolchain failure. It rejects direct
and symlinked paths into the checkout before creation. It also checks dirty
checkout rejection with an alternate Git index, without changing repository files.
These tests do not build D1 or compile code.
