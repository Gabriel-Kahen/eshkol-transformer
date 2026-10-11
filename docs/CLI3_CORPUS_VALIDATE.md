# CLI3 corpus validate slice

Status: source and command tests added; pinned-toolchain compilation and runtime
verification remain pending. This is the D1-only part of CLI3, not acceptance of the
full CLI3 workstream.

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
shards=N tokens=N vocab=N limit=N bytes=N fingerprint=TEXT
```

Fields come from the six public D1 summary accessors; `bytes` is the sum of shard
file bytes, not the manifest size. No summary is printed until every shard validates.

Argument errors exit 64 with empty stdout and two stderr lines: `error: MESSAGE`
and the usage line. A structured D1 failure exits 2 with empty stdout and one
stderr line: `error: category=CATEGORY operation=OPERATION message=MESSAGE`.
Missing or unreadable manifest files have category `io`; corrupt shards and stored
files exceeding a supplied policy limit have category `corrupt-data`. An unexpected
non-transformer condition exits 70 with empty stdout and one stderr line.

The command calls only public `token-corpus-validate`, its summary accessors, and
public error accessors. It requires the reviewed D1 archive. The D1-only staged
facade in `scripts/test-cli3-corpus-validate.sh` is extracted from the same markers
as `scripts/test-d1.sh`; no D2 wrapper or private D1 API is linked. The command is
Eshkol-native. Python is used only for command-level tests and fixture mutation.

Compiled gate, once the exact pinned toolchain and D1 archive are available:

```text
scripts/test-cli3-corpus-validate.sh
```

The gate covers a valid corpus, missing manifest, corrupt shard, all three
exceeded limits, and invalid command arguments. It must run with a D1 archive
built from the same pinned toolchain and source as the command.
