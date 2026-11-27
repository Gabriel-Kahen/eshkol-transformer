# Train a byte BPE tokenizer and build a D1 corpus

Set `cli` to the absolute path of your installed `eshkol-transformer` binary.
The commands below were replayed on Ubuntu 22.04 x86-64 using the verified
CLI3 artifact `15171126372`. [PR #262 run 45934721553](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45934721553)

The downloaded artifact runner verifies the archive and source identity.
[Artifact runner instructions](CLI3_DOWNLOADED_ARTIFACT_RUNNER.md)

## Train and inspect

Write two training documents without trailing newlines. With one merge and
minimum frequency two, the most frequent pair is `ab`. Its operand IDs are
97 and 98 and its learned ID is 256. Training with the documents in either
order produces the same tokenizer bytes.

```bash
cli=/absolute/path/to/eshkol-transformer
scratch=$(mktemp -d)
printf 'abab' >"$scratch/learn-a.txt"
printf 'ab' >"$scratch/learn-b.txt"
"$cli" tokenizer train-bpe \
  --document "$scratch/learn-a.txt" --document "$scratch/learn-b.txt" \
  --maximum-merges 1 --minimum-frequency 2 --utf8-policy raw --output "$scratch/bpe.tsv"
"$cli" tokenizer train-bpe \
  --document "$scratch/learn-b.txt" --document "$scratch/learn-a.txt" \
  --maximum-merges 1 --minimum-frequency 2 --utf8-policy raw --output "$scratch/repeat.tsv"
cmp "$scratch/bpe.tsv" "$scratch/repeat.tsv"
"$cli" tokenizer inspect --input "$scratch/bpe.tsv"
```

Both training invocations, the byte comparison, and inspection exit 0.
`tokenizer inspect` emits this JSON.

```json
{"artifact":"tokenizer","family":"byte-bpe","fingerprint":"sha256:eshkol-bpe-tokenizer-v1:b7e1a3355804a5e6ef1df3889eee835b11646e1f1981656d0f7711116f145c23","vocabulary_size":257}
```

`bpe.tsv` is the canonical T2 `eshkol-bpe-tokenizer` v1.0 ASCII TSV.
`ETBPE100` is a separate binary format with its own identity and a private
opt-in import. The installed commands here use canonical T2. [Native BPE trainer and import](T2_BPE_TRAINER.md)

## Build and validate

Build from two new documents into fresh directories. Keep their order fixed.
CLI3 encodes each document independently and D1 stores the concatenated IDs
in that order. With a shard limit of two, `manifest.etm` and both shard files
are byte-identical between builds.

```bash
printf 'aba' >"$scratch/first.txt"
printf 'bab' >"$scratch/second.txt"
mkdir "$scratch/corpus" "$scratch/repeat"
"$cli" corpus build --tokenizer "$scratch/bpe.tsv" \
  --document "$scratch/first.txt" --document "$scratch/second.txt" \
  --output-directory "$scratch/corpus" --shard-token-limit 2
"$cli" corpus build --tokenizer "$scratch/repeat.tsv" \
  --document "$scratch/first.txt" --document "$scratch/second.txt" \
  --output-directory "$scratch/repeat" --shard-token-limit 2
diff -r "$scratch/corpus" "$scratch/repeat"
"$cli" corpus validate "$scratch/corpus"
```

Both builds, the directory comparison, and validation exit 0. The build JSON
reports the same tokenizer fingerprint as inspection. Validation prints
`shards=2 tokens=4 vocab=257 limit=2 bytes=446` followed by `fingerprint_hex=`
and the hex-encoded bytes of the same fingerprint. `bytes` counts the two
complete shard files. [Validation command](CLI3_CORPUS_VALIDATE.md)

Read the token IDs from the validated [D1 shard layout](TOKEN_SHARD_FORMAT.md).
The first shard contains `256 97` for `aba` and the second contains `98 256`
for `bab`. The `ab` across the document boundary stays as byte IDs `97 98`.

```bash
python3 - "$scratch/corpus" <<'PY'
from pathlib import Path
from struct import iter_unpack, unpack_from
import sys

for shard in sorted(Path(sys.argv[1]).glob("shard-*.ets")):
    data = shard.read_bytes()
    header_bytes = unpack_from("<I", data, 12)[0]
    print(*(token for (token,) in iter_unpack("<q", data[header_bytes:-32])))
PY
```

Truncate one shard in a copy. Validation exits 2 with `category=corrupt-data`
on stderr and an empty stdout.

```bash
cp -R "$scratch/corpus" "$scratch/broken"
truncate -s -1 "$scratch/broken/shard-0000000000000000.ets"
status=0
"$cli" corpus validate "$scratch/broken" >"$scratch/broken.stdout" 2>"$scratch/broken.stderr" || status=$?
test "$status" -eq 2 &&
  test ! -s "$scratch/broken.stdout" &&
  grep -F 'category=corrupt-data' "$scratch/broken.stderr"
```

Remove the scratch directory when finished. Use the byte-tokenizer recipe for
the installed fixed-profile model workflow. [Corpus-to-model tutorial](DOC5_CORPUS_TO_MODEL_TUTORIAL.md)
