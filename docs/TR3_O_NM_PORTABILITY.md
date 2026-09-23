# TR3-O `nm` portability evidence

This note records the test-only portability repair derived from PR #124's
supported native-optimizer job. It changes no O2 runtime, public API, ABI,
provider, ownership, or transaction semantics.

## Provenance

- Frozen PR head: `79d71086293ccd51bd569785a19c7409d1e070a9`.
- Supported job: `107370666216` in workflow run `35916891673`.
- Raw job log: `/tmp/pr124-native-optimizer-107370666216.log` on the diagnosing
  host, SHA-256
  `fd357ceaed0e45fce67c23049118d85b77babdf559dfb67a9138dec63d335cb7`.
- Supported reproduction: Ubuntu 22.04, GNU nm 2.38, Clang/LLVM 21.1.8.

The job completed the full O2 gate, then failed the TR3-O package test while
enumerating global defined symbols. On Ubuntu 22.04, `nm -gU --format=posix`
exits 1 with:

```text
nm: invalid argument to -U/--unicode: --format=posix
```

Newer GNU nm assigns `-U` to `--defined-only`, which is why the unsupported
local lane did not reproduce the failure. The repaired test uses explicit,
portable global-defined predicate:

```text
nm -g --defined-only --format=posix OBJECT
```

The already-portable `nm -gu` undefined-symbol query remains byte-for-byte
unchanged. The assertions are unchanged: enabling the private bridge must add
exactly the three frozen global bridge definitions and must newly reference all
three frozen core symbols. No symbol assertion is dropped or broadened.
