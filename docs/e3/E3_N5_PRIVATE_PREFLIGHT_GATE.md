# E3 N5 private 8,192/8,193 preflight gate candidate

This is source for a **pending compiled-private** §9 N5 gate. It does not
establish a runtime result. The gate uses the existing noninstalled retention
aggregate, the real `e3-evaluate-into!`, M3 seed-1729 model, raw T1 tokenizer,
checksummed D1 resources, packed D2 datasets, E3 frame, and four I2 scalar
destinations. It adds no public evaluator API, native hook, or manufactured
logit/CE value.

`tests/e3_horizon/generate.py` creates 8,192 and 8,193 packed `[1,2]` rows
from deterministic D1 shards. The gate generates each corpus twice and
compares bytes. An independent D2 reference records the manifest digest,
exact entry cursor and row count. The retained payload is 133,680 and 133,696
bytes respectively, including the D1 manifest. The reference uses the private
test driver's actual D2 cap of 34 batch bytes; this matters to cursor identity.

The added `n5` phase of `tests/e3_private/retention_driver.esk` opens both
datasets under one stable private model/frame/destination authority. It seeds a
nonzero gradient and mixed 17-bit mode image. The first evaluator call must
complete exactly 8,192 batches/16,384 tokens with the exact mask-weight word
and original cursor. The second call uses the 8,193-row dataset. It must raise
the E1 `unsupported`, `e3-evaluate-into!`, stage-2 error with fixed details;
the private first-error index must be 29. The driver then checks the exact
8,193 entry cursor, all four I2 words and two counters, modes, every parameter
and gradient word/metadata, initializer RNG words, and the 29 native owner,
borrow, graph, D2 and FD counters against the pre-call snapshot. This catches
publication or traversal at the 8,193rd row without an extra test hook.

`scripts/test-e3-n5-private-preflight.sh` is the sole proposed compiled gate.
It builds and inventories the canonical private object, two byte-identical
clean noninstalled retention objects/archives/depfiles, and a sanitized
object. It compiles the dedicated strict optimized AOT runner twice plus an
ASan/UBSan/LSan runner. Normal A/B, poisoned and sanitized invocations compare
the runtime's observed cursor bytes against the independent reference and
require identical success stdout and empty stderr. It retains D1/D2 payload,
compile and runtime logs, GNU time elapsed/peak RSS records, and any failed
build's staging evidence in a fresh output directory. The default per-run
timeout is 7,200 seconds; RSS is measured, with no inferred bound before the
compiled runs. No compiler or runtime result is claimed from source checks.

The next gate is independent whole-tree review of this current-main source
composition, followed by one supported compiled run. PR #206's exact-head CI
passed before its merge at `c60779bb`; that CI did not run this N5 gate. If the
retention aggregate's exact symbols, source closure,
sanitizer imports, or AOT link differ, preserve the failed evidence and repair
the actual tuple before any N5 acceptance claim. Empty/singleton/EOS/one-token
and full N5 resource/ownership horizons remain separate open rows.
