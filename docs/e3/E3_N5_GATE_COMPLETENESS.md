# E3 §9 N5 compiled gate successor

Status: **prepared source and host gate; compiled profiles unrun**.

The accepted 8,193-row source proof is unchanged. A second checked D1
materialization drops precisely the first two raw tokens, and the independent
D2 reference confirms that its 8,192 full rows equal the original rows after
one real next/release. The noninstalled fault tuple can evaluate this separate
boundary corpus as an exact six-word metric oracle. The original one-over
scenario still rejects 8,193 at the first stage-2 unsupported error, then
advances and evaluates the same dataset, model, and frame at 8,192. Its four
metric words and two counters must exactly equal the separate oracle; the
`>=8192` mutant must reject that boundary without emitting a metric transcript.

`scripts/test-e3-n5-cap-compiled.sh FRESH_ABSOLUTE_OUTPUT_DIRECTORY` is the
deferred supported gate. It builds fresh normal and exact-overlay mutant
packages in unsanitized and ASan/UBSan forms, checks exact source closure and
private CABI, and runs absent/present gradients in normal, repeat, poisoned,
and sanitized process profiles. LSan is enabled via `detect_leaks=1`; a host
where it cannot run must report that limitation instead of counting a pass.
The gate records checksummed D1 payload sizes, per-process elapsed time and
peak RSS, plus in-process before/after RSS, and enforces the 512 MiB peak cap.
The original 17-mode/model/gradient/RNG/cursor/owner snapshots and stage-2
first-error checks remain active in every one-over invocation.

Only the focused Python source/corpus tests and shell syntax checks have run.
No compiled, poisoned, sanitizer, RSS, or numerical success is claimed here.
