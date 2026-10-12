# CLI3 exact-tree compiled gate handoff

This source-only harness prepares one compiled D1 and installed CLI3 check for
accepted source commit `adb51279e0d76a3b30f7985dd54c22f68a7999a3`, tree
`f397b14863cad08c1976a8a2581542e7ef094fb2`. The gate harness branch is
separate from that target. It does not modify the target source bundle.

The prepared evidence directory contains `source.bundle`, `pins.tsv`,
`inside.sh`, `launch.sh`, and `SHA256SUMS-PREPARED`. `launch.sh
--verify-prepared` authenticates those files and the referenced frozen fe9
dependency directory, including every entry in its parent manifest, without
invoking Docker. The immutable f31 image ID and
every dependency hash are in `pins.tsv`. The dependency directory is shared
read-only with another prepared gate; it must remain present and unchanged.

## Root prelaunch admission

Wait until PR #201 run `38233893150` is complete and the sole heavy build slot
is free. Recheck the prepared manifest, pinned dependency files, host resources,
and supported Docker image. Create exactly one sibling
`<bundle>.admission.tsv` containing these tab-separated records:

```text
pr201_run	38233893150
pr201_status	completed
sole_heavy_slot	confirmed
```

Record the independent basis for these assertions in the orchestration
handoff. Run `<bundle>/launch.sh` once. It refuses an existing
`<bundle>.result` directory. The host runner also requires at least 14 GiB
available memory, 20 GiB disk on a non-tmpfs result filesystem, two CPUs,
and an exclusive host flock. Docker uses no network, two CPUs, 12 GiB memory,
a read-only exact source checkout and dependencies, and a six-hour outer
timeout. The image identity is verified immediately before launch.

## Gate and retained result

The guest verifies source/tree/cleanliness, lock and fe9 toolchain pins before
and after the work. It runs the standalone D1 corpus validator gate, the
production CLI3 package and installed command gate, an independent archive,
symbol and app-depfile audit, then the existing cross-command chain. The
installed gate covers positive, negative, and exit behavior for `corpus
validate` and reruns the old CLI suite. Each step retains stdout, stderr,
time, and exit status. The host retains clone, Docker, source identity,
admission, and failure-stage evidence; its exit trap writes `overall.status`
and `SHA256SUMS-RUN` even after a failed child or timeout. The sealed
`host-command-exit.status` retains the original Docker or timeout status
before aggregate failure classification. A missing run seal
is itself a failed handoff.

Preparation and source review do not constitute compiled or installed runtime
validation. The prepared bundle has no admission record and has not been
launched. Root must review the sealed result before integration or full CLI3
acceptance.
