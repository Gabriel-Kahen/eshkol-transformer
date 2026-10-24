# Final B250 fresh FILE factory witness

This separate fixture extends the reviewed `c41c6403` trained producer only
to print its live final-B snapshot before B250 SAVE. The producer emits all
14 raw `f32` parameter planes in canonical `g3c4-model-shapes` order with
lengths `64,64,64,64,128,128,16,16,16,16,4096,16,16,64`. The external
driver must seal the 4,768 bytes and per-plane SHA-256 values from that stdout,
alongside the successful compiled producer exit and the distinct B250 C2
25,504-byte SHA. A second fresh process mounts that exact C2 read-only,
rehashes it before/after, uses the existing FILE factory with fresh T2/D2,
snapshots the restored trainer, and emits its 14 planes. The strict reader
compares every byte to the independent producer witness. `inspect` and
`handoff` are separate process modes so the latter may run only after the
first fresh witness matches.

`runtime.esk` checks factory readiness, retired C2 stage/image, trainer and
dataset identities, B250 count/cursor, 14/28 plane extents and fresh T1
fingerprint. `handoff` uses the existing factory generator handoff to adopt
the same restored model and closes the resulting G2 generator. No persisted
RNG/history or P3→G2 continuation is imported by this fixture. The source and
parser tests do not constitute compiled evidence.

`driver.py` accepts a sealed prior c41 output directory, pinned B250 C2 and
native-archive SHA values, and the existing compiler runner/C++ linker. It
reuses that archive, compiles the modified producer and fresh caller, reruns
only the second-EOS producer, verifies B125 and exact B250 C2 identities,
checks the standard 11-line trained result after removing only witness rows,
then seals the producer stdout/planes. It runs fresh `inspect`, compares all
planes, and only then starts fresh `handoff` and compares those planes again.
The original B250 file and archive must be mounted read-only; the driver
rehashes both before and after each fresh process. An immutable launch recipe
must verify the prior receipt seal and source/toolchain pins before this driver
is admitted for execution.
