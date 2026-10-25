# Fixed B250 C4 P3 fresh-process fixture

This gate starts with the sealed `c41c6403` trained B250 SAVE receipt. It
requires the outer `SHA256SUMS-RUN` seal, the inner combined `SHA256SUMS`
seal, exact 25,504-byte B250 C2 SHA, and the uninterrupted
`second-eos.run1.stdout` SHA, plus a read-only mount of that C2, reference
output, and source checkout. The driver validates the trained 11-line
reference before compiling one CPU4 archive with C4 history, P3 wire, P3
resume, and the producer boxed bridge enabled only on `g3c4_model_owner.c`.
The wire decoder and terminal resume boxed adapters are separate native units.
The prior source commit/tree and c41 recipe SHA are frozen from its reviewed
`TUPLE.json`; the two receipt seals are supplied after that run completes.

`runtime.esk produce` creates a fresh FILE factory from B250, checks its
trainer/model/T1, hands the same model to the real G2 generator, samples the
first token through the accepted first-frame/T1 path, commits non-EOS G1,
aborts the pending output, and captures the now-idle native P3 history/cache/
RNG and fourteen model planes. The fixed encoder writes a 400-byte C4GRCV1
wire and 4,768-byte values file bound to the exact B250 C2 SHA. The host
verifies both independent SHA layers, fixed history, EOS41 policy, model values
and file hashes.

`runtime.esk resume` is a new OS process. It creates a new FILE factory from
the same B250 C2, admits the persisted pair through the checked issuer,
hands off that factory's model, and invokes the private P3 terminal route.
The runtime checks both actual generated IDs, EOS41 terminal count, cache
length, T1 text, and four RNG words. The host compares that result to the
separately compiled uninterrupted trained-G2 reference and rehashes C2 and
both persisted files after the consumer exits.

The two process modes never share native pointers. The caller passes the
sealed B250 SHA to each as a 32-byte file. `driver.py` writes a full command,
depfile, object, archive, output, exit, and SHA256SUMS receipt. A launch
recipe must pin the actual f31/Clang21 compiler paths before executing the
driver under `heavy.lock`.

Source and host tests establish only fixture construction. Native compilation,
trained producer execution, fresh consumer execution, sanitizer, and installed
package behavior remain **NOT RUN** until their separate receipts pass.
