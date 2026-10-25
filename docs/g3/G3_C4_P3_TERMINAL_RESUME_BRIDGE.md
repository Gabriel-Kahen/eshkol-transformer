# Private P3 to G2 terminal bridge

The checked FILE issuer supplies an opaque ticket to
`g3c4-p2g2-terminal-resume-private!(generator, input, ticket)`. This is
a source-private alternate first stage of the existing P2/G2 terminal
coordinator. The public two-argument route still passes no resume ticket.

The separately loaded `cpu4-c4-p3-resume-issuer-internal` must accept
`(action, generator, input, ticket, operation)`. Its `'admit` action runs
before input preflight. Its `'verify` action runs after a genuine native call,
pending output and live authority exist, immediately before P3. Both calls
must return the same descriptor by `eq?`: a three-slot vector containing a
24-byte little-endian i64 ID span `[3,197,3]`, a 32-byte little-endian i64
RNG span `[1,41,0,0]`, and the first selected ID `3`. The bridge checks
that exact shape and fixed bytes. The issuer is responsible for FILE/C2,
model, T1, prompt and ticket provenance; the descriptor is not FILE authority.

Native P3 reconstructs the cache/history and commits the first no-draw
prefix while the output is pending. The bridge writes the saved G1 into the
original first staging byte, decodes through the genuine T1 into the original
raw carrier, and advances the protected Eshkol prefix phase. It then follows
the existing second-sample, terminal commit and publication tail. A failure
before terminal commit is handled by the existing protected abort. This
source-only bridge requires aggregate compilation and fresh-process
functional admission with the issuer before continuation can be claimed.
