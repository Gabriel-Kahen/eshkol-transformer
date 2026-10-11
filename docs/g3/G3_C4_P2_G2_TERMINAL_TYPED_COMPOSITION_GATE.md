# Private P2/G2 terminal-to-typed composition gate

This source candidate starts from accepted typed owner `8d44e3c5fd95e6a8072badc0c87d828aa8a3a15d`
(tree `136a8d410c510635d6bb09f8005fb409c4c77625`). The terminal
coordinator and detached adapter came through ancestor
`84b46f1dd1cd1d26206751560bd8fe47c82d09ee`. The public contract is
`e55df41474cf9cb4be3ea66988e100f58639ff61`; the adapter blocker is
`dbf414ae3a2b6b693113fb45fd72cc8d7aeb83de`.

`g3c4_p2g2_terminal_typed_private_root.esk` loads the installed G3-G source
once, then the private ownership, protected scope, prefix, terminal,
detached, and typed extensions in dependency order. It is an isolated
registry-owning test root. The bridge takes an authentic live terminal
output, snapshots it, creates an independently rooted logical clone, checks
its bytes against that snapshot, and builds three independent native owners:
CPU-i64 I1 `[G]` IDs, I1 `[1]` length `G`, and I1 `[1]` cache length `2+G`.
It reauthenticates the source after each constructor and reads back the
kind, shape length and elements of all three typed owners before releasing
the logical clone. On a recoverable failure it
releases only new owners; a cleanup failure after ownership mutation exits.
The returned tuple is private native handles, with no public tensor shell or
public dispatch.

## One-use review recipe

1. Pin the candidate commit and tree from the handoff, plus the four source
   references above. Confirm the installed
   `native/g3g_manual_package_root.esk` and its source closure do not load
   the three terminal extensions. Run
   `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.g3c4.test_p2_g2_terminal_source_prerequisites tests.g3c4.test_p2_g2_terminal_close_source tests.g3c4.test_p2_g2_terminal_detached_source tests.g3c4.test_p2_g2_terminal_typed_composition_source`
   and `git diff --check`.
2. In one isolated G3-G archive, compile `g3c4_model_owner.c` with all
   predecessor feature flags from `scripts/test-g3c4-p2g2-typed-result.sh`,
   including `ET_G3C4_P2_G2_TERMINAL_PRIVATE`,
   `ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE`, and
   `ET_G3C4_P2_G2_TYPED_RESULT_PRIVATE`. Compile the private root in that
   same archive and link the exact native dependency closure. Do not route
   the public `diagnostic-c4-p2g2` profile in this gate.
3. Run a genuine same-call P2/G2 compiled fixture for first EOS, second EOS,
   and two non-EOS IDs. Compare raw T1 IDs/text, count/cache, all four RNG
   words, A2 cache, and typed rank/shape/elements with independent oracles.
   Repeat after source output release and with two independent typed tuples.
   Cut preterminal operations, native terminal zero, detached snapshot,
   logical clone/copy, and each typed allocation. Prove exact rollback,
   lease rejection/retry, wrong-owner/alias/extent rejection, no postzero
   recoverable work, and no nested `m3-call`.
4. Rebuild with the three P2/G2 terminal, clone, and typed feature macros
   off. Require unchanged installed G3-G symbols/routes and rejection of
   private calls. Run normal, repeat and sanitizer modes with identical
   successful outputs and empty diagnostics; retain checksummed evidence
   and independent source/tests review before public adapter work.

Only the static source gate in step 1 ran for this candidate. Steps 2–4 are
required integration evidence; no compiled same-aggregate, public G2, or
end-to-end acceptance is claimed here.
