# CLI3 final acceptance gaps at main `1590fcd9`

The merged CLI supports tokenizer and corpus creation/inspection, checkpoint
inspection, fixed-profile pretrain/evaluate, one-byte CPU-f32 generation, and
installed public-D1 `corpus validate`. PR #202's exact-b983 hosted checks and
matching normal/repeat D1 gate admit that bounded addition. They do not make
the whole CLI3 workstream complete.

| Missing behavior | Existing contract or accepted prerequisite | Next evidence |
|---|---|---|
| Authentic P2/G2 terminal generation, first/second EOS, selected text/IDs, detached I1 output | [private Eshkol terminal contract](g3/G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md); native terminal and logical-I1 source are not integrated on this main | Integrate prerequisites, implement one same-call Eshkol coordinator, then normal/repeat/sanitizer, rollback/fail-stop, detached clone and independent Philox/T1 reference gates |
| Public generation beyond the current one-byte diagnostic tuple | [G3 generation proposal](G3_GENERATION_PROPOSAL.md) and terminal contract; no approved public G2 facade or larger repeated-decode profile | Review public facade/options only after private same-call proof; test public argument, owner, EOS, cache/RNG/output and error boundaries |
| Data-only generation continuation | [G3-R record](g3/G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md) and [restore prerequisites](g3/G3_R_RESTORE_PREREQUISITES_CONTRACT.md); main has codec, typed RNG import, same-image C2/SHA receipt, in-memory pair decode and native idle-word predicate, but no file pair publication or live candidate replay | Authenticated pair to restored model/T1/generator, no-draw manual replay, candidate retirement, atomic publication and three-process save/load/continue parity; later larger history and later-EOS profile |
| Whole CLI3 E3/G3/TR3 end-to-end union | [CLI3 proposal](CLI3_PROPOSAL.md), [G3 proposal](G3_GENERATION_PROPOSAL.md), ROADMAP E3/G3/CLI3 rows | Two fresh deterministic AOT builds, exact closure/export evidence, real cross-command artifact flow, strict argv/diagnostic/publication negatives, normal/poison/sanitizer/LSan where supported, independent result review and exact-head CI |

The next independently implementable source leaf is the [private Eshkol idle
RNG equality admission](g3/G3_R_IDLE_RNG_ESHKOL_LEAF.md). It uses an accepted
native predicate already on main and authentic same-registry G3-T/M3T/T1
checks. It adds no public CLI spelling or inferred upstream API. Its static
checks are source evidence only; it cannot establish replay or continuation.
