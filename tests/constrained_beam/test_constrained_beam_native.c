#include "eshkol_transformer/constrained_beam_abi.h"

#include <assert.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { EOS, A, B, C, X, AB, AC, V };

static et_cd_vocab_index *build_index(size_t choices_count, uint32_t vocab,
                                      uint32_t eos, int full_vocab) {
  const uint8_t *choices[] = {(const uint8_t *)"ab", (const uint8_t *)"ac"};
  const size_t lengths[] = {2, 2};
  const uint8_t *tokens[] = {NULL,
                             (const uint8_t *)"a",
                             (const uint8_t *)"b",
                             (const uint8_t *)"c",
                             (const uint8_t *)"x",
                             (const uint8_t *)"ab",
                             (const uint8_t *)"ac"};
  size_t sizes[] = {0, 1, 1, 1, 1, 2, 2};
  if (!full_vocab) {
    sizes[B] = sizes[C] = sizes[AB] = sizes[AC] = 0u;
  }
  et_cd_dfa *dfa = NULL;
  et_cd_vocab_index *index = NULL;
  assert(et_cd_dfa_compile_literal_choices(choices, lengths, choices_count,
                                           &dfa) == ET_CD_OK);
  assert(et_cd_vocab_index_build(dfa, tokens, sizes, vocab, eos, &index) ==
         ET_CD_OK);
  et_cd_dfa_destroy(dfa);
  return index;
}

static et_beam_config_v1 config(uint32_t beams, uint32_t max_steps) {
  et_beam_config_v1 cfg = {0};
  cfg.struct_size = sizeof(cfg);
  cfg.num_beams = beams;
  cfg.vocab_size = V;
  cfg.eos_token_id = EOS;
  cfg.max_new_tokens = max_steps;
  cfg.length_penalty = 1.0;
  cfg.repetition_penalty = 1.0;
  return cfg;
}

static void expect_unchanged(et_beam_state_v1 *beam,
                             const et_cd_vocab_index *index,
                             const float *logits, size_t rows, size_t vocab,
                             et_beam_status expected) {
  et_beam_state_v1 before;
  memcpy(&before, beam, sizeof(before));
  assert(et_constrained_beam_step_v1(beam, index, logits, rows, vocab) ==
         expected);
  assert(memcmp(beam, &before, sizeof(before)) == 0);
}

static void branches_reorder_and_eos(void) {
  et_cd_vocab_index *index = build_index(2, V, EOS, 1);
  const uint32_t prompt[] = {X, EOS};
  const et_beam_config_v1 cfg = config(2, 3);
  et_beam_state_v1 beam;
  assert(et_beam_init_v1(&beam, &cfg, prompt, 2) == ET_BEAM_OK);

  /* Whole-token "ab" outranks "a", although the raw maxima are forbidden. */
  const float first[] = {100, 3, 0, 0, 90, 4, 0};
  assert(et_constrained_beam_step_v1(&beam, index, first, 1, V) == ET_BEAM_OK);
  assert(et_beam_active_count_v1(&beam) == 2);
  assert(beam.active_tokens[0][2] == AB && beam.active_tokens[1][2] == A);
  const double first_den = log(1.0 + exp(-1.0) + exp(-4.0));
  assert(fabs(beam.active_log_probs[0] + first_den) < 1e-14);
  assert(fabs(beam.active_log_probs[1] + 1.0 + first_den) < 1e-14);
  assert(et_beam_finished_count_v1(&beam) == 0);

  /* The first beam can finish. The second must choose B or C and supplies
   * both next active beams, so old and new row orders differ. */
  float second[2 * V] = {5, 0, 0, 0, 90, 0, 0, 100, 0, 1, 4, 90, 0, 0};
  second[V + X] = NAN;
  expect_unchanged(&beam, index, second, 2, V, ET_BEAM_ERR_NONFINITE);
  second[V + X] = 90.0f;
  assert(et_constrained_beam_step_v1(&beam, index, second, 2, V) == ET_BEAM_OK);
  assert(et_beam_finished_count_v1(&beam) == 1);
  assert(et_beam_active_count_v1(&beam) == 2);
  assert(beam.active_tokens[0][2] == A && beam.active_tokens[0][3] == C);
  assert(beam.active_tokens[1][2] == A && beam.active_tokens[1][3] == B);
  const double second_best = -1.0 - first_den - log(1.0 + exp(-3.0));
  assert(fabs(beam.active_log_probs[0] - second_best) < 1e-14);

  const float last[2 * V] = {-100, 0, 0, 0, 99, 0, 0, 5, 0, 0, 0, 90, 0, 0};
  assert(et_constrained_beam_step_v1(&beam, index, last, 2, V) == ET_BEAM_OK);
  assert(et_beam_is_done_v1(&beam) == 1 &&
         et_beam_finished_count_v1(&beam) == 3);
  et_beam_hypothesis_v1 out[2];
  size_t count = 0;
  assert(et_beam_finalize_v1(&beam, out, 2, &count) == ET_BEAM_OK);
  assert(count == 2);
  assert(out[0].finished && out[0].generated_len == 2);
  assert(out[0].tokens[2] == AB && out[0].tokens[3] == EOS);
  assert(fabs(out[0].score + first_den / (7.0 / 6.0)) < 1e-14);
  assert(out[1].finished && out[1].generated_len == 3);
  assert(out[1].tokens[2] == A && out[1].tokens[3] == C &&
         out[1].tokens[4] == EOS);
  assert(fabs(out[1].score - second_best / (8.0 / 6.0)) < 1e-14);
  expect_unchanged(&beam, index, first, 1, V, ET_BEAM_ERR_STATE);
  et_cd_vocab_index_destroy(index);
}

static void rejects_invalid_inputs_and_histories(void) {
  et_cd_vocab_index *index = build_index(1, V, EOS, 1);
  et_cd_vocab_index *short_index = build_index(1, V - 1, EOS, 1);
  et_cd_vocab_index *other_eos = build_index(1, V, X, 1);
  const et_beam_config_v1 cfg = config(2, 3);
  const uint32_t prompt[] = {X};
  et_beam_state_v1 beam = {0};
  float raw[V] = {0};
  expect_unchanged(&beam, index, raw, 1, V, ET_BEAM_ERR_STATE);
  assert(et_beam_init_v1(&beam, &cfg, prompt, 1) == ET_BEAM_OK);
  expect_unchanged(&beam, NULL, raw, 1, V, ET_BEAM_ERR_INVALID_ARG);
  expect_unchanged(&beam, index, NULL, 1, V, ET_BEAM_ERR_INVALID_ARG);
  expect_unchanged(&beam, index, raw, SIZE_MAX, V, ET_BEAM_ERR_INVALID_ARG);
  expect_unchanged(&beam, index, raw, 1, SIZE_MAX, ET_BEAM_ERR_INVALID_ARG);
  expect_unchanged(&beam, index, raw, 0, V, ET_BEAM_ERR_INVALID_ARG);
  expect_unchanged(&beam, short_index, raw, 1, V, ET_BEAM_ERR_INVALID_ARG);
  expect_unchanged(&beam, other_eos, raw, 1, V, ET_BEAM_ERR_INVALID_ARG);
  expect_unchanged(&beam, index, (const float *)beam.scratch_logits, 1, V,
                   ET_BEAM_ERR_ALIASING);
  raw[X] = -INFINITY;
  expect_unchanged(&beam, index, raw, 1, V, ET_BEAM_ERR_NONFINITE);
  raw[X] = INFINITY;
  expect_unchanged(&beam, index, raw, 1, V, ET_BEAM_ERR_NONFINITE);
  raw[X] = 0.0f;
  assert(fesetround(FE_DOWNWARD) == 0);
  expect_unchanged(&beam, index, raw, 1, V, ET_BEAM_ERR_FP_ENV);
  assert(fesetround(FE_TONEAREST) == 0);

  /* Valid BEAM5 histories can be inadmissible under the CD5 index. */
  const double forbidden_active[] = {-INFINITY, 0,         -INFINITY, -INFINITY,
                                     0,         -INFINITY, -INFINITY};
  assert(et_beam_step_v1(&beam, forbidden_active, 1) == ET_BEAM_OK);
  assert(beam.active_tokens[0][1] == A && beam.active_tokens[1][1] == X);
  const float rows[2 * V] = {0};
  expect_unchanged(&beam, index, rows, 2, V, ET_BEAM_ERR_STATE);

  assert(et_beam_init_v1(&beam, &cfg, prompt, 1) == ET_BEAM_OK);
  const double forbidden_finish[] = {1,         0,         -INFINITY, -INFINITY,
                                     -INFINITY, -INFINITY, -INFINITY};
  assert(et_beam_step_v1(&beam, forbidden_finish, 1) == ET_BEAM_OK);
  assert(beam.finished_count == 1 && beam.active_tokens[0][1] == A);
  expect_unchanged(&beam, index, raw, 1, V, ET_BEAM_ERR_STATE);
  et_cd_vocab_index_destroy(other_eos);
  et_cd_vocab_index_destroy(short_index);
  et_cd_vocab_index_destroy(index);
}

static void empty_rows_are_atomic(void) {
  const uint32_t prompt[] = {X};
  const float raw[V] = {0, 3, 0, 0, 90, 4, 0};
  et_beam_state_v1 beam;
  et_beam_config_v1 cfg = config(1, 2);

  et_cd_vocab_index *index = build_index(1, V, EOS, 0);
  assert(et_beam_init_v1(&beam, &cfg, prompt, 1) == ET_BEAM_OK);
  assert(et_constrained_beam_step_v1(&beam, index, raw, 1, V) == ET_BEAM_OK);
  assert(beam.active_tokens[0][1] == A);
  expect_unchanged(&beam, index, raw, 1, V, ET_BEAM_ERR_NO_VALID_TOKENS);
  et_cd_vocab_index_destroy(index);

  /* The DFA admits EOS after AB but BEAM5's minimum length masks it.
   * BEAM5 errors after writing its scratch buffers in the staged state. */
  index = build_index(1, V, EOS, 1);
  cfg.min_new_tokens = 2;
  assert(et_beam_init_v1(&beam, &cfg, prompt, 1) == ET_BEAM_OK);
  assert(et_constrained_beam_step_v1(&beam, index, raw, 1, V) == ET_BEAM_OK);
  assert(beam.active_tokens[0][1] == AB);
  expect_unchanged(&beam, index, raw, 1, V, ET_BEAM_ERR_NO_VALID_TOKENS);
  et_cd_vocab_index_destroy(index);
}

static void budget_can_leave_incomplete_prefix(void) {
  et_cd_vocab_index *index = build_index(1, V, EOS, 1);
  const et_beam_config_v1 cfg = config(1, 1);
  const uint32_t prompt[] = {X};
  const float raw[V] = {100, 5, 0, 0, 90, 1, 0};
  et_beam_state_v1 beam;
  assert(et_beam_init_v1(&beam, &cfg, prompt, 1) == ET_BEAM_OK);
  assert(et_constrained_beam_step_v1(&beam, index, raw, 1, V) == ET_BEAM_OK);
  expect_unchanged(&beam, index, raw, 1, V, ET_BEAM_ERR_STATE);
  et_beam_hypothesis_v1 out;
  size_t count = 0;
  assert(et_beam_finalize_v1(&beam, &out, 1, &count) == ET_BEAM_OK);
  assert(count == 1 && out.finished == 0 && out.tokens[1] == A);
  et_cd_vocab_index_destroy(index);
}

static void prompt_remains_in_beam_processor_history(void) {
  et_cd_vocab_index *index = build_index(1, V, EOS, 1);
  const uint32_t prompt[] = {A};
  const float raw[V] = {100, 4, 0, 0, 90, 3, 0};
  et_beam_config_v1 cfg = config(2, 2);
  cfg.repetition_penalty = 2.0;
  et_beam_state_v1 beam;
  assert(et_beam_init_v1(&beam, &cfg, prompt, 1) == ET_BEAM_OK);

  /* DFA starts fresh, while BEAM5 reduces the repeated A from 4 to 2. */
  assert(et_constrained_beam_step_v1(&beam, index, raw, 1, V) == ET_BEAM_OK);
  assert(beam.active_tokens[0][1] == AB && beam.active_tokens[1][1] == A);
  assert(fabs(beam.active_log_probs[0] + log(1.0 + exp(-1.0))) < 1e-14);
  et_cd_vocab_index_destroy(index);
}

static void maximum_geometry(void) {
  const uint8_t *choices[] = {(const uint8_t *)"a"};
  const size_t lengths[] = {1};
  const uint8_t *tokens[ET_BEAM_MAX_VOCAB] = {0};
  size_t sizes[ET_BEAM_MAX_VOCAB] = {0};
  for (size_t t = ET_BEAM_MAX_VOCAB - ET_BEAM_MAX_BEAMS; t < ET_BEAM_MAX_VOCAB;
       ++t) {
    tokens[t] = choices[0];
    sizes[t] = lengths[0];
  }
  et_cd_dfa *dfa = NULL;
  et_cd_vocab_index *index = NULL;
  assert(et_cd_dfa_compile_literal_choices(choices, lengths, 1, &dfa) ==
         ET_CD_OK);
  assert(et_cd_vocab_index_build(dfa, tokens, sizes, ET_BEAM_MAX_VOCAB, EOS,
                                 &index) == ET_CD_OK);
  et_cd_dfa_destroy(dfa);

  const uint32_t prompt[ET_BEAM_MAX_SEQ_LEN - 2u] = {0};
  const float raw[ET_BEAM_MAX_BEAMS * ET_BEAM_MAX_VOCAB] = {0};
  et_beam_config_v1 cfg = config(ET_BEAM_MAX_BEAMS, 2);
  cfg.vocab_size = ET_BEAM_MAX_VOCAB;
  et_beam_state_v1 beam;
  assert(et_beam_init_v1(&beam, &cfg, prompt, ET_BEAM_MAX_SEQ_LEN - 2u) ==
         ET_BEAM_OK);
  assert(et_constrained_beam_step_v1(&beam, index, raw, 1, ET_BEAM_MAX_VOCAB) ==
         ET_BEAM_OK);
  assert(et_beam_active_count_v1(&beam) == ET_BEAM_MAX_BEAMS);
  assert(et_constrained_beam_step_v1(&beam, index, raw, ET_BEAM_MAX_BEAMS,
                                     ET_BEAM_MAX_VOCAB) == ET_BEAM_OK);
  assert(et_beam_is_done_v1(&beam) == 1);
  assert(et_beam_finished_count_v1(&beam) == ET_BEAM_MAX_BEAMS);
  for (size_t b = 0; b < ET_BEAM_MAX_BEAMS; ++b) {
    const et_beam_hypothesis_v1 *hyp = &beam.finished_pool[b];
    assert(hyp->seq_len == ET_BEAM_MAX_SEQ_LEN);
    assert(hyp->tokens[hyp->seq_len - 2u] ==
           ET_BEAM_MAX_VOCAB - ET_BEAM_MAX_BEAMS + b);
    assert(hyp->tokens[hyp->seq_len - 1u] == EOS);
  }
  et_cd_vocab_index_destroy(index);
}

int main(void) {
  branches_reorder_and_eos();
  rejects_invalid_inputs_and_histories();
  empty_rows_are_atomic();
  budget_can_leave_incomplete_prefix();
  prompt_remains_in_beam_processor_history();
  maximum_geometry();
  puts("Constrained beam step: PASS");
  return 0;
}
