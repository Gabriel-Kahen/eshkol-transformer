/* Development-only CPU composition witness; neither production ABI changes. */
#include "eshkol_transformer/beam_search_abi.h"
#include "eshkol_transformer/constrained_decode_abi.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { V = 7, EOS = 0, MAX_ROWS = 2 };

static et_beam_config_v1 config(uint32_t beams, uint32_t vocab) {
  et_beam_config_v1 cfg = {0};
  cfg.struct_size = sizeof(cfg);
  cfg.num_beams = beams;
  cfg.vocab_size = vocab;
  cfg.eos_token_id = EOS;
  cfg.max_new_tokens = 3;
  cfg.repetition_penalty = 1.0;
  return cfg;
}

/* Stage every row before asking the beam engine to commit a step. On a mask
 * failure, neither caller-owned state has been changed. */
static et_cd_status stage_masked_rows(const et_cd_vocab_index *index,
                                      const et_beam_state_v1 *beam,
                                      const uint16_t states[MAX_ROWS],
                                      const float logits[MAX_ROWS][V],
                                      double masked[MAX_ROWS][V]) {
  float row_mask[V];
  for (uint32_t b = 0; b < et_beam_active_count_v1(beam); ++b) {
    const et_cd_status st = et_cd_apply_logit_mask_v1(
        index, states[b], logits[b], row_mask);
    if (st != ET_CD_OK)
      return st;
    for (uint32_t t = 0; t < V; ++t) {
      masked[b][t] = (double)row_mask[t];
      assert(isfinite(masked[b][t]) || masked[b][t] == -INFINITY);
    }
  }
  return ET_CD_OK;
}

/* Recover selected parents by exact prefix; the beam ABI exposes sequences,
 * not parent indices. Distinct active histories make this unambiguous. */
static void selected_step(const et_cd_dfa *dfa, const et_cd_vocab_index *index,
                          et_beam_state_v1 *beam, uint16_t states[MAX_ROWS],
                          const float logits[MAX_ROWS][V]) {
  const uint32_t rows = et_beam_active_count_v1(beam);
  assert(rows > 0 && rows <= MAX_ROWS);
  uint32_t old_len[MAX_ROWS];
  uint32_t old_tokens[MAX_ROWS][ET_BEAM_MAX_SEQ_LEN];
  uint16_t old_states[MAX_ROWS];
  double masked[MAX_ROWS][V];
  const uint32_t prior_finished = et_beam_finished_count_v1(beam);

  for (uint32_t b = 0; b < rows; ++b) {
    size_t len = 0;
    double lp = 0;
    assert(et_beam_get_active_beam_v1(beam, b, old_tokens[b],
                                     ET_BEAM_MAX_SEQ_LEN, &len, &lp) == ET_BEAM_OK);
    old_len[b] = (uint32_t)len;
    old_states[b] = states[b];
  }
  assert(stage_masked_rows(index, beam, states, logits, masked) == ET_CD_OK);
  assert(et_beam_step_v1(beam, &masked[0][0], rows) == ET_BEAM_OK);

  for (uint32_t b = 0; b < et_beam_active_count_v1(beam); ++b) {
    uint32_t tokens[ET_BEAM_MAX_SEQ_LEN];
    size_t len = 0;
    double lp = 0;
    assert(et_beam_get_active_beam_v1(beam, b, tokens,
                                     ET_BEAM_MAX_SEQ_LEN, &len, &lp) == ET_BEAM_OK);
    uint32_t matches = 0;
    for (uint32_t p = 0; p < rows; ++p) {
      if (len != (size_t)old_len[p] + 1u ||
          memcmp(tokens, old_tokens[p], old_len[p] * sizeof(uint32_t)) != 0)
        continue;
      uint32_t finished = UINT32_MAX;
      uint16_t next = ET_CD_DEAD_STATE;
      assert(et_cd_advance_token_v1(index, old_states[p], tokens[len - 1u],
                                    &next, &finished) == ET_CD_OK);
      assert(finished == 0 && next != ET_CD_DEAD_STATE);
      states[b] = next;
      ++matches;
    }
    assert(matches == 1);
  }

  /* Newly completed hypotheses must have selected EOS from an accepting
   * parent, with the DFA reporting its terminal sentinel. */
  for (uint32_t i = 0; i < et_beam_finished_count_v1(beam); ++i) {
    const et_beam_hypothesis_v1 *h = &beam->finished_pool[i];
    if (h->generated_len != beam->steps_done)
      continue;
    uint32_t matches = 0;
    for (uint32_t p = 0; p < rows; ++p) {
      if (h->seq_len != old_len[p] + 1u ||
          memcmp(h->tokens, old_tokens[p], old_len[p] * sizeof(uint32_t)) != 0)
        continue;
      assert(h->tokens[h->seq_len - 1u] == EOS);
      assert(et_cd_dfa_is_accepting(dfa, old_states[p]) == 1);
      assert(et_cd_vocab_index_is_token_allowed(index, old_states[p], EOS) == 1);
      uint16_t next = 0;
      uint32_t finished = 0;
      assert(et_cd_advance_token_v1(index, old_states[p], EOS,
                                    &next, &finished) == ET_CD_OK);
      assert(finished == 1 && next == ET_CD_DEAD_STATE);
      ++matches;
    }
    assert(matches == 1);
  }
  assert(et_beam_finished_count_v1(beam) >= prior_finished);
}

static void branching_witness(void) {
  const uint8_t ab[] = {'a', 'b'}, ac[] = {'a', 'c'};
  const uint8_t *choices[] = {ab, ac};
  const size_t lengths[] = {sizeof(ab), sizeof(ac)};
  et_cd_dfa *dfa = NULL;
  assert(et_cd_dfa_compile_literal_choices(choices, lengths, 2, &dfa) == ET_CD_OK);
  const uint8_t a[] = {'a'}, b[] = {'b'}, c[] = {'c'}, x[] = {'x'};
  const uint8_t *tokens[V] = {NULL, a, b, c, x, ab, ac};
  const size_t sizes[V] = {0, 1, 1, 1, 1, 2, 2};
  et_cd_vocab_index *index = NULL;
  assert(et_cd_vocab_index_build(dfa, tokens, sizes, V, EOS, &index) == ET_CD_OK);
  et_cd_vocab_index_info info;
  assert(et_cd_vocab_index_info_get(index, &info) == ET_CD_OK);
  const uint32_t prompt[] = {4};
  et_beam_state_v1 beam;
  et_beam_config_v1 cfg = config(2, V);
  assert(et_beam_init_v1(&beam, &cfg, prompt, 1) == ET_BEAM_OK);
  uint16_t states[MAX_ROWS] = {info.start_state, ET_CD_DEAD_STATE};

  float rejected[V];
  for (size_t i = 0; i < V; ++i) rejected[i] = 42.0f;
  const float nonfinite[V] = {0, NAN, 0, 0, 0, 0, 0};
  assert(et_cd_apply_logit_mask_v1(index, states[0], nonfinite, rejected) ==
         ET_CD_ERR_NONFINITE);
  for (size_t i = 0; i < V; ++i) assert(rejected[i] == 42.0f);

  /* Forbidden x and early EOS have the highest supplied logits. */
  const float step0[MAX_ROWS][V] = {{100, 4, 0, 0, 90, 3, 0}, {0}};
  selected_step(dfa, index, &beam, states, step0);
  assert(beam.active_count == 2 && beam.active_tokens[0][1] == 1 &&
         beam.active_tokens[1][1] == 5);
  assert(!et_cd_dfa_is_accepting(dfa, states[0]));
  assert(et_cd_dfa_is_accepting(dfa, states[1]));

  const float step1[MAX_ROWS][V] = {{100, 0, 0, 4, 90, 0, 0},
                                     {5, 0, 0, 0, 90, 0, 0}};
  selected_step(dfa, index, &beam, states, step1);
  assert(beam.finished_count >= 1 && beam.active_count >= 1);
  for (uint32_t i = 0; i < beam.active_count; ++i)
    assert(et_cd_dfa_is_accepting(dfa, states[i]));

  const float step2[MAX_ROWS][V] = {{5, 0, 0, 0, 90, 0, 0},
                                     {5, 0, 0, 0, 90, 0, 0}};
  selected_step(dfa, index, &beam, states, step2);
  assert(et_beam_is_done_v1(&beam) == 1 && beam.finished_count >= 2);
  et_beam_hypothesis_v1 out[2];
  size_t count = 0;
  assert(et_beam_finalize_v1(&beam, out, 2, &count) == ET_BEAM_OK);
  assert(count == 2);
  for (size_t i = 0; i < count; ++i)
    assert(out[i].finished && out[i].tokens[out[i].seq_len - 1u] == EOS);
  et_cd_vocab_index_destroy(index);
  et_cd_dfa_destroy(dfa);
}

static void empty_admissibility_is_atomic(void) {
  const uint8_t ab[] = {'a', 'b'}, a[] = {'a'}, x[] = {'x'};
  const uint8_t *choices[] = {ab};
  const size_t lengths[] = {2};
  et_cd_dfa *dfa = NULL;
  assert(et_cd_dfa_compile_literal_choices(choices, lengths, 1, &dfa) == ET_CD_OK);
  const uint8_t *tokens[V] = {NULL, a, x, NULL, NULL, NULL, NULL};
  const size_t sizes[V] = {0, 1, 1, 0, 0, 0, 0};
  et_cd_vocab_index *index = NULL;
  assert(et_cd_vocab_index_build(dfa, tokens, sizes, V, EOS, &index) == ET_CD_OK);
  et_cd_vocab_index_info info;
  assert(et_cd_vocab_index_info_get(index, &info) == ET_CD_OK);
  et_beam_state_v1 beam, before;
  et_beam_config_v1 cfg = config(1, V);
  const uint32_t prompt[] = {2};
  assert(et_beam_init_v1(&beam, &cfg, prompt, 1) == ET_BEAM_OK);
  uint16_t states[MAX_ROWS] = {info.start_state, ET_CD_DEAD_STATE};
  const float first[MAX_ROWS][V] = {{100, 2, 90, 0, 0, 0, 0}, {0}};
  selected_step(dfa, index, &beam, states, first);
  assert(beam.active_count == 1 && beam.active_tokens[0][1] == 1);
  memcpy(&before, &beam, sizeof(beam));
  const uint16_t saved_state = states[0];
  double output[MAX_ROWS][V];
  for (size_t b = 0; b < MAX_ROWS; ++b)
    for (size_t t = 0; t < V; ++t) output[b][t] = 42.0;
  const float second[MAX_ROWS][V] = {{100, 0, 90, 0, 0, 0, 0}, {0}};
  assert(stage_masked_rows(index, &beam, states, second, output) ==
         ET_CD_ERR_NO_VALID_TOKENS);
  assert(memcmp(&beam, &before, sizeof(beam)) == 0 && states[0] == saved_state);
  for (size_t b = 0; b < MAX_ROWS; ++b)
    for (size_t t = 0; t < V; ++t) assert(output[b][t] == 42.0);
  et_cd_vocab_index_destroy(index);
  et_cd_dfa_destroy(dfa);
}

int main(void) {
  branching_witness();
  empty_admissibility_is_atomic();
  puts("BEAM_CD composition witness: PASS");
  return 0;
}
