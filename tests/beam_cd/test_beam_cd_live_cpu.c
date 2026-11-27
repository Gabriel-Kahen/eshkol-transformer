#include "eshkol_transformer/ac4_checkpointing_abi.h"
#include "eshkol_transformer/constrained_beam_abi.h"
#include "eshkol_transformer/t2_bpe_trainer_abi.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

enum {
  ROWS = 2,
  CONTEXT = 3,
  HIDDEN = 4,
  VOCAB = 260,
  STREAM = ROWS * CONTEXT * HIDDEN,
  PARAMETERS = 2 * HIDDEN + 7 * HIDDEN * HIDDEN,
  WORKSPACE_FLOATS = 12 * STREAM
};

typedef struct {
  uint32_t tokens[CONTEXT];
  size_t length;
} prefix;

typedef struct {
  float embedding[VOCAB][HIDDEN];
  float head[VOCAB][HIDDEN];
  float bias[VOCAB];
  float parameters[PARAMETERS];
} cpu_model;

static const et_ac4_config_v1 model_config = {
    .shape = {1, ROWS, CONTEXT, 2, 2, 2},
    .mode = ET_AC4_FORWARD_ONLY,
    .policy = ET_AC4_POLICY_NONE,
    .epsilon = 0x1p-12f};

static void initialize_model(cpu_model *model, uint32_t eos) {
  *model = (cpu_model){0};
  et_ac4_memory_v1 plan;
  assert(et_ac4_plan_v1(&model_config, &plan) == ET_AC4_OK);
  assert(plan.activation_floats == STREAM);
  assert(plan.parameter_floats == PARAMETERS);
  assert(plan.workspace_bytes == WORKSPACE_FLOATS * sizeof(float));

  for (uint32_t token = 0; token < VOCAB; ++token) {
    for (size_t h = 0; h < HIDDEN; ++h) {
      model->embedding[token][h] =
          (float)((int)((token * (h + 3u) + h * 7u) % 23u) - 11) * 0x1p-4f;
      model->head[token][h] =
          (float)((int)((token * (h + 7u) + h * 5u) % 17u) - 8) * 0x1p-4f;
    }
  }
  for (size_t p = 0; p < PARAMETERS; ++p)
    model->parameters[p] = (float)((int)((p * 11u) % 17u) - 8) * 0x1p-6f;
  for (size_t h = 0; h < HIDDEN; ++h) {
    model->parameters[h] = 1.0f;
    model->parameters[HIDDEN + 4 * HIDDEN * HIDDEN + h] = 1.0f;
  }
  /* Give the forbidden byte and early EOS the two highest prompt logits. */
  model->bias['x'] = 16.0f;
  model->bias[eos] = 8.0f;
}

/* Embed [rows, prefix], run the admitted CPU causal block on [2,3,4], and
 * project each final prefix position through the fixture's affine head. */
static void model_logits(const cpu_model *model, const prefix rows[ROWS],
                         size_t count, float *logits) {
  assert(count > 0 && count <= ROWS);
  float input[STREAM] = {0};
  const float target[STREAM] = {0};
  float hidden[STREAM];
  float workspace[WORKSPACE_FLOATS];
  for (size_t row = 0; row < count; ++row) {
    assert(rows[row].length > 0 && rows[row].length <= CONTEXT);
    for (size_t t = 0; t < rows[row].length; ++t) {
      assert(rows[row].tokens[t] < VOCAB);
      memcpy(input + (row * CONTEXT + t) * HIDDEN,
             model->embedding[rows[row].tokens[t]], HIDDEN * sizeof(float));
    }
  }

  const et_ac4_io_v1 io = {
      .input = {input, STREAM},
      .target = {target, STREAM},
      .parameters = {model->parameters, PARAMETERS},
      .output = {hidden, STREAM},
      .workspace = workspace,
      .workspace_bytes = sizeof(workspace)};
  et_ac4_result_v1 result;
  assert(et_ac4_run_v1(&model_config, &io, &result) == ET_AC4_OK);
  for (size_t row = 0; row < count; ++row) {
    const float *last = hidden + (row * CONTEXT + rows[row].length - 1u) * HIDDEN;
    for (size_t token = 0; token < VOCAB; ++token) {
      float logit = model->bias[token];
      for (size_t h = 0; h < HIDDEN; ++h)
        logit += last[h] * model->head[token][h];
      assert(isfinite(logit));
      logits[row * VOCAB + token] = logit;
    }
  }
}

static et_cd_vocab_index *index_tokenizer(const et_cd_dfa *dfa,
                                         const et_t2_bpe_model *tokenizer,
                                         uint32_t eos) {
  uint8_t storage[VOCAB][ET_T2_BPE_MAX_TOKEN_BYTES];
  const uint8_t *spellings[VOCAB];
  size_t lengths[VOCAB];
  for (uint32_t token = 0; token < VOCAB; ++token) {
    assert(et_t2_bpe_token_bytes(tokenizer, token, storage[token],
                                 sizeof(storage[token]), &lengths[token]) ==
           ET_T2_BPE_OK);
    spellings[token] = storage[token];
  }
  et_cd_vocab_index *index = NULL;
  assert(et_cd_vocab_index_build(dfa, spellings, lengths, VOCAB, eos, &index) ==
         ET_CD_OK);
  return index;
}

typedef struct {
  size_t count;
  et_beam_hypothesis_v1 hypotheses[ROWS];
} generation;

static et_beam_status manual_step(et_beam_state_v1 *beam,
                                  const et_cd_vocab_index *index,
                                  uint16_t start_state, const prefix rows[ROWS],
                                  const float *logits, size_t count) {
  double masked[ROWS * VOCAB];
  for (size_t row = 0; row < count; ++row) {
    uint16_t state = start_state;
    /* Replay only generated IDs. The prompt conditions the model. */
    for (size_t t = 1; t < rows[row].length; ++t) {
      uint16_t next;
      uint32_t finished;
      assert(et_cd_advance_token_v1(index, state, rows[row].tokens[t],
                                    &next, &finished) == ET_CD_OK);
      assert(!finished);
      state = next;
    }
    const size_t offset = row * VOCAB;
    float constrained[VOCAB];
    const et_cd_status mask_status =
        et_cd_apply_logit_mask_v1(index, state, logits + offset, constrained);
    if (mask_status == ET_CD_ERR_NONFINITE)
      return ET_BEAM_ERR_NONFINITE;
    assert(mask_status == ET_CD_OK);
    for (size_t token = 0; token < VOCAB; ++token)
      masked[offset + token] = (double)constrained[token];
  }
  return et_beam_step_v1(beam, masked, count);
}

/* The fixture requires at least one EOS-completed hypothesis. All search state
 * is local, and a failed step or exhausted incomplete search publishes nothing. */
static et_beam_status generate(const cpu_model *model,
                              const et_cd_vocab_index *index,
                              const et_beam_config_v1 *config, uint32_t prompt,
                              bool inject_second_row_nan, generation *out) {
  et_cd_vocab_index_info info;
  assert(et_cd_vocab_index_info_get(index, &info) == ET_CD_OK);
  assert(info.vocab_size == config->vocab_size && config->vocab_size == VOCAB);
  assert(info.eos_token_id == config->eos_token_id);
  assert(config->num_beams <= ROWS && config->max_new_tokens <= CONTEXT);
  et_beam_state_v1 beam, manual;
  assert(et_beam_init_v1(&beam, config, &prompt, 1) == ET_BEAM_OK);
  memcpy(&manual, &beam, sizeof(manual));
  while (!et_beam_is_done_v1(&beam)) {
    const size_t count = et_beam_active_count_v1(&beam);
    prefix rows[ROWS];
    for (size_t row = 0; row < count; ++row) {
      double score;
      assert(et_beam_get_active_beam_v1(&beam, row, rows[row].tokens, CONTEXT,
                                        &rows[row].length, &score) == ET_BEAM_OK);
    }

    float logits[ROWS * VOCAB];
    model_logits(model, rows, count, logits);
    /* The prompt's byte 'x' is forbidden in either continuation row. */
    if (inject_second_row_nan && count == ROWS)
      logits[VOCAB + prompt] = NAN;
    const et_beam_status status =
        et_constrained_beam_step_v1(&beam, index, logits, count, VOCAB);
    /* Before advancing the control, it is a byte snapshot of the prior step. */
    if (status != ET_BEAM_OK)
      assert(memcmp(&beam, &manual, sizeof(beam)) == 0);
    assert(manual_step(&manual, index, info.start_state, rows, logits, count) ==
           status);
    if (status != ET_BEAM_OK)
      return status;
    assert(memcmp(&beam, &manual, sizeof(beam)) == 0);
  }

  et_beam_hypothesis_v1 ranked[ROWS];
  size_t count;
  assert(et_beam_finalize_v1(&beam, ranked, ROWS, &count) == ET_BEAM_OK);
  generation result = {0};
  for (size_t rank = 0; rank < count; ++rank)
    if (ranked[rank].finished)
      result.hypotheses[result.count++] = ranked[rank];
  if (result.count == 0)
    return ET_BEAM_ERR_NO_VALID_TOKENS;
  *out = result;
  return ET_BEAM_OK;
}

static void check_decoded(const et_t2_bpe_model *tokenizer,
                          const et_beam_hypothesis_v1 *hyp, uint32_t prompt,
                          uint32_t eos, const uint8_t *literal, size_t length) {
  assert(hyp->finished && hyp->seq_len == hyp->generated_len + 1u);
  assert(hyp->tokens[0] == prompt && hyp->tokens[hyp->seq_len - 1u] == eos);
  uint8_t decoded[CONTEXT];
  size_t decoded_length;
  assert(et_t2_bpe_decode(tokenizer, hyp->tokens + 1, hyp->generated_len, decoded,
                          sizeof(decoded), &decoded_length) == ET_T2_BPE_OK);
  assert(decoded_length == length && memcmp(decoded, literal, length) == 0);
}

int main(void) {
  const uint8_t literal[] = {'a', 'b'}, alternative[] = {'a', 'c'};
  const uint8_t corpus[] = {'a', 'b', 'a', 'b', 'a', 'b'};
  const et_t2_bpe_document document = {corpus, sizeof(corpus)};
  const et_t2_bpe_train_config training = {
      VOCAB, 2, ET_T2_BPE_SPECIAL_BOS | ET_T2_BPE_SPECIAL_EOS | ET_T2_BPE_SPECIAL_PAD};
  et_t2_bpe_model *tokenizer = NULL;
  et_t2_bpe_info info;
  assert(et_t2_bpe_train(&document, 1, &training, &tokenizer) == ET_T2_BPE_OK);
  assert(et_t2_bpe_get_info(tokenizer, &info) == ET_T2_BPE_OK);
  assert(info.vocab_size == VOCAB && info.merge_count == 1);
  uint32_t merged, prompt;
  size_t encoded;
  assert(et_t2_bpe_encode(tokenizer, literal, sizeof(literal), &merged, 1,
                           &encoded) == ET_T2_BPE_OK && encoded == 1);
  assert(merged >= ET_T2_BPE_BASE_VOCAB_SIZE && merged < info.bos_id);
  const uint8_t input[] = {'x'};
  assert(et_t2_bpe_encode(tokenizer, input, sizeof(input), &prompt, 1,
                           &encoded) == ET_T2_BPE_OK && encoded == 1);

  const uint8_t *choices[] = {literal, alternative};
  const size_t lengths[] = {sizeof(literal), sizeof(alternative)};
  et_cd_dfa *dfa = NULL;
  assert(et_cd_dfa_compile_literal_choices(choices, lengths, 2, &dfa) == ET_CD_OK);
  et_cd_vocab_index *index = index_tokenizer(dfa, tokenizer, info.eos_id);
  et_cd_vocab_index_info index_info;
  assert(et_cd_vocab_index_info_get(index, &index_info) == ET_CD_OK);

  cpu_model model;
  initialize_model(&model, info.eos_id);
  const prefix start[ROWS] = {{{prompt}, 1}};
  float first[VOCAB];
  model_logits(&model, start, 1, first);
  for (uint32_t token = 0; token < VOCAB; ++token) {
    assert(token == prompt || first[prompt] > first[token]);
    assert(token == prompt || token == info.eos_id ||
           first[info.eos_id] > first[token]);
    const uint32_t expected = token == 'a' || token == merged;
    assert(et_cd_vocab_index_is_token_allowed(index, index_info.start_state,
                                               token) == expected);
  }

  /* Changing the batch order preserves both live prefix rows. */
  const prefix next[ROWS] = {{{prompt, 'a'}, 2}, {{prompt, merged}, 2}};
  const prefix swapped[ROWS] = {next[1], next[0]};
  float second[ROWS * VOCAB], permuted[ROWS * VOCAB];
  model_logits(&model, next, ROWS, second);
  model_logits(&model, swapped, ROWS, permuted);
  const size_t row_bytes = VOCAB * sizeof(float);
  assert(memcmp(second, second + VOCAB, row_bytes) != 0);
  assert(memcmp(second, permuted + VOCAB, row_bytes) == 0);
  assert(memcmp(second + VOCAB, permuted, row_bytes) == 0);

  const et_beam_config_v1 config = {.struct_size = sizeof(config),
                                   .num_beams = ROWS,
                                   .vocab_size = VOCAB,
                                   .eos_token_id = info.eos_id,
                                   .max_new_tokens = CONTEXT,
                                   .repetition_penalty = 1.0};
  generation result;
  assert(generate(&model, index, &config, prompt, false, &result) == ET_BEAM_OK);
  assert(result.count == ROWS);

  /* Enumerate the three completions. The split row chooses b or c from its
   * second forward, and each accepting row has exactly one allowed ID, EOS. */
  const double a = (double)first['a'];
  const double ab = (double)first[merged];
  const double b = (double)second['b'];
  const double c = (double)second['c'];
  assert(b != c);
  const double first_normalizer = fmax(a, ab) + log1p(exp(-fabs(a - ab)));
  const double split_normalizer = fmax(b, c) + log1p(exp(-fabs(b - c)));
  const double joined_score = ab - first_normalizer;
  const double split_score = (a - first_normalizer) + (fmax(b, c) - split_normalizer);
  const double other_score = (a - first_normalizer) + (fmin(b, c) - split_normalizer);
  assert(joined_score > other_score && joined_score != split_score);
  const size_t split_rank = split_score > joined_score ? 0u : 1u;
  const et_beam_hypothesis_v1 *split = &result.hypotheses[split_rank];
  const et_beam_hypothesis_v1 *joined = &result.hypotheses[1u - split_rank];
  assert(split->generated_len == 3 && split->tokens[1] == 'a' &&
         split->tokens[2] == (b > c ? 'b' : 'c'));
  assert(joined->generated_len == 2 && joined->tokens[1] == merged);
  assert(fabs(split->sum_log_prob - split_score) < 1e-14);
  assert(fabs(joined->sum_log_prob - joined_score) < 1e-14);
  assert(split->score == split->sum_log_prob && joined->score == joined->sum_log_prob);
  check_decoded(tokenizer, split, prompt, info.eos_id,
                b > c ? literal : alternative, sizeof(literal));
  check_decoded(tokenizer, joined, prompt, info.eos_id, literal, sizeof(literal));

  et_beam_config_v1 short_budget = config;
  short_budget.max_new_tokens = 2;
  generation partial;
  assert(generate(&model, index, &short_budget, prompt, false, &partial) ==
         ET_BEAM_OK);
  assert(partial.count == 1 && partial.hypotheses[0].tokens[1] == merged);
  assert(partial.hypotheses[0].score == joined->score);
  check_decoded(tokenizer, &partial.hypotheses[0], prompt, info.eos_id,
                literal, sizeof(literal));

  /* Prior successful output survives each failed generation. */
  generation saved;
  memcpy(&saved, &result, sizeof(saved));
  assert(generate(&model, index, &config, prompt, true, &result) ==
         ET_BEAM_ERR_NONFINITE);
  assert(memcmp(&saved, &result, sizeof(result)) == 0);

  /* Even with a viable split branch, min-length suppression empties the
   * EOS-only merged row. */
  et_beam_config_v1 reject = config;
  reject.min_new_tokens = 2;
  assert(generate(&model, index, &reject, prompt, false, &result) ==
         ET_BEAM_ERR_NO_VALID_TOKENS);
  assert(memcmp(&saved, &result, sizeof(result)) == 0);
  reject = config;
  reject.max_new_tokens = 1;
  assert(generate(&model, index, &reject, prompt, false, &result) ==
         ET_BEAM_ERR_NO_VALID_TOKENS);
  assert(memcmp(&saved, &result, sizeof(result)) == 0);

  et_cd_vocab_index_destroy(index);
  et_cd_dfa_destroy(dfa);
  et_t2_bpe_destroy(tokenizer);
  printf("BEAM/CD live CPU: PASS (T2 vocabulary %u, causal [2,3,4], "
         "split a%c %a, merged ab %a, coordinated/manual byte parity, "
         "nonfinite, empty and budget rejection)\n",
         info.vocab_size, (int)split->tokens[2], split->score, joined->score);
  return 0;
}
