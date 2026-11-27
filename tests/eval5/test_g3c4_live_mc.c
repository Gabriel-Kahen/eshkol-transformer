#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3C4_PROVIDER_ROUTES_PRIVATE 1
#define ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE 1

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <inttypes.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);           \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

enum { WINDOW = 4, VOCAB = 256, PROMPT = 2 };
enum { SHORT, LONG, TIED_SUFFIX, LONG_SUFFIX, DIFFERENT_ANSWER, CASES };
static const char text[CASES][WINDOW + 1] = {
    "Q=1.", "Q=11", "Q=1?", "Q=17", "Q=2."};

typedef struct live_case {
  uint32_t tokens[WINDOW];
  float rows[WINDOW * VOCAB];
} live_case;

static void load_mapping(const char *path, uint8_t mapping[VOCAB]) {
  FILE *file = fopen(path, "rb");
  CHECK(file != NULL);
  CHECK(fread(mapping, 1, VOCAB, file) == VOCAB);
  CHECK(fgetc(file) == EOF && !ferror(file) && fclose(file) == 0);
  for (size_t i = 0; i < VOCAB; ++i)
    CHECK(mapping[i] == i);
}

static void produce_cases(const uint8_t mapping[VOCAB], live_case cases[CASES]) {
  unsigned char identities[14];
  void *owner = et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner != NULL);
  for (int64_t i = 0; i < (int64_t)COUNT(identities); ++i)
    CHECK(et_g3c4_private_model_owner_bind_v1(owner, i, &identities[i]) == 0);
  for (int64_t i = 0; i < (int64_t)COUNT(identities); ++i)
    CHECK(et_g3c4_private_model_owner_initialize_v1(owner, i) == 0);
  CHECK(et_g3c4_private_model_owner_prepare_seal_v1(owner) == 0);
  CHECK(et_g3c4_private_model_owner_commit_seal_v1(owner) == 0);
  void *context = et_g3c4_private_generator_seed_v1(
      owner, 7, 0, INT64_C(0x3f800000), VOCAB, INT64_C(0x3f800000), 1, -1);
  CHECK(context != NULL);
  CHECK(et_g3c4_private_call_acquire_v1(context, 0, 0) == 0);
  for (size_t i = 0; i < CASES; ++i) {
    int64_t ids[WINDOW];
    CHECK(strlen(text[i]) == WINDOW);
    for (size_t t = 0; t < WINDOW; ++t) {
      cases[i].tokens[t] = mapping[(uint8_t)text[i][t]];
      ids[t] = cases[i].tokens[t];
    }
    CHECK(et_g3c4_private_full_prefix_forward_v1(context, ids, cases[i].rows) == 0);
    for (size_t j = 0; j < WINDOW * VOCAB; ++j)
      CHECK(isfinite(cases[i].rows[j]));
  }
  CHECK(et_g3c4_private_call_finish_v1(context) == 0);
  CHECK(et_g3c4_private_generator_close_v1(context) == 0);
}

static void check_causal_rows(const live_case cases[CASES]) {
  for (size_t i = 0; i < CASES; ++i) {
    for (size_t j = i + 1u; j < CASES; ++j) {
      size_t common = 0;
      while (common < WINDOW && cases[i].tokens[common] == cases[j].tokens[common])
        ++common;
      CHECK(common >= PROMPT && common < WINDOW);
      const size_t words = common * VOCAB;
      CHECK(memcmp(cases[i].rows, cases[j].rows, words * sizeof(float)) == 0);
      CHECK(memcmp(cases[i].rows + words, cases[j].rows + words,
                   VOCAB * sizeof(float)) != 0);
    }
  }
}

static et_eval5_sequence choice(const live_case *source, size_t length) {
  CHECK(length > 0 && length <= WINDOW - PROMPT);
  return (et_eval5_sequence){
      .logits = source->rows + (PROMPT - 1) * VOCAB,
      .target_ids = source->tokens + PROMPT,
      .token_count = length, .vocab_size = VOCAB, .logit_count = length * VOCAB};
}

static long double oracle_cost(const et_eval5_sequence *choice, uint32_t mode) {
  long double cost = 0;
  for (size_t r = 0; r < choice->token_count; ++r) {
    const float *row = choice->logits + r * VOCAB;
    long double top = row[0], sum = 0;
    for (size_t t = 1; t < VOCAB; ++t)
      top = fmaxl(top, (long double)row[t]);
    for (size_t t = 0; t < VOCAB; ++t)
      sum += expl((long double)row[t] - top);
    cost += (top - (long double)row[choice->target_ids[r]]) + logl(sum);
  }
  CHECK(isfinite(cost));
  return mode ? cost / (long double)choice->token_count : cost;
}

static size_t oracle_winner(const et_eval5_choice_item *item, uint32_t mode) {
  size_t winner = 0;
  long double lowest = oracle_cost(&item->choices[0], mode);
  for (size_t i = 1; i < item->choice_count; ++i) {
    const long double cost = oracle_cost(&item->choices[i], mode);
    if (cost < lowest) {
      lowest = cost;
      winner = i;
    }
  }
  return winner;
}

static void check_scoring(const live_case cases[CASES]) {
  const et_eval5_sequence contrast[] = {
      choice(&cases[SHORT], 1), choice(&cases[LONG], 2)};
  const et_eval5_sequence alternatives[] = {
      choice(&cases[DIFFERENT_ANSWER], 1), choice(&cases[LONG_SUFFIX], 2)};
  for (size_t i = 0; i < COUNT(contrast); ++i) {
    CHECK(contrast[i].target_ids[i] != alternatives[i].target_ids[i]);
    CHECK(memcmp(contrast[i].logits, alternatives[i].logits,
                 contrast[i].logit_count * sizeof(float)) == 0);
  }
  const et_eval5_sequence tied[] = {
      choice(&cases[SHORT], 1), choice(&cases[TIED_SUFFIX], 1)};
  const et_eval5_choice_item items[] = {
      {contrast, COUNT(contrast), 1}, {tied, COUNT(tied), 0}, {tied, COUNT(tied), 1}};
  CHECK(oracle_cost(&contrast[1], 0) - oracle_cost(&contrast[0], 0) > 1.0L);
  CHECK(oracle_cost(&contrast[0], 1) - oracle_cost(&contrast[1], 1) > 0.01L);
  CHECK(oracle_cost(&tied[0], 0) == oracle_cost(&tied[1], 0));

  for (uint32_t mode = 0; mode <= 1; ++mode) {
    uint64_t hits = 0;
    for (size_t i = 0; i < COUNT(items); ++i) {
      const size_t winner = oracle_winner(&items[i], mode);
      CHECK(winner == (i == 0 ? mode : 0u));
      hits += winner == items[i].correct_index;
      for (size_t label = 0; label < items[i].choice_count; ++label) {
        et_eval5_choice_item probe = items[i];
        probe.correct_index = label;
        et_eval5_mc_metrics answer;
        CHECK(et_eval5_score_mc(&probe, 1, mode, &answer) == ET_EVAL5_OK);
        CHECK(answer.total_count == 1 && answer.correct_count == (label == winner));
      }
    }
    et_eval5_mc_metrics result;
    CHECK(et_eval5_score_mc(items, COUNT(items), mode, &result) == ET_EVAL5_OK);
    CHECK(result.total_count == COUNT(items) && result.correct_count == hits &&
          result.accuracy == (double)hits / (double)COUNT(items));
    printf("EVAL5 G3-C4 live MC %s: %" PRIu64 "/%" PRIu64
           " correct, length-choice winner %zu, tie winner 0\n",
           mode ? "normalized" : "raw", result.correct_count, result.total_count,
           oracle_winner(&items[0], mode));

    /* Reject a later item's malformed target after first traversing a live item. */
    et_eval5_sequence invalid = contrast[1];
    const uint32_t bad_targets[] = {invalid.target_ids[0], VOCAB};
    invalid.target_ids = bad_targets;
    const et_eval5_choice_item malformed[] = {items[0], {&invalid, 1, 0}};
    unsigned char before[sizeof(result)];
    memset(&result, 0xa5, sizeof(result));
    memcpy(before, &result, sizeof(result));
    CHECK(et_eval5_score_mc(malformed, COUNT(malformed), mode, &result) ==
          ET_EVAL5_ERR_TARGET);
    CHECK(memcmp(&result, before, sizeof(result)) == 0);
  }
}

int main(int argc, char **argv) {
  CHECK(argc == 3);
  uint8_t mapping[VOCAB];
  live_case cases[CASES];
  load_mapping(argv[1], mapping);
  produce_cases(mapping, cases);
  check_causal_rows(cases);
  check_scoring(cases);
  FILE *trace = fopen(argv[2], "wb");
  CHECK(trace != NULL);
  for (size_t i = 0; i < CASES; ++i) {
    CHECK(fwrite(cases[i].tokens, sizeof(uint32_t), WINDOW, trace) == WINDOW);
    CHECK(fwrite(cases[i].rows, sizeof(float), WINDOW * VOCAB, trace) == WINDOW * VOCAB);
  }
  CHECK(fclose(trace) == 0);
  puts("EVAL5 G3-C4 live MC PASS: five complete T1 windows, causal rows and late rejection");
  return EXIT_SUCCESS;
}
