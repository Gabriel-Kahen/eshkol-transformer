/* Executable detector controls for the exact noninstalled native wrapper. */
#define ET_E3_TESTING 1
#include "retention_first_eval_frame.c"

#include <stdio.h>

static unsigned char identities[14];
static size_t checks;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
  fprintf(stderr, "first-eval native control failed at line %d: %s\n", \
          __LINE__, #condition); \
  exit(1); \
} } while (0)
#define OK(call) CHECK((call) == 0)

typedef struct model_image {
  size_t count[14][2];
  uint32_t words[14][2][1024];
  int64_t metadata[14][3];
} model_image;

static owner *make_sealed_owner(void) {
  initializer *initial = et_m3t_private_initializer_create_v1(1729);
  CHECK(initial != NULL);
  OK(et_m3t_private_initializer_seal_v1(initial));
  owner *model = et_m3t_private_owner_create_v1(initial);
  CHECK(model != NULL);
  for (size_t i = 0; i < 14; ++i)
    OK(et_m3t_private_owner_bind_v1(model, (int64_t)i, &identities[i]));
  for (size_t i = 0; i < 14; ++i)
    OK(et_m3t_private_owner_initialize_v1(model, (int64_t)i));
  OK(et_m3t_private_owner_seal_v1(model));
  return model;
}

static void *make_stale_owner(void) {
  initializer *initial = et_m3t_private_initializer_create_v1(1730);
  CHECK(initial != NULL);
  OK(et_m3t_private_initializer_seal_v1(initial));
  void *stale = et_m3t_private_owner_create_v1(initial);
  CHECK(stale != NULL);
  OK(et_m3t_private_owner_abort_v1(stale));
  return stale;
}

static void snapshot(owner *model, model_image *image) {
  memset(image, 0, sizeof(*image));
  for (int64_t parameter = 0; parameter < 14; ++parameter) {
    for (int64_t gradient = 0; gradient < 2; ++gradient) {
      int64_t count = et_e3_test_model_word_count_v1(
          model, parameter, gradient);
      CHECK(count > 0 && count <= 1024);
      image->count[parameter][gradient] = (size_t)count;
      for (int64_t index = 0; index < count; ++index) {
        int64_t word = et_e3_test_model_word_v1(
            model, parameter, gradient, index);
        CHECK(word >= 0 && word <= UINT32_MAX);
        image->words[parameter][gradient][index] = (uint32_t)word;
      }
    }
    for (int64_t field = 0; field < 3; ++field) {
      int64_t value = et_e3_test_gradient_metadata_v1(model, parameter, field);
      CHECK(value >= 0);
      image->metadata[parameter][field] = value;
    }
  }
}

static void denied_unchanged(owner *model, void *candidate, int64_t selector) {
  model_image before, after;
  snapshot(model, &before);
  CHECK(et_e3_test_first_eval_corrupt_v1(candidate, selector) == -1);
  CHECK(first_eval_mutation_used == 0);
  snapshot(model, &after);
  CHECK(memcmp(&before, &after, sizeof(before)) == 0);
}

int main(int argc, char **argv) {
  CHECK(argc == 2 && argv[1][0] >= '0' && argv[1][0] <= '2' &&
        argv[1][1] == '\0');
  const int64_t selector = argv[1][0] - '0';
  void *stale = make_stale_owner();
  owner *model = make_sealed_owner();
  if (selector != 0) OK(et_e3_test_seed_gradient_v1(model));
  CHECK(first_eval_mutation_used == 0);

  denied_unchanged(model, stale, selector);
  denied_unchanged(model, &checks, selector);
  denied_unchanged(model, model->original, selector);
  denied_unchanged(model, model, -1);
  denied_unchanged(model, model, 3);
  model->active = model;
  denied_unchanged(model, model, selector);
  model->active = NULL;
  model->r.state = 0;
  denied_unchanged(model, model, selector);
  model->r.state = 1;
  if (selector != 0) {
    et_f32_parameter_test_set_metadata_v1(
        model->p[0], ET_F32_GRADIENT_PRESENT, 2,
        UINT32_C(0x3f800000));
    denied_unchanged(model, model, selector);
    et_f32_parameter_test_set_metadata_v1(
        model->p[0], ET_F32_GRADIENT_PRESENT, 1,
        UINT32_C(0x3f800000));
  }

  model_image before, after, second;
  snapshot(model, &before);
  CHECK(et_e3_test_first_eval_corrupt_v1(model, selector) == 0);
  CHECK(first_eval_mutation_used == 1);
  snapshot(model, &after);
  for (size_t parameter = 0; parameter < 14; ++parameter) {
    for (size_t gradient = 0; gradient < 2; ++gradient) {
      CHECK(before.count[parameter][gradient] ==
            after.count[parameter][gradient]);
      for (size_t index = 0; index < before.count[parameter][gradient];
           ++index) {
        if (parameter == 0 && gradient == 1 && index == 0) {
          CHECK(after.words[parameter][gradient][index] ==
                (selector == 1 ? UINT32_C(0x3f000001)
                               : UINT32_C(0x3f000000)));
        } else {
          CHECK(after.words[parameter][gradient][index] ==
                before.words[parameter][gradient][index]);
        }
      }
    }
    for (size_t field = 0; field < 3; ++field) {
      const int64_t expected = parameter == 0
          ? (field == 0 ? ET_F32_GRADIENT_PRESENT
             : field == 1 ? (selector == 2 ? 2 : 1)
                          : UINT32_C(0x3f800000))
          : before.metadata[parameter][field];
      CHECK(after.metadata[parameter][field] == expected);
    }
  }
  CHECK(memcmp(&before, &after, sizeof(before)) != 0);
  CHECK(et_e3_test_first_eval_corrupt_v1(model, selector) == -1);
  CHECK(first_eval_mutation_used == 1);
  snapshot(model, &second);
  CHECK(memcmp(&after, &second, sizeof(after)) == 0);
  printf("FIRST-EVAL-NATIVE-CONTROL\t%lld\t%zu\n",
         (long long)selector, checks);
  return 0;
}
