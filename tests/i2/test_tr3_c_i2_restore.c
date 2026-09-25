#define ET_F32_TENSOR_TESTING 1
#define ET_I2_NATIVE_HELPERS_ONLY 1
#define ET_I2_PRIVATE_OWNED_CLONE_MATCH 1
#define ET_TR3_C_I2_RESTORE_PRIVATE 1
#include "../../native/i2_wave2_package_bridge.c"

#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct restore_fixture {
  uint64_t authority;
  uint64_t identities[ET_TR3_C_MODEL_DESTINATIONS];
  et_f32_parameter *parameters[ET_TR3_C_MODEL_DESTINATIONS];
  et_f32_tensor *model_sources[ET_TR3_C_MODEL_DESTINATIONS];
  et_f32_tensor *moment_destinations[ET_TR3_C_O2_MOMENT_ASSIGNMENTS];
  et_f32_tensor *moment_sources[ET_TR3_C_O2_MOMENT_ASSIGNMENTS];
  et_tr3_c_i2_restore42_append_v1 request;
} restore_fixture;

typedef struct builder_snapshot {
  et_i2_copy_builder shell;
  et_f32_tensor_copy_assignment_v1
      assignments[ET_TR3_C_RESTORE_ASSIGNMENTS];
} builder_snapshot;

typedef struct allocation_snapshot {
  size_t bridge;
  size_t f32;
} allocation_snapshot;

enum { TEST_MAX_ELEMENTS = 1024 };

static const size_t model_ranks[ET_TR3_C_MODEL_DESTINATIONS] = {
    2u, 2u, 2u, 2u, 2u, 2u, 1u, 1u, 1u, 1u, 2u, 1u, 1u, 2u,
};

static const uint64_t model_shapes[ET_TR3_C_MODEL_DESTINATIONS][2] = {
    {4u, 4u},   {4u, 4u}, {4u, 4u}, {4u, 4u}, {4u, 8u},
    {8u, 4u},   {4u, 0u}, {4u, 0u}, {4u, 0u}, {4u, 0u},
    {256u, 4u}, {4u, 0u}, {4u, 0u}, {2u, 4u},
};

static int checks;
static int failures;

#define CHECK(condition)                                                       \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,          \
                    #condition);                                               \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static et_f32_tensor *tensor_create(size_t elements, uint32_t bits) {
  const uint64_t shape[] = {(uint64_t)elements};
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  uint32_t payload[2] = {bits, bits};
  if (elements > 2u ||
      et_f32_tensor_create_v1(1u, shape, &tensor, &error) != 0 ||
      et_f32_tensor_copy_bits_from_v1(tensor, payload, elements, &error) != 0) {
    return NULL;
  }
  return tensor;
}

static et_f32_tensor *tensor_create_shape(size_t rank, const uint64_t *shape,
                                          uint32_t bits) {
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  uint32_t payload[TEST_MAX_ELEMENTS];
  size_t elements = 1u;
  for (size_t dimension = 0u; dimension < rank; ++dimension) {
    elements *= (size_t)shape[dimension];
  }
  if (elements > TEST_MAX_ELEMENTS ||
      et_f32_tensor_create_v1(rank, shape, &tensor, &error) != 0) {
    return NULL;
  }
  for (size_t index = 0u; index < elements; ++index) {
    payload[index] = bits;
  }
  if (et_f32_tensor_copy_bits_from_v1(tensor, payload, elements, &error) != 0) {
    return NULL;
  }
  return tensor;
}

static et_f32_tensor *tensor_create_model(size_t index, uint32_t bits) {
  return tensor_create_shape(model_ranks[index], model_shapes[index], bits);
}

static et_f32_tensor *owned_create(size_t elements, uint32_t bits) {
  et_f32_tensor_error error;
  et_f32_tensor *source = tensor_create(elements, bits);
  et_f32_tensor *owned = NULL;
  if (source == NULL ||
      et_f32_owned_tensor_clone_v1(source, &owned, &error) != 0 ||
      et_f32_tensor_destroy_v1(&source, &error) != 0) {
    return NULL;
  }
  return owned;
}

static et_f32_tensor *owned_create_model(size_t index, uint32_t bits) {
  et_f32_tensor_error error;
  et_f32_tensor *source = tensor_create_model(index, bits);
  et_f32_tensor *owned = NULL;
  if (source == NULL ||
      et_f32_owned_tensor_clone_v1(source, &owned, &error) != 0 ||
      et_f32_tensor_destroy_v1(&source, &error) != 0) {
    return NULL;
  }
  return owned;
}

static void set_bits(et_f32_tensor *tensor, uint32_t bits) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, &bits, 1u, &error) == 0);
}

static void fill_bits(et_f32_tensor *tensor, uint32_t bits) {
  et_f32_tensor_error error;
  uint32_t payload[TEST_MAX_ELEMENTS];
  size_t elements = 0u;
  CHECK(et_f32_tensor_element_count_v1(tensor, &elements, &error) == 0);
  CHECK(elements <= TEST_MAX_ELEMENTS);
  for (size_t index = 0u; index < elements; ++index) {
    payload[index] = bits;
  }
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, payload, elements, &error) == 0);
}

static void check_bits_equal(const et_f32_tensor *left,
                             const et_f32_tensor *right) {
  et_f32_tensor_error error;
  int32_t equal = 0;
  CHECK(et_f32_tensor_bits_equal_v1(left, right, &equal, &error) == 0);
  CHECK(equal == 1);
}

static void check_all_bits(const et_f32_tensor *tensor, uint32_t expected) {
  et_f32_tensor_error error;
  uint32_t payload[TEST_MAX_ELEMENTS];
  size_t elements = 0u;
  CHECK(et_f32_tensor_element_count_v1(tensor, &elements, &error) == 0);
  CHECK(elements <= TEST_MAX_ELEMENTS);
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, payload, elements, &error) == 0);
  for (size_t index = 0u; index < elements; ++index) {
    CHECK(payload[index] == expected);
  }
}

static uint32_t model_bits(size_t index) {
  return UINT32_C(0x3f000000) + (uint32_t)index;
}

static uint32_t moment_bits(size_t index) {
  return index % 2u == 0u
             ? UINT32_C(0xbf000000) + (uint32_t)(index << 8u)
             : UINT32_C(0x3e800000) + (uint32_t)(index << 8u);
}

static uint32_t get_bits(const et_f32_tensor *tensor) {
  et_f32_tensor_error error;
  uint32_t bits = UINT32_C(0xffffffff);
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, &bits, 1u, &error) == 0);
  return bits;
}

static void fixture_init(restore_fixture *fixture) {
  et_f32_tensor_error error;
  memset(fixture, 0, sizeof(*fixture));
  fixture->authority = UINT64_C(0x726573746f726534);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    et_f32_tensor *initial = tensor_create_model(index, 0u);
    fixture->identities[index] = UINT64_C(0x1000) + index;
    CHECK(initial != NULL);
    CHECK(et_f32_parameter_create_v1(initial, &fixture->parameters[index],
                                     &error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(
              fixture->parameters[index], &fixture->identities[index],
              &error) == 0);
    CHECK(et_f32_tensor_destroy_v1(&initial, &error) == 0);
    fixture->model_sources[index] = owned_create_model(index, model_bits(index));
    CHECK(fixture->model_sources[index] != NULL);
    fixture->request.parameters[index] = fixture->parameters[index];
    fixture->request.p1_handles[index] = &fixture->identities[index];
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    const size_t parameter_index = index / 2u;
    const uint32_t bits = moment_bits(index);
    fixture->moment_destinations[index] =
        tensor_create_model(parameter_index, 0u);
    fixture->moment_sources[index] = owned_create_model(parameter_index, bits);
    CHECK(fixture->moment_destinations[index] != NULL);
    CHECK(fixture->moment_sources[index] != NULL);
    fixture->request.moment_destinations[index] =
        fixture->moment_destinations[index];
    fixture->request.moment_sources[index] = fixture->moment_sources[index];
  }
  fixture->request.struct_size = sizeof(fixture->request);
  fixture->request.restore_authority = &fixture->authority;
}

static void fixture_release(restore_fixture *fixture) {
  et_f32_tensor_error error;
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    if (fixture->model_sources[index] != NULL) {
      CHECK(et_f32_owned_tensor_release_v1(fixture->model_sources[index],
                                           &error) == 0);
      fixture->model_sources[index] = NULL;
    }
    if (fixture->parameters[index] != NULL) {
      CHECK(et_f32_parameter_destroy_v1(&fixture->parameters[index], &error) ==
            0);
    }
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    if (fixture->moment_sources[index] != NULL) {
      CHECK(et_f32_owned_tensor_release_v1(fixture->moment_sources[index],
                                           &error) == 0);
      fixture->moment_sources[index] = NULL;
    }
    if (fixture->moment_destinations[index] != NULL) {
      CHECK(et_f32_tensor_destroy_v1(&fixture->moment_destinations[index],
                                     &error) == 0);
    }
  }
}

static void check_fixture_plan_pins(const restore_fixture *fixture,
                                    size_t expected) {
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    size_t destination_pins = SIZE_MAX;
    size_t source_pins = SIZE_MAX;
    CHECK(et_f32_tensor_test_plan_pins_v1(
              et_f32_parameter_canonical_owner_v1(fixture->parameters[index]),
              &destination_pins) == 0);
    CHECK(et_f32_tensor_test_plan_pins_v1(fixture->model_sources[index],
                                          &source_pins) == 0);
    CHECK(destination_pins == expected);
    CHECK(source_pins == expected);
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    size_t destination_pins = SIZE_MAX;
    size_t source_pins = SIZE_MAX;
    CHECK(et_f32_tensor_test_plan_pins_v1(
              fixture->moment_destinations[index], &destination_pins) == 0);
    CHECK(et_f32_tensor_test_plan_pins_v1(fixture->moment_sources[index],
                                          &source_pins) == 0);
    CHECK(destination_pins == expected);
    CHECK(source_pins == expected);
  }
}

static void snapshot_builder(const et_i2_copy_builder *builder,
                             builder_snapshot *snapshot) {
  snapshot->shell = *builder;
  memcpy(snapshot->assignments, builder->assignments,
         sizeof(snapshot->assignments));
}

static void check_snapshot(const et_i2_copy_builder *builder,
                           const builder_snapshot *snapshot) {
  CHECK(memcmp(&snapshot->shell, builder, sizeof(*builder)) == 0);
  CHECK(memcmp(snapshot->assignments, builder->assignments,
               sizeof(snapshot->assignments)) == 0);
}

static allocation_snapshot allocations_now(void) {
  const allocation_snapshot snapshot = {
      .bridge = et_i2_successful_allocations,
      .f32 = et_f32_tensor_test_successful_allocations_v1(),
  };
  return snapshot;
}

static void check_allocations(allocation_snapshot expected) {
  const allocation_snapshot actual = allocations_now();
  CHECK(actual.bridge == expected.bridge);
  CHECK(actual.f32 == expected.f32);
}

static void *fixture_builder(restore_fixture *fixture, size_t prefix) {
  void *opaque = NULL;
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(&fixture->authority,
                                                      &opaque) == 0);
  CHECK(opaque != NULL);
  for (size_t index = 0u; index < prefix; ++index) {
    CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
              opaque, &fixture->authority, fixture->parameters[index],
              &fixture->identities[index], fixture->model_sources[index]) == 0);
  }
  return opaque;
}

static void test_matcher(void) {
  et_f32_tensor_error error;
  et_f32_tensor *destination = tensor_create(1u, 0u);
  et_f32_tensor *owned = owned_create(1u, UINT32_C(0x3f800000));
  et_f32_tensor *ordinary = tensor_create(1u, UINT32_C(0x3f800000));
  et_f32_tensor *ordinary_clone = NULL;
  et_f32_tensor *wrong_shape = tensor_create(2u, 0u);
  et_f32_tensor *metadata_destination = tensor_create_model(13u, 0u);
  et_f32_tensor *metadata_source =
      owned_create_model(13u, UINT32_C(0x3f800000));
  et_f32_parameter *parameter = NULL;
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_copy_plan *plan = NULL;
  et_f32_tensor_copy_assignment_v1 assignment = {
      .struct_size = sizeof(assignment),
      .destination = destination,
      .source = owned,
  };

  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) == 0);
  CHECK(et_i2_private_owned_clone_match_v1(
            NULL, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_i2_private_owned_clone_match_v1(
            (const et_f32_tensor *)(uintptr_t)1u, destination,
            ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_i2_private_owned_clone_match_v1(
            ordinary, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_f32_tensor_clone_v1(ordinary, &ordinary_clone, &error) == 0);
  CHECK(et_i2_private_owned_clone_match_v1(
            ordinary_clone, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_f32_parameter_create_v1(ordinary, &parameter, &error) == 0);
  CHECK(et_i2_private_owned_clone_match_v1(
            et_f32_parameter_canonical_owner_v1(parameter), destination,
            ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_i2_private_owned_clone_match_v1(owned, destination, 99u) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, wrong_shape, ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH);

  et_f32_test_tensor_metadata_v1 saved_metadata = {
      .struct_size = sizeof(saved_metadata),
  };
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(metadata_destination,
                                                 &saved_metadata) == 0);
  static const uint32_t scalar_metadata_members[] = {
      ET_F32_TEST_TENSOR_METADATA_RANK,
      ET_F32_TEST_TENSOR_METADATA_ELEMENT_COUNT,
      ET_F32_TEST_TENSOR_METADATA_BYTE_LENGTH,
      ET_F32_TEST_TENSOR_METADATA_DATA,
  };
  for (size_t index = 0u;
       index < sizeof(scalar_metadata_members) /
                   sizeof(scalar_metadata_members[0]);
       ++index) {
    et_f32_test_tensor_metadata_v1 restored = {
        .struct_size = sizeof(restored),
    };
    CHECK(et_f32_tensor_test_metadata_corrupt_v1(
              metadata_destination, scalar_metadata_members[index], 0u) == 0);
    CHECK(et_i2_private_owned_clone_match_v1(
              metadata_source, metadata_destination,
              ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
          ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH);
    CHECK(et_f32_tensor_test_metadata_restore_v1(
              metadata_destination, &saved_metadata) == 0);
    CHECK(et_f32_tensor_test_metadata_snapshot_v1(metadata_destination,
                                                   &restored) == 0);
    CHECK(memcmp(&saved_metadata, &restored, sizeof(restored)) == 0);
  }
  for (size_t dimension = 0u; dimension < saved_metadata.rank; ++dimension) {
    et_f32_test_tensor_metadata_v1 restored = {
        .struct_size = sizeof(restored),
    };
    CHECK(et_f32_tensor_test_metadata_corrupt_v1(
              metadata_destination, ET_F32_TEST_TENSOR_METADATA_SHAPE,
              dimension) == 0);
    CHECK(et_i2_private_owned_clone_match_v1(
              metadata_source, metadata_destination,
              ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
          ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH);
    CHECK(et_f32_tensor_test_metadata_restore_v1(
              metadata_destination, &saved_metadata) == 0);
    CHECK(et_f32_tensor_test_metadata_snapshot_v1(metadata_destination,
                                                   &restored) == 0);
    CHECK(memcmp(&saved_metadata, &restored, sizeof(restored)) == 0);
  }
  for (size_t dimension = 0u; dimension < saved_metadata.rank; ++dimension) {
    et_f32_test_tensor_metadata_v1 restored = {
        .struct_size = sizeof(restored),
    };
    CHECK(et_f32_tensor_test_metadata_corrupt_v1(
              metadata_destination, ET_F32_TEST_TENSOR_METADATA_STRIDE,
              dimension) == 0);
    CHECK(et_i2_private_owned_clone_match_v1(
              metadata_source, metadata_destination,
              ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
          ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH);
    CHECK(et_f32_tensor_test_metadata_restore_v1(
              metadata_destination, &saved_metadata) == 0);
    CHECK(et_f32_tensor_test_metadata_snapshot_v1(metadata_destination,
                                                   &restored) == 0);
    CHECK(memcmp(&saved_metadata, &restored, sizeof(restored)) == 0);
  }

  CHECK(et_f32_tensor_borrow_begin_v1(owned, &borrow, &error) == 0);
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1u, &assignment, &plan, &error) ==
        0);
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  CHECK(et_f32_tensor_copy_plan_release_v1(&plan, &error) == 0);

  set_bits(owned, UINT32_C(0x7fc00001));
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_NONFINITE);
  set_bits(owned, UINT32_C(0xff800000));
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination,
            ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE) ==
        ET_I2_PRIVATE_OWNED_RESULT_NONFINITE);
  set_bits(owned, UINT32_C(0x80000001));
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination,
            ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_VALUE);
  set_bits(owned, UINT32_C(0x80000000));
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination,
            ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE) == 0);
  set_bits(owned, 0u);
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination,
            ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE) == 0);

  et_f32_tensor_test_fail_alloc_after_v1(0u);
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) == 0);
  CHECK(et_f32_tensor_test_successful_allocations_v1() == 0u);
  et_f32_tensor_test_reset_allocator_v1();

  CHECK(et_f32_owned_tensor_release_v1(owned, &error) == 0);
  CHECK(et_i2_private_owned_clone_match_v1(
            owned, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_f32_tensor_destroy_v1(&ordinary, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&ordinary_clone, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&wrong_shape, &error) == 0);
  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  CHECK(et_f32_owned_tensor_release_v1(metadata_source, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&metadata_destination, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&destination, &error) == 0);
}

static void test_create_and_prefix(void) {
  restore_fixture fixture;
  void *opaque = NULL;
  void *nonnull = &fixture;
  void *same_slot = NULL;
  _Alignas(void *) unsigned char unaligned_output[sizeof(void *) + 1u];
  fixture_init(&fixture);

  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(NULL, &opaque) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(&fixture.authority,
                                                      NULL) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(&fixture.authority,
                                                      &nonnull) ==
        ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(&same_slot, &same_slot) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(
            &fixture.authority, (void **)(void *)(unaligned_output + 1u)) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(
            &fixture.authority,
            (void **)(uintptr_t)(UINTPTR_MAX - sizeof(void *) + 1u)) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  et_i2_test_fail_alloc_after_v1(0u);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(&fixture.authority,
                                                      &opaque) ==
        ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED);
  CHECK(opaque == NULL && et_i2_successful_allocations == 0u);
  et_i2_test_fail_alloc_after_v1(1u);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(&fixture.authority,
                                                      &opaque) ==
        ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED);
  CHECK(opaque == NULL && et_i2_successful_allocations == 1u);
  et_i2_test_reset_allocator_v1();

  opaque = fixture_builder(&fixture, 0u);
  et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
  CHECK(builder->count == 42u && builder->tr3_c_next_index == 0u);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(
            &fixture.authority, (void **)(void *)builder) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(
            &fixture.authority,
            (void **)(void *)&builder->assignments[0]) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_i2_private_copy_builder_set_v1(opaque, 0, NULL, 0, NULL, NULL, 0,
                                          NULL) == -1);
  CHECK(et_i2_private_copy_builder_prepare_v1(opaque) == -1);
  CHECK(et_i2_private_copy_builder_commit_v1(opaque) == -1);
  CHECK(et_i2_private_copy_builder_abort_v1(opaque) == -1);
  CHECK(et_tr3_c_i2_copy_builder_commit_checked_v1(opaque) == -1);
  CHECK(et_tr3_c_i2_copy_builder_abort_checked_v1(opaque) == -1);
  builder_snapshot snapshot;
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, (const void *)(uintptr_t)1u, fixture.parameters[0],
            &fixture.identities[0], fixture.model_sources[0]) ==
        ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, NULL, fixture.parameters[0], &fixture.identities[0],
            fixture.model_sources[0]) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  allocation_snapshot allocation = allocations_now();
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture.authority, fixture.parameters[0],
            &fixture.identities[0], fixture.model_sources[0]) == 0);
  check_allocations(allocation);
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture.authority, fixture.parameters[1],
            &fixture.identities[0], fixture.model_sources[1]) ==
        ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  check_allocations(allocation);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture.authority, fixture.parameters[0],
            &fixture.identities[0], fixture.model_sources[0]) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &fixture.authority) == 0);
  check_allocations(allocation);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_ABORTED) == 0);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_COMMITTED) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(opaque, 99u) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            NULL, ET_TR3_C_I2_TERMINAL_ABORTED) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            (const void *)(uintptr_t)1u, ET_TR3_C_I2_TERMINAL_ABORTED) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  builder->count = 1u;
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_ABORTED) ==
        ET_I2_PRIVATE_OWNED_RESULT_INTERNAL);
  builder->count = 0u;
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_ABORTED) == 0);
  fixture_release(&fixture);
}

static void test_suffix_failures(void) {
  restore_fixture fixture;
  fixture_init(&fixture);
  void *incomplete = fixture_builder(
      &fixture, ET_TR3_C_MODEL_DESTINATIONS - 1u);
  builder_snapshot incomplete_snapshot;
  snapshot_builder((et_i2_copy_builder *)incomplete, &incomplete_snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            incomplete, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  check_snapshot((et_i2_copy_builder *)incomplete, &incomplete_snapshot);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            incomplete, &fixture.authority) == 0);
  void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
  et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
  builder_snapshot snapshot;
  et_tr3_c_i2_restore42_append_v1 bad = fixture.request;
  et_f32_tensor_error error;
  et_f32_tensor_borrow *borrow = NULL;
  _Alignas(et_tr3_c_i2_restore42_append_v1)
      unsigned char unaligned_request[sizeof(bad) + 1u];

  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, NULL) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque,
            (const et_tr3_c_i2_restore42_append_v1 *)(const void *)
                (unaligned_request + 1u)) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque,
            (const et_tr3_c_i2_restore42_append_v1 *)(uintptr_t)(
                UINTPTR_MAX - sizeof(bad) + 1u)) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque,
            (const et_tr3_c_i2_restore42_append_v1 *)(const void *)builder) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque,
            (const et_tr3_c_i2_restore42_append_v1 *)(const void *)
                builder->assignments) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  bad.struct_size--;
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_VERSION_MISMATCH);
  check_snapshot(builder, &snapshot);
  bad = fixture.request;
  bad.restore_authority = NULL;
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  bad = fixture.request;
  bad.restore_authority = (const void *)(uintptr_t)1u;
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  check_snapshot(builder, &snapshot);
  bad = fixture.request;
  bad.p1_handles[0] = &fixture.identities[1];
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  check_snapshot(builder, &snapshot);
  bad = fixture.request;
  bad.parameters[0] = fixture.parameters[1];
  bad.p1_handles[0] = &fixture.identities[1];
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  check_snapshot(builder, &snapshot);
  bad = fixture.request;
  bad.moment_sources[2] = bad.moment_sources[0];
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  bad = fixture.request;
  bad.moment_destinations[2] = bad.moment_destinations[0];
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  bad = fixture.request;
  bad.moment_destinations[0] =
      (et_f32_tensor *)fixture.model_sources[0];
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);

  CHECK(et_f32_tensor_borrow_begin_v1(fixture.moment_sources[0], &borrow,
                                      &error) == 0);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  check_snapshot(builder, &snapshot);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_f32_tensor_borrow_begin_v1(fixture.moment_destinations[0], &borrow,
                                      &error) == 0);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  check_snapshot(builder, &snapshot);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);

  et_f32_tensor *pin_destination = tensor_create_model(0u, 0u);
  et_f32_tensor_copy_plan *pin_plan = NULL;
  et_f32_tensor_copy_assignment_v1 pin_assignment = {
      .struct_size = sizeof(pin_assignment),
      .destination = pin_destination,
      .source = fixture.moment_sources[0],
  };
  size_t source_pins = SIZE_MAX;
  size_t destination_pins = SIZE_MAX;
  CHECK(pin_destination != NULL);
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1u, &pin_assignment, &pin_plan,
                                           &error) == 0);
  CHECK(et_f32_tensor_test_plan_pins_v1(fixture.moment_sources[0],
                                        &source_pins) == 0);
  CHECK(et_f32_tensor_test_plan_pins_v1(fixture.moment_destinations[0],
                                        &destination_pins) == 0);
  CHECK(source_pins == 1u && destination_pins == 0u);
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  check_snapshot(builder, &snapshot);
  CHECK(et_f32_tensor_copy_plan_release_v1(&pin_plan, &error) == 0);
  CHECK(et_f32_tensor_test_plan_pins_v1(fixture.moment_sources[0],
                                        &source_pins) == 0);
  CHECK(source_pins == 0u);
  CHECK(et_f32_tensor_destroy_v1(&pin_destination, &error) == 0);

  et_f32_tensor *pin_source = owned_create_model(0u, UINT32_C(0x3f000000));
  pin_assignment.destination = fixture.moment_destinations[0];
  pin_assignment.source = pin_source;
  CHECK(pin_source != NULL);
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1u, &pin_assignment, &pin_plan,
                                           &error) == 0);
  CHECK(et_f32_tensor_test_plan_pins_v1(fixture.moment_sources[0],
                                        &source_pins) == 0);
  CHECK(et_f32_tensor_test_plan_pins_v1(fixture.moment_destinations[0],
                                        &destination_pins) == 0);
  CHECK(source_pins == 0u && destination_pins == 1u);
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  check_snapshot(builder, &snapshot);
  CHECK(et_f32_tensor_copy_plan_release_v1(&pin_plan, &error) == 0);
  CHECK(et_f32_tensor_test_plan_pins_v1(fixture.moment_destinations[0],
                                        &destination_pins) == 0);
  CHECK(destination_pins == 0u);
  CHECK(et_f32_owned_tensor_release_v1(pin_source, &error) == 0);

  et_f32_tensor *ordinary = tensor_create(1u, 0u);
  bad = fixture.request;
  bad.moment_sources[0] = ordinary;
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  CHECK(et_f32_tensor_destroy_v1(&ordinary, &error) == 0);

  et_f32_tensor *wrong_shape = owned_create(2u, 0u);
  bad = fixture.request;
  bad.moment_sources[0] = wrong_shape;
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, &bad) ==
        ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH);
  check_snapshot(builder, &snapshot);
  CHECK(et_f32_owned_tensor_release_v1(wrong_shape, &error) == 0);

  fill_bits(fixture.moment_sources[0], UINT32_C(0x7f800000));
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_NONFINITE);
  check_snapshot(builder, &snapshot);
  fill_bits(fixture.moment_sources[0], moment_bits(0u));

  fill_bits(fixture.moment_sources[1], UINT32_C(0xbf800000));
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_VALUE);
  check_snapshot(builder, &snapshot);
  fill_bits(fixture.moment_sources[1], UINT32_C(0x80000000));
  allocation_snapshot allocation = allocations_now();
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) == 0);
  check_allocations(allocation);
  CHECK(builder->tr3_c_next_index == 42u &&
        builder->tr3_c_restore_phase == ET_TR3_C_I2_RESTORE_PHASE_COMPLETE);
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_verify_restore42_v1(
            opaque, &fixture.request) == 0);
  check_allocations(allocation);
  CHECK(et_tr3_c_i2_copy_builder_verify_restore42_v1(
            opaque, (const et_tr3_c_i2_restore42_append_v1 *)(const void *)
                        builder->assignments) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);

  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &fixture.authority) == 0);
  check_allocations(allocation);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_ABORTED) == 0);
  check_allocations(allocation);
  fixture_release(&fixture);
}

static void test_value_policy_matrix(void) {
  static const uint32_t nonfinite[] = {
      UINT32_C(0x7fc00001), UINT32_C(0x7f800000), UINT32_C(0xff800000),
  };
  static const uint32_t signed_zero[] = {0u, UINT32_C(0x80000000)};
  restore_fixture fixture;
  fixture_init(&fixture);

  for (size_t model = 0u; model < ET_TR3_C_MODEL_DESTINATIONS; ++model) {
    for (size_t value = 0u; value < sizeof(nonfinite) / sizeof(nonfinite[0]);
         ++value) {
      void *opaque = fixture_builder(&fixture, model);
      et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
      builder_snapshot snapshot;
      fill_bits(fixture.model_sources[model], nonfinite[value]);
      snapshot_builder(builder, &snapshot);
      CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
                opaque, &fixture.authority, fixture.parameters[model],
                &fixture.identities[model], fixture.model_sources[model]) ==
            ET_I2_PRIVATE_OWNED_RESULT_NONFINITE);
      check_snapshot(builder, &snapshot);
      fill_bits(fixture.model_sources[model], model_bits(model));
      CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
                opaque, &fixture.authority) == 0);
    }
  }

  for (size_t moment = 0u; moment < ET_TR3_C_O2_MOMENT_ASSIGNMENTS;
       ++moment) {
    for (size_t value = 0u; value < sizeof(nonfinite) / sizeof(nonfinite[0]);
         ++value) {
      void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
      et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
      builder_snapshot snapshot;
      fill_bits(fixture.moment_sources[moment], nonfinite[value]);
      snapshot_builder(builder, &snapshot);
      CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
                opaque, &fixture.request) ==
            ET_I2_PRIVATE_OWNED_RESULT_NONFINITE);
      check_snapshot(builder, &snapshot);
      fill_bits(fixture.moment_sources[moment], moment_bits(moment));
      CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
                opaque, &fixture.authority) == 0);
    }
  }

  for (size_t parameter = 0u; parameter < ET_TR3_C_MODEL_DESTINATIONS;
       ++parameter) {
    const size_t moment = parameter * 2u + 1u;
    void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
    et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
    builder_snapshot snapshot;
    fill_bits(fixture.moment_sources[moment], UINT32_C(0xbf000000));
    snapshot_builder(builder, &snapshot);
    CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
              opaque, &fixture.request) ==
          ET_I2_PRIVATE_OWNED_RESULT_INVALID_VALUE);
    check_snapshot(builder, &snapshot);
    fill_bits(fixture.moment_sources[moment], moment_bits(moment));
    CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
              opaque, &fixture.authority) == 0);
  }

  for (size_t value = 0u; value < sizeof(signed_zero) / sizeof(signed_zero[0]);
       ++value) {
    for (size_t parameter = 0u; parameter < ET_TR3_C_MODEL_DESTINATIONS;
         ++parameter) {
      const size_t moment = parameter * 2u + 1u;
      void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
      fill_bits(fixture.moment_sources[moment], signed_zero[value]);
      CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
                opaque, &fixture.request) == 0);
      CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
                opaque, &fixture.authority) == 0);
      fill_bits(fixture.moment_sources[moment], moment_bits(moment));
    }
  }
  fixture_release(&fixture);
}

static void expect_suffix_rejection(
    void *opaque, const et_tr3_c_i2_restore42_append_v1 *request,
    int32_t expected) {
  et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
  builder_snapshot snapshot;
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(opaque, request) ==
        expected);
  check_snapshot(builder, &snapshot);
}

static void test_order_alias_and_authority_matrix(void) {
  restore_fixture fixture;
  uint64_t other_authority = UINT64_C(0x6f746865722d6932);
  fixture_init(&fixture);

  void *opaque = fixture_builder(&fixture, 0u);
  et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
  builder_snapshot snapshot;
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture.authority, fixture.parameters[0],
            &fixture.identities[0],
            et_f32_parameter_canonical_owner_v1(fixture.parameters[0])) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &fixture.authority) == 0);

  opaque = fixture_builder(&fixture, 1u);
  builder = (et_i2_copy_builder *)opaque;
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture.authority, fixture.parameters[1],
            &fixture.identities[1], fixture.model_sources[0]) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture.authority, fixture.parameters[0],
            &fixture.identities[0], fixture.model_sources[1]) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &fixture.authority) == 0);

  opaque = fixture_builder(&fixture, 0u);
  builder = (et_i2_copy_builder *)opaque;
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture.authority, fixture.parameters[1],
            &fixture.identities[1], fixture.model_sources[1]) == 0);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    if (index == 1u) {
      continue;
    }
    CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
              opaque, &fixture.authority, fixture.parameters[index],
              &fixture.identities[index], fixture.model_sources[index]) == 0);
  }
  expect_suffix_rejection(opaque, &fixture.request,
                          ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &fixture.authority) == 0);

  opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
  builder = (et_i2_copy_builder *)opaque;
  builder->assignments[ET_TR3_C_O2_FIRST_INDEX].struct_size =
      ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  check_snapshot(builder, &snapshot);
  builder->assignments[ET_TR3_C_O2_FIRST_INDEX].struct_size = 0u;

  et_tr3_c_i2_restore42_append_v1 bad = fixture.request;
  bad.moment_sources[2] = bad.moment_sources[0];
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  bad = fixture.request;
  bad.moment_sources[0] = fixture.model_sources[0];
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  bad = fixture.request;
  bad.moment_sources[0] =
      et_f32_parameter_canonical_owner_v1(fixture.parameters[0]);
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  bad = fixture.request;
  bad.moment_destinations[2] = bad.moment_destinations[0];
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  bad = fixture.request;
  bad.moment_destinations[0] = (et_f32_tensor *)
      et_f32_parameter_canonical_owner_v1(fixture.parameters[0]);
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  bad = fixture.request;
  bad.moment_destinations[0] = (et_f32_tensor *)fixture.model_sources[0];
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  bad = fixture.request;
  bad.moment_destinations[0] = (et_f32_tensor *)fixture.moment_sources[2];
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  bad = fixture.request;
  bad.moment_sources[0] = fixture.moment_destinations[2];
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  bad = fixture.request;
  bad.moment_destinations[0] = (et_f32_tensor *)fixture.moment_sources[0];
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);

  void *other = NULL;
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(&other_authority,
                                                      &other) == 0);
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &other_authority, fixture.parameters[0],
            &fixture.identities[0], fixture.model_sources[0]) ==
        ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  check_snapshot(builder, &snapshot);
  bad = fixture.request;
  bad.restore_authority = &other_authority;
  expect_suffix_rejection(opaque, &bad,
                          ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            other, &other_authority) == 0);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &fixture.authority) == 0);
  fixture_release(&fixture);
}

static void test_prepare_commit(void) {
  restore_fixture fixture;
  fixture_init(&fixture);
  void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
  et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) == 0);
  builder_snapshot snapshot;
  snapshot_builder(builder, &snapshot);

  for (size_t allowed = 0u; allowed < 2u; ++allowed) {
    et_f32_tensor_test_fail_alloc_after_v1(allowed);
    CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(
              opaque, &fixture.authority) ==
          ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED);
    check_snapshot(builder, &snapshot);
    CHECK(builder->plan == NULL);
    CHECK(builder->tr3_c_restore_phase == ET_TR3_C_I2_RESTORE_PHASE_COMPLETE);
    check_fixture_plan_pins(&fixture, 0u);
  }
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(
            opaque, (const void *)(uintptr_t)1u) ==
        ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT);
  CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(
            opaque, &fixture.authority) == 0);
  CHECK(builder->plan != NULL &&
        builder->tr3_c_restore_phase == ET_TR3_C_I2_RESTORE_PHASE_PREPARED);
  snapshot_builder(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture.authority, fixture.parameters[0],
            &fixture.identities[0], fixture.model_sources[0]) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  check_snapshot(builder, &snapshot);
  CHECK(et_tr3_c_i2_copy_builder_verify_restore42_v1(
            opaque, &fixture.request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  CHECK(et_i2_private_copy_builder_abort_v1(opaque) == -1);
  allocation_snapshot allocation = allocations_now();
  CHECK(et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
            opaque, &fixture.authority) == 0);
  check_allocations(allocation);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_COMMITTED) == 0);
  check_allocations(allocation);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_ABORTED) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    const et_f32_tensor *value =
        et_f32_parameter_canonical_owner_v1(fixture.parameters[index]);
    check_bits_equal(value, fixture.model_sources[index]);
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    check_bits_equal(fixture.moment_destinations[index],
                     fixture.moment_sources[index]);
  }
  fixture_release(&fixture);
}

static void test_prepared_abort(void) {
  restore_fixture fixture;
  fixture_init(&fixture);
  void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture.request) == 0);
  CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(
            opaque, &fixture.authority) == 0);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &fixture.authority) == 0);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_ABORTED) == 0);
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    check_all_bits(fixture.moment_destinations[index], 0u);
  }
  fixture_release(&fixture);
}

static void exercise_tombstone_rejections(restore_fixture *fixture,
                                          void *opaque) {
  et_i2_copy_builder *builder = (et_i2_copy_builder *)opaque;
  const et_i2_copy_builder snapshot = *builder;
  CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
            opaque, &fixture->authority, fixture->parameters[0],
            &fixture->identities[0], fixture->model_sources[0]) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(memcmp(&snapshot, builder, sizeof(*builder)) == 0);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &fixture->request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(memcmp(&snapshot, builder, sizeof(*builder)) == 0);
  CHECK(et_tr3_c_i2_copy_builder_verify_restore42_v1(
            opaque, &fixture->request) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(
            opaque, &fixture->authority) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
            opaque, &fixture->authority) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &fixture->authority) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_i2_private_copy_builder_set_v1(opaque, 0, NULL, 0, NULL, NULL, 0,
                                          NULL) == -1);
  CHECK(et_i2_private_copy_builder_prepare_v1(opaque) == -1);
  CHECK(et_i2_private_copy_builder_commit_v1(opaque) == -1);
  CHECK(et_i2_private_copy_builder_abort_v1(opaque) == -1);
  CHECK(memcmp(&snapshot, builder, sizeof(*builder)) == 0);
}

static void test_terminal_rejections(void) {
  restore_fixture aborted;
  fixture_init(&aborted);
  void *opaque = fixture_builder(&aborted, ET_TR3_C_MODEL_DESTINATIONS);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
            opaque, &aborted.authority) == 0);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_ABORTED) == 0);
  exercise_tombstone_rejections(&aborted, opaque);
  fixture_release(&aborted);

  restore_fixture committed;
  fixture_init(&committed);
  opaque = fixture_builder(&committed, ET_TR3_C_MODEL_DESTINATIONS);
  CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(
            opaque, &committed.request) == 0);
  CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(
            opaque, &committed.authority) == 0);
  CHECK(et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
            opaque, &committed.authority) == 0);
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            opaque, ET_TR3_C_I2_TERMINAL_COMMITTED) == 0);
  exercise_tombstone_rejections(&committed, opaque);
  fixture_release(&committed);
}

static void test_abort_prefixes(void) {
  restore_fixture fixture;
  fixture_init(&fixture);
  for (size_t prefix = 0u; prefix <= ET_TR3_C_MODEL_DESTINATIONS; ++prefix) {
    void *opaque = fixture_builder(&fixture, prefix);
    CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
              opaque, &fixture.authority) == 0);
    CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
              opaque, ET_TR3_C_I2_TERMINAL_ABORTED) == 0);
  }
  fixture_release(&fixture);
}

static void test_ordinary_builder_isolation(void) {
  et_f32_tensor_error error;
  et_f32_tensor *destination = tensor_create(1u, 0u);
  et_f32_tensor *source = owned_create(1u, UINT32_C(0x3f800000));
  void *builder = et_i2_private_copy_builder_create_v1(1);
  CHECK(builder != NULL);
  CHECK(et_i2_private_copy_builder_set_v1(
            builder, 0, destination, ET_I2_CARRIER_TENSOR, NULL, source,
            ET_I2_CARRIER_OWNED_CLONE, NULL) == 0);
  CHECK(et_i2_private_copy_builder_prepare_v1(builder) == 0);
  CHECK(et_tr3_c_i2_copy_builder_commit_checked_v1(builder) == 0);
  CHECK(get_bits(destination) == UINT32_C(0x3f800000));
  CHECK(et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
            builder, ET_TR3_C_I2_TERMINAL_COMMITTED) ==
        ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
  CHECK(et_f32_owned_tensor_release_v1(source, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&destination, &error) == 0);

  for (int64_t count = 41; count <= 43; ++count) {
    builder = et_i2_private_copy_builder_create_v1(count);
    CHECK(builder != NULL);
    et_i2_copy_builder shell = *(et_i2_copy_builder *)builder;
    et_f32_tensor_copy_assignment_v1 assignments[43];
    memcpy(assignments, ((et_i2_copy_builder *)builder)->assignments,
           (size_t)count * sizeof(assignments[0]));
    CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
              builder, &builder, NULL, NULL, NULL) ==
          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
    CHECK(memcmp(&shell, builder, sizeof(shell)) == 0);
    CHECK(memcmp(assignments, ((et_i2_copy_builder *)builder)->assignments,
                 (size_t)count * sizeof(assignments[0])) == 0);
    CHECK(et_tr3_c_i2_copy_builder_append_restore42_v1(builder, NULL) ==
          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
    CHECK(memcmp(&shell, builder, sizeof(shell)) == 0);
    CHECK(memcmp(assignments, ((et_i2_copy_builder *)builder)->assignments,
                 (size_t)count * sizeof(assignments[0])) == 0);
    CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(builder, &builder) ==
          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
    CHECK(et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
              builder, &builder) ==
          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
    CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
              builder, &builder) ==
          ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT);
    CHECK(et_i2_private_copy_builder_abort_v1(builder) == 0);
  }
}

static void expect_exit_134(void (*action)(void)) {
  int status = 0;
  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    action();
    _Exit(90);
  }
  if (child < 0) {
    return;
  }
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 134);
}

static void fail_restore_commit_child(void) {
  restore_fixture fixture;
  fixture_init(&fixture);
  void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
  if (et_tr3_c_i2_copy_builder_append_restore42_v1(
          opaque, &fixture.request) != 0 ||
      et_tr3_c_i2_copy_builder_prepare_restore42_v1(
          opaque, &fixture.authority) != 0) {
    _Exit(91);
  }
  et_tr3_c_i2_test_fail_copy_commit_v1(1);
  (void)et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
      opaque, &fixture.authority);
}

static void fail_restore_commit_release(void) {
  restore_fixture fixture;
  fixture_init(&fixture);
  void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
  if (et_tr3_c_i2_copy_builder_append_restore42_v1(
          opaque, &fixture.request) != 0 ||
      et_tr3_c_i2_copy_builder_prepare_restore42_v1(
          opaque, &fixture.authority) != 0) {
    _Exit(91);
  }
  et_tr3_c_i2_test_fail_copy_release_v1(1);
  (void)et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
      opaque, &fixture.authority);
}

static void fail_restore_abort_release(void) {
  restore_fixture fixture;
  fixture_init(&fixture);
  void *opaque = fixture_builder(&fixture, ET_TR3_C_MODEL_DESTINATIONS);
  if (et_tr3_c_i2_copy_builder_append_restore42_v1(
          opaque, &fixture.request) != 0 ||
      et_tr3_c_i2_copy_builder_prepare_restore42_v1(
          opaque, &fixture.authority) != 0) {
    _Exit(91);
  }
  et_tr3_c_i2_test_fail_copy_release_v1(1);
  (void)et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
      opaque, &fixture.authority);
}

static void test_fail_stop_tail(void) {
  expect_exit_134(fail_restore_commit_child);
  expect_exit_134(fail_restore_commit_release);
  expect_exit_134(fail_restore_abort_release);
}

static void test_retained_slope(void) {
  size_t copy_before = 0u;
  size_t reset_before = 0u;
  size_t decode_before = 0u;
  size_t bytes_before = 0u;
  size_t copy_after = 0u;
  size_t reset_after = 0u;
  size_t decode_after = 0u;
  size_t bytes_after = 0u;
  uint64_t authority = UINT64_C(0x736c6f70652d6932);
  et_i2_test_retired_builder_counts_v1(
      &copy_before, &reset_before, &decode_before, &bytes_before);
  for (size_t index = 0u; index < 1024u; ++index) {
    void *builder = NULL;
    CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(&authority, &builder) ==
          0);
    CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
              builder, &authority) == 0);
  }
  et_i2_test_retired_builder_counts_v1(
      &copy_after, &reset_after, &decode_after, &bytes_after);
  CHECK(copy_after - copy_before == 1024u);
  CHECK(reset_after == reset_before && decode_after == decode_before);
  CHECK(bytes_after - bytes_before == 1024u * sizeof(et_i2_copy_builder));
  (void)printf("TR3-C I2 retained-control slope: %zu bytes/restore\n",
               sizeof(et_i2_copy_builder));
}

int main(void) {
  test_matcher();
  test_create_and_prefix();
  test_suffix_failures();
  test_value_policy_matrix();
  test_order_alias_and_authority_matrix();
  test_prepare_commit();
  test_prepared_abort();
  test_terminal_rejections();
  test_abort_prefixes();
  test_ordinary_builder_isolation();
  test_fail_stop_tail();
  test_retained_slope();
  if (failures != 0) {
    (void)fprintf(stderr, "TR3-C I2 restore FAIL: %d/%d checks\n", failures,
                  checks);
    return 1;
  }
  (void)printf("TR3-C I2 restore PASS: %d checks\n", checks);
  return 0;
}
