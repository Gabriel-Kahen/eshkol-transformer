#ifndef AMD4_LOADER_FIXTURE_H
#define AMD4_LOADER_FIXTURE_H

#include <stddef.h>
#include <stdint.h>

/* Shared by the fixture DSOs through the test executable's dynamic symbols. */
typedef struct amd4_loader_fixture_state {
  int32_t count;
  int32_t active;
  int32_t memory_code;
  int32_t create_code;
  int32_t destroy_code;
  int32_t restore_code;
  size_t event_count;
  char events[64];
} amd4_loader_fixture_state;

extern amd4_loader_fixture_state amd4_loader_fixture;

void test_amd4_loader_fixture(const char *scenario);

#endif
