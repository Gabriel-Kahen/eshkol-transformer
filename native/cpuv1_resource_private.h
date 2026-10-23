#ifndef ET_CPUV1_RESOURCE_PRIVATE_H
#define ET_CPUV1_RESOURCE_PRIVATE_H

/* Noninstalled test producer. OFF expands to the original allocator call. */
#ifdef ET_CPUV1_RESOURCE_TESTING
#include <stddef.h>
#include <stdint.h>
#if defined(ET_CPU4_C4_HISTORY_PRIVATE) || defined(ET_G3R_COMMITTED_HISTORY_PRIVATE) || defined(ET_G3R_SAME_OWNER_SAVE_PRIVATE)
#error "CPUv1 resource producer is frozen to history/capture OFF"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum et_cpuv1_resource_domain {
  ET_R_F32, ET_R_P1, ET_R_C4, ET_R_TRAIN, ET_R_E3, ET_R_O2,
  ET_R_I64_T1, ET_R_GENERATION, ET_R_D2, ET_R_C2_IO, ET_R_K1,
  ET_R_SCHEME, ET_R_RUNTIME, ET_R_DOMAINS
};
enum { ET_R_KINDS = 64, ET_R_PACKET_VERSION = 1 };
enum et_cpuv1_resource_status {
  ET_R_OK, ET_R_ARGUMENT, ET_R_OVERFLOW, ET_R_UNSUPPORTED,
  ET_R_COVERAGE, ET_R_MISMATCH, ET_R_SCRATCH, ET_R_NOT_QUIESCENT
};
enum et_cpuv1_resource_state {
  ET_R_LIVE, ET_R_PENDING, ET_R_PREPARED, ET_R_BUSY, ET_R_DEAD
};
/* Kind numbers are scoped by domain. A byte buffer is an allocation base,
 * not a reference from its parent. All concrete kinds are in kinds.json. */
typedef struct et_cpuv1_resource_row_v1 {
  uint64_t created, freed, allocated_requested, freed_requested;
  uint64_t allocated_usable, freed_usable, failed;
  uint64_t bases, requested, usable;
  uint64_t live, pending, prepared, busy, dead;
  uint64_t control_size, control_alignment;
} et_cpuv1_resource_row_v1;

typedef struct et_cpuv1_resource_packet_v1 {
  uint64_t struct_size, version, status, coverage;
  uint64_t observer_static_bytes, observer_packet_bytes;
  uint64_t scheme_ready, reserved;
  et_cpuv1_resource_row_v1 rows[ET_R_DOMAINS][ET_R_KINDS];
  /* Semantic/reference counts, never extra allocation bases or byte charges. */
  uint64_t details[ET_R_DOMAINS][64];
  /* Runtime values are named in runtime-fields.json; scalar words only. */
  uint64_t runtime[64];
} et_cpuv1_resource_packet_v1;

void *et_cpuv1_test_alloc_event_v1(unsigned domain, unsigned kind, void *base,
                                 size_t count, size_t size);
void et_cpuv1_test_free_event_v1(unsigned domain, unsigned kind, void *base,
                               size_t count, size_t size);
void *et_cpuv1_test_realloc_event_v1(unsigned domain, unsigned kind, void *base,
                                   size_t old_count, size_t new_count,
                                   size_t size);
int et_cpuv1_test_resource_begin_v1(et_cpuv1_resource_packet_v1 *packet);
int et_cpuv1_test_resource_finish_v1(et_cpuv1_resource_packet_v1 *packet);
int et_cpuv1_test_resource_add_v1(et_cpuv1_resource_packet_v1 *packet,
                                unsigned domain, unsigned kind,
                                const void *base, size_t count, size_t size,
                                unsigned state, size_t alignment);
void et_cpuv1_test_resource_fail_v1(et_cpuv1_resource_packet_v1 *packet,
                                  unsigned status);
void et_cpuv1_test_resource_domain_v1(et_cpuv1_resource_packet_v1 *packet,
                                    unsigned domain);
/* Test mutation only; absence/mismatch must never produce COMPLETE. */
void et_cpuv1_test_resource_mutation_v1(unsigned mutation);

#ifdef __cplusplus
}
#endif
#define ET_R_ALLOC(D,K,E,N,S) \
  et_cpuv1_test_alloc_event_v1((D),(K),(E),(N),(S))
#define ET_R_FREE(D,K,P,N,S) do { \
  void *et_cpuv1_base__=(void *)(P); \
  if (et_cpuv1_base__) et_cpuv1_test_free_event_v1((D),(K),et_cpuv1_base__,(N),(S)); \
  free(et_cpuv1_base__); \
} while (0)
#define ET_R_REALLOC(D,K,E,P,O,N,S) \
  et_cpuv1_test_realloc_event_v1((D),(K),(P),(O),(N),(S))
#else
#define ET_R_ALLOC(D,K,E,N,S) E
#define ET_R_FREE(D,K,P,N,S) free(P)
#define ET_R_REALLOC(D,K,E,P,O,N,S) E
#endif
#endif
