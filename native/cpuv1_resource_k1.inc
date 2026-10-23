#ifdef ET_CPUV1_RESOURCE_TESTING
/* The caller supplies only direct, already-published owner slots. Duplicate
 * runtime references are expected; allocation bases within a runtime are not. */
void et_cpuv1_test_census_k1_v1(et_cpuv1_resource_packet_v1 *p,
                               et_kernel_runtime *const *slots,size_t count) {
  et_cpuv1_test_resource_layout_v1(p,ET_R_K1,0,sizeof(et_kernel_runtime),_Alignof(et_kernel_runtime));
  et_cpuv1_test_resource_layout_v1(p,ET_R_K1,1,sizeof(et_kernel_capability_v1),_Alignof(et_kernel_capability_v1));
  et_cpuv1_test_resource_layout_v1(p,ET_R_K1,2,sizeof(char),_Alignof(char));
  et_cpuv1_test_resource_layout_v1(p,ET_R_K1,3,sizeof(const char *),_Alignof(const char *));
  et_cpuv1_test_resource_layout_v1(p,ET_R_K1,4,sizeof(et_kernel_shape_range_v1),_Alignof(et_kernel_shape_range_v1));
  et_cpuv1_test_resource_layout_v1(p,ET_R_K1,5,sizeof(et_kernel_dimension_range_v1),_Alignof(et_kernel_dimension_range_v1));

  if (!slots || count!=7u) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);return;
  }
#define K1_ADD(K,B,N,S,A) do { \
  if (et_cpuv1_test_resource_add_v1(p,ET_R_K1,K,B,N,S,ET_R_LIVE,A)) return; \
} while (0)
#define K1_TEXT(B) do { const char *text__=(B); \
  if (text__) K1_ADD(2,text__,strlen(text__)+1u,1u,1u); \
} while (0)
  for (size_t slot=0;slot<count;slot++) {
    et_kernel_runtime *runtime=slots[slot];
    if (!runtime) continue;
    size_t earlier=0;
    for (;earlier<slot;earlier++) if (slots[earlier]==runtime) break;
    if (earlier<slot) continue;
    K1_ADD(0,runtime,1u,sizeof(*runtime),_Alignof(et_kernel_runtime));
    K1_ADD(1,runtime->capabilities,runtime->capability_count,
           sizeof(*runtime->capabilities),_Alignof(et_kernel_capability_v1));
    if (runtime->capability_count && !runtime->capabilities) {
      et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
    }
    for (size_t i=0;i<runtime->capability_count;i++) {
      const et_kernel_capability_v1 *c=&runtime->capabilities[i];
      K1_TEXT(c->name);K1_TEXT(c->implementation);
      K1_TEXT(c->version);K1_TEXT(c->evidence);
      const char *const *arrays[]={c->operations,c->dtypes,c->devices};
      size_t counts[]={c->operation_count,c->dtype_count,c->device_count};
      for (size_t a=0;a<3;a++) {
        K1_ADD(3,arrays[a],counts[a],sizeof(*arrays[a]),_Alignof(const char *));
        if (counts[a] && !arrays[a]) {
          et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
        }
        for (size_t j=0;j<counts[a];j++) K1_TEXT(arrays[a][j]);
      }
      K1_ADD(4,c->shape_ranges,c->shape_range_count,sizeof(*c->shape_ranges),
             _Alignof(et_kernel_shape_range_v1));
      if (c->shape_range_count && !c->shape_ranges) {
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
      }
      for (size_t j=0;j<c->shape_range_count;j++) {
        const et_kernel_shape_range_v1 *s=&c->shape_ranges[j];
        K1_ADD(5,s->dimensions,s->rank,sizeof(*s->dimensions),
               _Alignof(et_kernel_dimension_range_v1));
        if (s->rank && !s->dimensions) {
          et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
        }
      }
    }
  }
#undef K1_TEXT
#undef K1_ADD
  et_cpuv1_test_resource_domain_v1(p,ET_R_K1);
}

/* Bounded unit fixture for the existing partial cleanup algorithm. It does
 * not simulate discover/OOM: eight pointer slots own only three real strings. */
int et_cpuv1_test_k1_partial_cleanup_v1(void) {
  const size_t allocated_count=8u,initialized_count=3u;
  const char **values=(const char **)ET_R_ALLOC(ET_R_K1,3,
      calloc(allocated_count,sizeof(*values)),allocated_count,sizeof(*values));
  if (!values) return ET_R_UNSUPPORTED;
  size_t initialized=0;
  for (;initialized<initialized_count;initialized++) {
    values[initialized]=duplicate_text("cpuv1-partial");
    if (!values[initialized]) break;
  }
  free_string_array(initialized,values ET_R_EXTRA_ARG(allocated_count));
  /* Metadata must not be evaluated for NULL, just as original free(NULL). */
  const char *absent=NULL;
  ET_R_FREE(ET_R_K1,2,(void *)absent,strlen(absent)+1u,1u);
  return initialized==initialized_count ? ET_R_OK : ET_R_UNSUPPORTED;
}
#endif
