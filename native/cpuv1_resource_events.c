#include "cpuv1_resource_private.h"
#ifdef ET_CPUV1_RESOURCE_TESTING
#include <errno.h>
#include <limits.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#if defined(__has_feature)
# if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer) || __has_feature(thread_sanitizer)
#  define ET_R_ALTERNATE_PROFILE 1
# endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
# define ET_R_ALTERNATE_PROFILE 1
#endif
#if !defined(ET_CPUV1_RESOURCE_NORMAL_GLIBC_PROFILE) || !defined(__GLIBC__) || !defined(__linux__)
# define ET_R_ALTERNATE_PROFILE 1
#endif

/* Aggregate scalars only. No table of pointers, owner fields or allocator
 * interposition. BSS is reported separately, never malloc_usable_size'd. */
static et_cpuv1_resource_row_v1 events[ET_R_DOMAINS][ET_R_KINDS];
static unsigned event_error;
static unsigned mutation;
static et_cpuv1_resource_packet_v1 observer_packets[2];
/* Snapshot-only duplicate/cycle scratch, not an allocation tracking registry.
 * BSS is preallocated before P0, reported separately and never usable-queried. */
enum { ET_R_BASE_SLOTS = 1u << 20, ET_R_BASE_LIMIT = ET_R_BASE_SLOTS / 2 };
static uintptr_t snapshot_bases[ET_R_BASE_SLOTS];
static size_t snapshot_base_count;
static et_cpuv1_resource_packet_v1 *snapshot_packet;

static int unique_base(const void *base) {
  uintptr_t word=(uintptr_t)base;
  size_t slot=(size_t)((word >> 3) * UINT64_C(11400714819323198485)) &
              (ET_R_BASE_SLOTS-1u);
  if (snapshot_base_count>=ET_R_BASE_LIMIT) return ET_R_SCRATCH;
  for (size_t i=0;i<ET_R_BASE_SLOTS;i++) {
    if (snapshot_bases[slot]==word) return ET_R_MISMATCH;
    if (!snapshot_bases[slot]) {
      snapshot_bases[slot]=word; snapshot_base_count++; return ET_R_OK;
    }
    slot=(slot+1u)&(ET_R_BASE_SLOTS-1u);
  }
  return ET_R_SCRATCH;
}

static const unsigned kind_count[ET_R_DOMAINS]={23,5,13,11,5,22,6,30,4,6,6,64,8};
static int valid_kind(unsigned d,unsigned k) {
  return d<ET_R_DOMAINS && k<kind_count[d] &&
         !(d==ET_R_GENERATION && k>=9 && k<16);
}
static int valid(unsigned d,unsigned k) {
  return valid_kind(d,k) && d!=ET_R_SCHEME &&
         (d!=ET_R_RUNTIME || k==6);
}
static int bytes(size_t n, size_t s, uint64_t *out) {
  if (s && n > SIZE_MAX / s) return 0;
  *out = (uint64_t)(n * s);
  return (size_t)*out == n * s;
}
static int add(uint64_t *value, uint64_t amount) {
  if (*value > UINT64_MAX - amount) return 0;
  *value += amount;
  return 1;
}
static void note(unsigned d, unsigned k, uint64_t requested,
                 uint64_t usable, int freed) {
  if (snapshot_packet) event_error=ET_R_NOT_QUIESCENT;
  if (!valid(d,k)) { event_error = ET_R_COVERAGE; return; }
  et_cpuv1_resource_row_v1 *r = &events[d][k];
  if (freed) {
    if (r->freed >= r->created ||
        requested > r->allocated_requested - r->freed_requested ||
        usable > r->allocated_usable - r->freed_usable) {
      event_error = ET_R_MISMATCH; return;
    }
    if (!add(&r->freed,1) || !add(&r->freed_requested,requested) ||
        !add(&r->freed_usable,usable)) event_error = ET_R_OVERFLOW;
  } else if (!add(&r->created,1) ||
             !add(&r->allocated_requested,requested) ||
             !add(&r->allocated_usable,usable)) event_error = ET_R_OVERFLOW;
}
static void *alloc_event(unsigned d, unsigned k, void *p,
                                 size_t n, size_t s) {
  uint64_t requested;
  if (snapshot_packet) event_error=ET_R_NOT_QUIESCENT;
  if (!valid(d,k)) { event_error = ET_R_COVERAGE; return p; }
  if (!bytes(n,s,&requested)) { event_error = ET_R_OVERFLOW; return p; }
  if (!p) {
    if (!add(&events[d][k].failed,1)) event_error = ET_R_OVERFLOW;
    return p;
  }
#ifndef ET_R_ALTERNATE_PROFILE
  size_t usable = malloc_usable_size(p);
  if (usable < requested) event_error = ET_R_MISMATCH;
  note(d,k,requested,(uint64_t)usable,0);
#else
  event_error = ET_R_UNSUPPORTED;
#endif
  return p;
}
static void free_event(unsigned d, unsigned k, void *p,
                               size_t n, size_t s) {
  uint64_t requested;
  if (!p) return;
  if (!bytes(n,s,&requested)) { event_error = ET_R_OVERFLOW; return; }
#ifndef ET_R_ALTERNATE_PROFILE
  /* Read only while the exact allocation base is still allocated. */
  note(d,k,requested,(uint64_t)malloc_usable_size(p),1);
#else
  (void)d; (void)k; event_error = ET_R_UNSUPPORTED;
#endif
}
/* Bookkeeping must preserve the actual allocator/IO failure errno. */
void *et_cpuv1_test_alloc_event_v1(unsigned d,unsigned k,void *p,size_t n,size_t z) {
  int saved_errno=errno;
  void *result=alloc_event(d,k,p,n,z);
  errno=saved_errno;
  return result;
}
void et_cpuv1_test_free_event_v1(unsigned d,unsigned k,void *p,size_t n,size_t z) {
  int saved_errno=errno;
  free_event(d,k,p,n,z);
  errno=saved_errno;
}
void *et_cpuv1_test_realloc_event_v1(unsigned d, unsigned k, void *p,
                                   size_t old_n, size_t new_n, size_t s) {
  uint64_t old_bytes, new_bytes;
  if (!bytes(old_n,s,&old_bytes) || !bytes(new_n,s,&new_bytes) ||
      !valid(d,k)) { event_error = ET_R_OVERFLOW; return NULL; }
#ifndef ET_R_ALTERNATE_PROFILE
  int incoming_errno=errno;
  uint64_t old_usable = p ? (uint64_t)malloc_usable_size(p) : 0;
  int had_old = p != NULL;
  errno=incoming_errno;
  void *result = realloc(p,(size_t)new_bytes);
  int allocator_errno=errno;
  /* On normal glibc realloc(p,0) releases p. Failure at positive size leaves
   * its exact old allocation owned, including same-address successful cases. */
  if (!result && new_bytes != 0) {
    if (!add(&events[d][k].failed,1)) event_error = ET_R_OVERFLOW;
    errno=allocator_errno;
    return NULL;
  }
  if (had_old) note(d,k,old_bytes,old_usable,1);
  if (result) note(d,k,new_bytes,(uint64_t)malloc_usable_size(result),0);
  errno=allocator_errno;
  return result;
#else
  event_error = ET_R_UNSUPPORTED;
  return NULL;
#endif
}
void et_cpuv1_test_resource_fail_v1(et_cpuv1_resource_packet_v1 *p,unsigned s) {
  if (p && p->status == ET_R_OK) p->status = s;
}
void et_cpuv1_test_resource_domain_v1(et_cpuv1_resource_packet_v1 *p,unsigned d) {
  if (!p || d >= ET_R_DOMAINS) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE); return;
  }
  p->coverage |= UINT64_C(1) << d;
}
int et_cpuv1_test_resource_begin_v1(et_cpuv1_resource_packet_v1 *p) {
  if (!p || p->struct_size != sizeof(*p) ||
      p->version != ET_R_PACKET_VERSION) return ET_R_ARGUMENT;
  if (snapshot_packet) return ET_R_NOT_QUIESCENT;
  memset(snapshot_bases,0,sizeof(snapshot_bases)); snapshot_base_count=0;
  snapshot_packet=p;
  if(mutation==3)snapshot_base_count=ET_R_BASE_LIMIT;
  memset(p,0,sizeof(*p)); p->struct_size=sizeof(*p);
  p->version=ET_R_PACKET_VERSION; p->status=event_error;
  p->observer_static_bytes=sizeof(events)+sizeof(event_error)+sizeof(mutation)+sizeof(observer_packets)+
    sizeof(snapshot_bases)+sizeof(snapshot_base_count)+sizeof(snapshot_packet)+sizeof(kind_count)+
    et_cpuv1_test_runtime_static_bytes_v1();
  p->observer_packet_bytes=sizeof(*p);
#ifdef ET_R_ALTERNATE_PROFILE
  p->status=ET_R_UNSUPPORTED;
#endif
  for (unsigned d=0;d<ET_R_DOMAINS;d++)
    for (unsigned k=0;k<ET_R_KINDS;k++) p->rows[d][k]=events[d][k];
  return (int)p->status;
}
/* Owning TUs publish all applicable layout facts even with no live bases. */
void et_cpuv1_test_resource_layout_v1(et_cpuv1_resource_packet_v1 *p,
 unsigned d,unsigned k,size_t size,size_t alignment) {
  if (!p || p!=snapshot_packet || !valid_kind(d,k) || d==ET_R_SCHEME ||
      !size || !alignment || (alignment&(alignment-1u))) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_ARGUMENT);return;
  }
  et_cpuv1_resource_row_v1 *r=&p->rows[d][k];
  if ((r->control_size && r->control_size!=size) ||
      (r->control_alignment && r->control_alignment!=alignment)) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
  }
  r->control_size=size;r->control_alignment=alignment;
}
int et_cpuv1_test_resource_add_v1(et_cpuv1_resource_packet_v1 *p,
 unsigned d,unsigned k,const void *base,size_t n,size_t s,unsigned state,
 size_t alignment) {
  uint64_t requested;
  if (!p || p!=snapshot_packet || (!valid_kind(d,k) || d==ET_R_SCHEME) || state>ET_R_DEAD ||
      !alignment || (alignment&(alignment-1u)) ||
      !bytes(n,s,&requested)) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_ARGUMENT); return ET_R_ARGUMENT;
  }
  if (!base) return ET_R_OK;
  if ((uintptr_t)base%alignment) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  int unique=unique_base(base);
  if (unique!=ET_R_OK) {
    et_cpuv1_test_resource_fail_v1(p,(unsigned)unique); return unique;
  }
#ifndef ET_R_ALTERNATE_PROFILE
  et_cpuv1_resource_row_v1 *r=&p->rows[d][k];
  uint64_t usable=(uint64_t)malloc_usable_size((void *)base);
  uint64_t *states[]={&r->live,&r->pending,&r->prepared,&r->busy,&r->dead};
  if (usable<requested || !add(&r->bases,1) || !add(&r->requested,requested) ||
      !add(&r->usable,usable) || !add(states[state],1))
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);
  if ((r->control_size && r->control_size!=s) ||
      (r->control_alignment && r->control_alignment!=alignment))
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);
  r->control_size=s; r->control_alignment=alignment;
#else
  et_cpuv1_test_resource_fail_v1(p,ET_R_UNSUPPORTED);
#endif
  return (int)p->status;
}
int et_cpuv1_test_resource_finish_v1(et_cpuv1_resource_packet_v1 *p) {
  if (!p || p!=snapshot_packet || p->struct_size!=sizeof(*p) || p->version!=ET_R_PACKET_VERSION)
    return ET_R_ARGUMENT;
  et_cpuv1_test_resource_fail_v1(p,event_error);
  if (mutation==1) p->rows[ET_R_F32][0].bases++;
  if (mutation==2) p->coverage &= ~(UINT64_C(1)<<ET_R_F32);
  for (unsigned d=0;d<ET_R_DOMAINS;d++) {
    if (d==ET_R_SCHEME) continue;
    for (unsigned k=0;k<ET_R_KINDS;k++) {
      et_cpuv1_resource_row_v1 *r=&p->rows[d][k];
      /* Arena backing/metadata are census-only; allocated observer buffers
       * retain their ordinary independent event conservation. */
      if (d==ET_R_RUNTIME && k!=6) continue;
      uint64_t states=0;
      if (!add(&states,r->live) || !add(&states,r->pending) ||
          !add(&states,r->prepared) || !add(&states,r->busy) ||
          !add(&states,r->dead) || states!=r->bases)
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);
      if (r->freed>r->created || r->freed_requested>r->allocated_requested ||
          r->freed_usable>r->allocated_usable ||
          r->created-r->freed!=r->bases ||
          r->allocated_requested-r->freed_requested!=r->requested ||
          r->allocated_usable-r->freed_usable!=r->usable)
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);
    }
  }
  if (!p->scheme_ready ||
      p->coverage != (UINT64_C(1)<<ET_R_DOMAINS)-1)
    et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);
  memset(snapshot_bases,0,sizeof(snapshot_bases));snapshot_base_count=0;
  snapshot_packet=NULL;
  return (int)p->status;
}
void *et_cpuv1_test_resource_packet_slot_v1(int64_t slot) {
  if(slot<0 || slot>=2 || snapshot_packet) return NULL;
  et_cpuv1_resource_packet_v1 *p=&observer_packets[slot];
  p->struct_size=sizeof(*p);p->version=ET_R_PACKET_VERSION;
  return p;
}
int64_t et_cpuv1_test_scheme_reject_v1(void *raw) {
  et_cpuv1_test_resource_fail_v1(raw,ET_R_NOT_QUIESCENT);return ET_R_NOT_QUIESCENT;
}
int64_t et_cpuv1_test_scheme_value_v1(void *raw,int64_t index,
                                    int64_t lexical,int64_t kind,int64_t count) {
  et_cpuv1_resource_packet_v1 *p=raw;
  if(!p || p!=snapshot_packet || index<0 || index>=64 ||
     lexical<0 || lexical>1 || kind<1 || kind>7 || count<0) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_ARGUMENT);return ET_R_ARGUMENT;
  }
  if(lexical) {
    if(index>=51 || p->scheme_lexical[index][0]) {
      et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);return ET_R_COVERAGE;
    }
    p->scheme_lexical[index][0]=1u;p->scheme_lexical[index][1]=(uint64_t)kind;
    p->scheme_lexical[index][2]=(uint64_t)count;
  } else {
    if(p->details[ET_R_SCHEME][index]) {
      et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);return ET_R_COVERAGE;
    }
    p->details[ET_R_SCHEME][index]=(uint64_t)kind;
    p->rows[ET_R_SCHEME][index].live=(uint64_t)count;
  }
  return (int64_t)p->status;
}
/* Only concrete owning-scope visitors call this after schema/identity checks.
 * This is a semantic census, never another native allocation or byte charge. */
int64_t et_cpuv1_test_scheme_state_v1(void *raw,int64_t index,
                                     int64_t state,int64_t links,int64_t source_state) {
  et_cpuv1_resource_packet_v1 *p=raw;
  if(!p || p!=snapshot_packet || index<0 || index>=65 ||
     state<1 || state>6 || links<0 || source_state<0 || source_state>=15 ||
     (index<64 && (p->scheme_classified&(UINT64_C(1)<<index)))) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);return ET_R_COVERAGE;
  }
  uint64_t *states=index==64?p->scheme_candidate:p->scheme_states[index];
  if(!add(&p->scheme_source[index][source_state],1) ||
     !add(&states[0],1) || !add(&states[state],1) ||
     !add(&states[7],(uint64_t)links))
    et_cpuv1_test_resource_fail_v1(p,ET_R_OVERFLOW);
  return (int64_t)p->status;
}
int64_t et_cpuv1_test_scheme_seal_v1(void *raw,int64_t index,int64_t count) {
  et_cpuv1_resource_packet_v1 *p=raw;
  if(p && p==snapshot_packet && index==64 && count>=0) {
    if(p->scheme_candidate_ready || p->scheme_candidate[0]!=(uint64_t)count)
      et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);
    if(p->status==ET_R_OK)p->scheme_candidate_ready=1;
    return (int64_t)p->status;
  }
  if(!p || p!=snapshot_packet || index<0 || index>=64 || count<0 ||
     !p->details[ET_R_SCHEME][index] ||
     (p->scheme_classified&(UINT64_C(1)<<index)) ||
     p->scheme_states[index][0]!=(uint64_t)count) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);return ET_R_COVERAGE;
  }
  if(p->status==ET_R_OK)p->scheme_classified|=UINT64_C(1)<<index;
  return (int64_t)p->status;
}
int64_t et_cpuv1_test_scheme_scalar_v1(void *raw,int64_t index,
                                     int64_t lexical,int64_t value) {
  et_cpuv1_resource_packet_v1 *p=raw;
  if(!p || p!=snapshot_packet || index<0 || index>=64 ||
     lexical<0 || lexical>1 || value<0) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_ARGUMENT);return ET_R_ARGUMENT;
  }
  if(lexical)p->scheme_lexical[index][3]=(uint64_t)value;
  else p->scheme_scalar[index]=(uint64_t)value;
  return (int64_t)p->status;
}
int64_t et_cpuv1_test_scheme_finish_v1(void *raw) {
  et_cpuv1_resource_packet_v1 *p=raw;
  if(!p || p!=snapshot_packet)return ET_R_ARGUMENT;
  for(unsigned i=0;i<64;i++)if(!p->details[ET_R_SCHEME][i])
    et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);
  for(unsigned i=0;i<51;i++)if(!p->scheme_lexical[i][0])
    et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);
  et_cpuv1_test_p1_links_finish_v1(p);
  et_cpuv1_test_candidate_links_finish_v1(p);
  if(!p->scheme_p1[0] || !p->scheme_candidate_ready || p->scheme_classified!=UINT64_MAX)
    et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);
  if(p->status==ET_R_OK) {
    p->scheme_ready=1u;et_cpuv1_test_resource_domain_v1(p,ET_R_SCHEME);
  }
  return (int64_t)p->status;
}
void et_cpuv1_test_resource_mutation_v1(unsigned m) { mutation=m; }
#endif
