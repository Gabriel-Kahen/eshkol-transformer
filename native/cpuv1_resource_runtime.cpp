#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
/* Noninstalled bridge compiled only against the authenticated pinned fe9
 * private header. No changes to the Eshkol runtime or public runtime ABI. */
#include "cpuv1_resource_private.h"
#ifdef ET_CPUV1_RESOURCE_TESTING
#include "arena_memory.h"
#include "eshkol/core/resource_limits.h"
#include <cerrno>
#include <climits>
#include <cstdint>
#include <fcntl.h>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <malloc.h>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>

static arena_t *resource_root;
static long resource_page_bytes;
static bool resource_warmed;
/* Sum declared static object extents. Linker padding/code/rodata are excluded. */
extern "C" size_t et_cpuv1_test_runtime_static_bytes_v1(void) {
  return sizeof(resource_root)+sizeof(resource_page_bytes)+sizeof(resource_warmed);
}
static bool libc_symbol(const void *symbol) {
  Dl_info info;
  if(!dladdr(symbol,&info) || !info.dli_fname)return false;
  const char *base=std::strrchr(info.dli_fname,'/');
  base=base?base+1:info.dli_fname;
  return std::strcmp(base,"libc.so.6")==0;
}

static bool plus(et_cpuv1_resource_packet_v1 *p,uint64_t &a,uint64_t b) {
  if(a>UINT64_MAX-b) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_OVERFLOW);return false;
  }
  a+=b;return true;
}
static bool owned(et_cpuv1_resource_packet_v1 *p,unsigned k,const void *base,
                  size_t n,size_t size,size_t align) {
  return !et_cpuv1_test_resource_add_v1(p,ET_R_RUNTIME,k,base,n,size,ET_R_LIVE,align);
}
extern "C" int et_cpuv1_test_runtime_resource_warm_v1(void) {
  /* Initialization belongs BEFORE P0, including libc lazy paths used below. */
  resource_root=eshkol_root_arena_v1();
  resource_page_bytes=sysconf(_SC_PAGESIZE);
  struct rusage usage;
  (void)mallinfo2();(void)getrusage(RUSAGE_SELF,&usage);
  (void)eshkol_get_heap_usage();(void)eshkol_get_peak_heap_usage();
  resource_warmed=libc_symbol(reinterpret_cast<const void *>(&std::malloc)) &&
      libc_symbol(reinterpret_cast<const void *>(&std::calloc)) &&
      libc_symbol(reinterpret_cast<const void *>(&std::realloc)) &&
      libc_symbol(reinterpret_cast<const void *>(&std::free)) &&
      libc_symbol(reinterpret_cast<const void *>(&malloc_usable_size)) &&
      libc_symbol(reinterpret_cast<const void *>(&mallinfo2)) && resource_root && resource_page_bytes>0 && !region_current() &&
      get_global_arena()==resource_root && get_global_arena_shared()==resource_root;
  return resource_warmed?ET_R_OK:ET_R_UNSUPPORTED;
}
extern "C" void et_cpuv1_test_runtime_resource_snapshot_v1(et_cpuv1_resource_packet_v1 *p) {
  if(!p || p->status!=ET_R_OK) return;
  if(!resource_warmed || region_current() ||
      get_global_arena()!=resource_root || get_global_arena_shared()!=resource_root) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_NOT_QUIESCENT);return;
  }
  /* External caller also guarantees serial execution/no other Scheme threads.
   * The pinned runtime has no supported complete foreign-thread enumerator. */
  uint64_t *v=p->runtime;
  v[ET_R_ROOT]=reinterpret_cast<uintptr_t>(resource_root);
  v[ET_R_CURRENT_ARENA]=reinterpret_cast<uintptr_t>(get_global_arena());
  v[ET_R_SHARED_ARENA]=reinterpret_cast<uintptr_t>(get_global_arena_shared());
  v[ET_R_REGION]=reinterpret_cast<uintptr_t>(region_current());
  v[ET_R_SIZEOF_ARENA]=sizeof(arena_t);v[ET_R_SIZEOF_BLOCK]=sizeof(arena_block_t);
  v[ET_R_SIZEOF_SCOPE]=sizeof(arena_scope_t);v[ET_R_SIZEOF_MUTEX]=sizeof(pthread_mutex_t);
  v[ET_R_SIZEOF_REFLECTION]=sizeof(eshkol_lambda_registry_t);
  v[ET_R_SIZEOF_REFLECTION_ENTRY]=sizeof(eshkol_lambda_entry_t);
  et_cpuv1_test_resource_layout_v1(p,ET_R_RUNTIME,0,sizeof(arena_t),alignof(arena_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_RUNTIME,1,sizeof(arena_block_t),alignof(arena_block_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_RUNTIME,2,1u,1u);
  et_cpuv1_test_resource_layout_v1(p,ET_R_RUNTIME,3,sizeof(arena_scope_t),alignof(arena_scope_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_RUNTIME,4,sizeof(pthread_mutex_t),alignof(pthread_mutex_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_RUNTIME,5,sizeof(eshkol_lambda_entry_t),alignof(eshkol_lambda_entry_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_RUNTIME,7,sizeof(eshkol_lambda_registry_t),alignof(eshkol_lambda_registry_t));
  if(!owned(p,0,resource_root,1,sizeof(*resource_root),alignof(arena_t)))return;
  v[ET_R_METADATA_REQUESTED]=sizeof(*resource_root);
  for(unsigned chain=0;chain<2;chain++) {
    for(arena_block_t *b=chain?resource_root->adopted_blocks:resource_root->current_block;b;b=b->next) {
      /* add rejects duplicates/cycles BEFORE any usable-size query; trusted
       * registered metadata only, not arbitrary native-pointer validation. */
      if(!owned(p,1,b,1,sizeof(*b),alignof(arena_block_t)))return;
      if(!b->memory || b->used>b->size ||
         !owned(p,2,b->memory,b->size,1u,1u)) {
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
      }
      if(!plus(p,v[chain?ET_R_ADOPTED_BLOCKS:ET_R_CURRENT_BLOCKS],1u) ||
         !plus(p,v[chain?ET_R_ADOPTED_USED:ET_R_CURRENT_USED],b->used) ||
         !plus(p,v[chain?ET_R_ADOPTED_CAPACITY:ET_R_CURRENT_CAPACITY],b->size) ||
         !plus(p,v[ET_R_METADATA_REQUESTED],sizeof(*b)))return;
    }
  }
  for(arena_scope_t *s=resource_root->current_scope;s;s=s->parent) {
    if(!owned(p,3,s,1,sizeof(*s),alignof(arena_scope_t)) ||
       !plus(p,v[ET_R_SCOPES],1u) ||
       !plus(p,v[ET_R_METADATA_REQUESTED],sizeof(*s)))return;
  }
  if(resource_root->mutex &&
     (!owned(p,4,resource_root->mutex,1,sizeof(pthread_mutex_t),alignof(pthread_mutex_t)) ||
      !plus(p,v[ET_R_METADATA_REQUESTED],sizeof(pthread_mutex_t))))return;
  v[ET_R_SCHEME_CAPACITY]=v[ET_R_CURRENT_CAPACITY];
  v[ET_R_SCHEME_USED]=v[ET_R_CURRENT_USED];
  if(!plus(p,v[ET_R_SCHEME_CAPACITY],v[ET_R_ADOPTED_CAPACITY]) ||
     !plus(p,v[ET_R_SCHEME_USED],v[ET_R_ADOPTED_USED]))return;
  v[ET_R_TRACKED_CURRENT]=eshkol_get_heap_usage();
  v[ET_R_TRACKED_PEAK]=eshkol_get_peak_heap_usage();
  if(v[ET_R_TRACKED_CURRENT]!=v[ET_R_SCHEME_CAPACITY] ||
     resource_root->total_allocated!=v[ET_R_SCHEME_CAPACITY] ||
     v[ET_R_TRACKED_PEAK]<v[ET_R_TRACKED_CURRENT]) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
  }
  if(g_lambda_registry) {
    if(!owned(p,7,g_lambda_registry,1,sizeof(*g_lambda_registry),alignof(eshkol_lambda_registry_t)))return;
    v[ET_R_LAMBDA_COUNT]=g_lambda_registry->count;
    v[ET_R_LAMBDA_CAPACITY]=g_lambda_registry->capacity;
    if(g_lambda_registry->count>g_lambda_registry->capacity ||
       (g_lambda_registry->capacity && !g_lambda_registry->entries) ||
       !owned(p,5,g_lambda_registry->entries,g_lambda_registry->capacity,
              sizeof(*g_lambda_registry->entries),alignof(eshkol_lambda_entry_t))) {
      et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
    }
  }
  struct rusage usage;
  if(getrusage(RUSAGE_SELF,&usage) || usage.ru_maxrss<0 ||
     static_cast<uint64_t>(usage.ru_maxrss)>UINT64_MAX/1024u) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_UNSUPPORTED);return;
  }
  v[ET_R_RSS_PEAK]=static_cast<uint64_t>(usage.ru_maxrss)*1024u;
  v[ET_R_PID]=static_cast<uint64_t>(getpid());v[ET_R_PAGE_BYTES]=resource_page_bytes;
  /* Linux statm uses no stdio/iostream heap-backed buffer. */
  char buffer[128];int fd=open("/proc/self/statm",O_RDONLY|O_CLOEXEC);
  if(fd<0) {et_cpuv1_test_resource_fail_v1(p,ET_R_UNSUPPORTED);return;}
  ssize_t bytes=read(fd,buffer,sizeof(buffer));int saved_errno=errno;
  int closed=close(fd);errno=saved_errno;
  if(bytes<=0 || bytes==static_cast<ssize_t>(sizeof(buffer)) || closed) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_UNSUPPORTED);return;
  }
  size_t i=0;while(i<static_cast<size_t>(bytes) && buffer[i]>='0' && buffer[i]<='9')i++;
  if(!i || i==static_cast<size_t>(bytes) || buffer[i++]!=' ') {
    et_cpuv1_test_resource_fail_v1(p,ET_R_UNSUPPORTED);return;
  }
  uint64_t pages=0;size_t start=i;
  while(i<static_cast<size_t>(bytes) && buffer[i]>='0' && buffer[i]<='9') {
    unsigned digit=static_cast<unsigned>(buffer[i++]-'0');
    if(pages>(UINT64_MAX-digit)/10u) {et_cpuv1_test_resource_fail_v1(p,ET_R_OVERFLOW);return;}
    pages=pages*10u+digit;
  }
  if(i==start || pages>UINT64_MAX/static_cast<uint64_t>(resource_page_bytes)) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_OVERFLOW);return;
  }
  v[ET_R_RSS_CURRENT]=pages*static_cast<uint64_t>(resource_page_bytes);
  struct mallinfo2 info=mallinfo2();
  v[ET_R_MALL_ARENA]=info.arena;v[ET_R_MALL_HBLKHD]=info.hblkhd;
  v[ET_R_MALL_UORDBLKS]=info.uordblks;v[ET_R_MALL_FORDBLKS]=info.fordblks;
  v[ET_R_MALL_HBLKS]=info.hblks;v[ET_R_MALL_ORDBLKS]=info.ordblks;
  v[ET_R_ALLOCATOR_BACKING]=info.arena;
  v[ET_R_ALLOCATOR_INUSE]=info.uordblks;
  if(!plus(p,v[ET_R_ALLOCATOR_BACKING],info.hblkhd) ||
     !plus(p,v[ET_R_ALLOCATOR_INUSE],info.hblkhd))return;
  /* Every base has already passed global duplicate rejection, including owned
   * allocator-returned mmap bases; hblkhd is never added wholesale here. */
  for(unsigned d=0;d<ET_R_DOMAINS;d++)for(unsigned k=0;k<ET_R_KINDS;k++)
    if(!plus(p,v[ET_R_KNOWN_USABLE],p->rows[d][k].usable))return;
  if(v[ET_R_KNOWN_USABLE]>v[ET_R_ALLOCATOR_INUSE]) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
  }
  v[ET_R_ALLOCATOR_RESIDUAL]=v[ET_R_ALLOCATOR_INUSE]-v[ET_R_KNOWN_USABLE];
  v[ET_R_NORMAL_PROFILE]=1u;
  et_cpuv1_test_resource_domain_v1(p,ET_R_RUNTIME);
}
extern "C" int64_t et_cpuv1_test_scheme_hash_v1(void *raw,int64_t index,
 int64_t lexical,void *candidate) {
  et_cpuv1_resource_packet_v1 *p=static_cast<et_cpuv1_resource_packet_v1 *>(raw);
  if(!resource_warmed || !candidate || !arena_contains(resource_root,candidate)) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  const eshkol_hash_table_t *h=static_cast<const eshkol_hash_table_t *>(candidate);
  if(h->size>h->capacity || h->tombstones>h->capacity-h->size ||
     (h->capacity && (!h->keys || !h->values || !h->status)) || h->size>INT64_MAX) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  size_t occupied=0,deleted=0;
  for(size_t i=0;i<h->capacity;i++) {
    if(h->status[i]==HASH_ENTRY_OCCUPIED)occupied++;
    else if(h->status[i]==HASH_ENTRY_DELETED)deleted++;
    else if(h->status[i]!=HASH_ENTRY_EMPTY) {
      et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return ET_R_MISMATCH;
    }
  }
  if(occupied!=h->size || deleted!=h->tombstones) {
    et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  p->scheme_p1[1]=h->size;p->scheme_p1[2]=h->capacity;p->scheme_p1[3]=h->tombstones;
  return et_cpuv1_test_scheme_value_v1(raw,index,lexical,3,static_cast<int64_t>(h->size));
}
#endif
