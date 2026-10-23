/* Real native producer unit; source staged only. Root dispatches compilation. */
#include "cpuv1_resource_private.h"
#include "eshkol_transformer/f32_tensor.h"
#include "f32_parameter_internal.h"
#define ET_P1_PRIVATE_API
#include "p1_identity_internal.h"
#include "arena_memory.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
extern "C" int et_cpuv1_test_k1_partial_cleanup_v1(void);
static unsigned checks;
static void require(bool value,const char *message) {
  ++checks;
  if(!value) {std::fprintf(stderr,"CPUv1 producer unit FAIL %u: %s\n",checks,message);std::exit(1);}
}
static et_cpuv1_resource_packet_v1 before,after;
static int observe(et_cpuv1_resource_packet_v1 *p) {
  p->struct_size=sizeof(*p);p->version=ET_R_PACKET_VERSION;
  int r=et_cpuv1_test_resource_begin_v1(p);
  if(!r)r=et_cpuv1_test_native_resource_snapshot_v1(p);
  if(!r)et_cpuv1_test_runtime_resource_snapshot_v1(p);
  return et_cpuv1_test_resource_finish_v1(p);
}
/* Prepared genuine P1 callback and retained-dead identity controls. These
 * call original constructors/revocation; no fixture asserts Scheme COMPLETE. */
static void p1_callback_and_dead_checks() {
  void *context=et_p1_private_context_create_v1();
  require(context!=nullptr,"genuine P1 context");
  const char name[]="cpuv1-unit-provider";
  require(et_p1_private_provider_create_v1(context,name,sizeof(name)-1)==0,"genuine provider token");
  void *provider=et_p1_private_result_ptr_v1(context),*callbacks[8];
  for(unsigned i=0;i<8;i++) {
    require(et_p1_private_callback_identity_create_v1(context)==0,"genuine callback token");
    callbacks[i]=et_p1_private_result_ptr_v1(context);
  }
  require(et_p1_private_provider_seal_release_v1(context,provider,callbacks[0],callbacks[1],
    callbacks[2],callbacks[3],callbacks[4],callbacks[5],callbacks[6],callbacks[7])==0,
    "seven immutable callback slots plus eighth release-owned binding");
  auto start=[&](){
    after.struct_size=sizeof(after);after.version=ET_R_PACKET_VERSION;
    require(et_cpuv1_test_resource_begin_v1(&after)==0,"P1 fixture snapshot begin");
    require(et_cpuv1_test_native_resource_snapshot_v1(&after)==0,"P1 actual native census");
    require(et_cpuv1_test_p1_context_v1(&after,context)==0,"selected context split census");
  };
  start();
  require(et_cpuv1_test_p1_shell_v1(&after,context,provider,1,0)==0,"provider is one Scheme shell");
  for(int64_t i=0;i<8;i++)
    require(et_cpuv1_test_p1_callback_v1(&after,context,provider,callbacks[i],i)==0,"exact live callback/native slot");
  require(after.scheme_p1[14]==1 && after.scheme_p1[15]==8 && after.scheme_p1[13]==8 &&
          after.rows[ET_R_P1][1].bases==9,"one Scheme shell plus eight native-only callback bases");
  /* Isolate link reconciliation with a one-entry Scheme-index scalar fixture.
   * No typed roots/readiness are synthesized; overall finish stays COVERAGE. */
  after.scheme_p1[1]=1;
  require(et_cpuv1_test_p1_links_finish_v1(&after)==0,"full callback coverage reconciles");
  require(et_cpuv1_test_resource_finish_v1(&after)==ET_R_COVERAGE,"P1 fixture cannot fabricate Scheme COMPLETE");
  start();after.scheme_p1[1]=1;
  require(et_cpuv1_test_p1_shell_v1(&after,context,provider,1,0)==0,"omission fixture provider");
  for(int64_t i=0;i<7;i++)
    require(et_cpuv1_test_p1_callback_v1(&after,context,provider,callbacks[i],i)==0,"first seven callbacks");
  require(et_cpuv1_test_p1_links_finish_v1(&after)==ET_R_MISMATCH,"omitted eighth callback rejected");
  require(et_cpuv1_test_resource_finish_v1(&after)==ET_R_MISMATCH,"callback omission cannot COMPLETE");
  start();
  require(et_cpuv1_test_p1_callback_v1(&after,context,provider,callbacks[1],0)==ET_R_MISMATCH,
          "duplicate/wrong-slot callback rejected");
  require(et_cpuv1_test_resource_finish_v1(&after)==ET_R_MISMATCH,"wrong slot cannot COMPLETE");
  start();
  require(et_cpuv1_test_p1_callback_v1(&after,nullptr,provider,callbacks[0],0)==ET_R_MISMATCH,
          "foreign selected context rejected");
  require(et_cpuv1_test_resource_finish_v1(&after)==ET_R_MISMATCH,"wrong context cannot COMPLETE");

  require(et_p1_private_construction_begin_v1(context)==0,"genuine construction begin");
  void *construction=et_p1_private_result_ptr_v1(context);
  require(et_p1_private_construction_module_create_v1(context,construction)==0,"genuine scoped module identity");
  void *module=et_p1_private_result_ptr_v1(context);
  require(et_p1_private_construction_abort_v1(context,construction)==0,"original revocation leaves DEAD identity");
  start();
  require(et_cpuv1_test_p1_shell_v1(&after,context,module,2,1)==0,"retained DEAD identity has valid nonce/PID/context");
  require(after.scheme_states[3][2]==1,"actual native DEAD state reported");
  require(et_cpuv1_test_resource_finish_v1(&after)==ET_R_COVERAGE,"dead fixture lacks complete Scheme roots");
  for(unsigned mutation=1;mutation<=2;mutation++) {
    start();
    require(et_cpuv1_test_p1_shell_mutation_v1(&after,context,module,2,1,mutation)==ET_R_MISMATCH,
            "genuine retained DEAD nonce/PID mutation rejected and restored");
    require(et_cpuv1_test_resource_finish_v1(&after)==ET_R_MISMATCH,"identity mutation cannot COMPLETE");
  }
  start();
  require(et_cpuv1_test_p1_shell_v1(&after,context,module,2,1)==0,"restored DEAD identity remains authentic");
  require(et_cpuv1_test_resource_finish_v1(&after)==ET_R_COVERAGE,"restored identity census");
}
int main() {
  require(et_cpuv1_test_runtime_resource_warm_v1()==ET_R_OK,"normal glibc/root warm");
  require(observe(&before)==ET_R_COVERAGE,"native-only cannot claim Scheme COMPLETE");
  require(before.coverage==((UINT64_C(1)<<ET_R_DOMAINS)-1-(UINT64_C(1)<<ET_R_SCHEME)),"all twelve native/runtime domains");
  const uint64_t c4_na=(UINT64_C(1)<<8)|(UINT64_C(1)<<12);
  require(before.details[ET_R_C4][ET_R_C4_NOT_APPLICABLE_KINDS]==c4_na,
          "exact frozen C4 profile classifies logits/manual frame N/A");
  require(before.details[ET_R_C4][ET_R_C4_COMPILED_KINDS]==(((UINT64_C(1)<<13)-1u)^c4_na),
          "all remaining C4 kinds retain actual compiled layout facts");
  require(!before.rows[ET_R_C4][8].control_size && !before.rows[ET_R_C4][12].control_size,
          "N/A kinds have no invented control size");
  et_cpuv1_test_resource_mutation_v1(4);
  require(observe(&after)==ET_R_MISMATCH,"balanced absent-kind event history rejects profile");
  et_cpuv1_test_resource_mutation_v1(0);
  require(observe(&after)==ET_R_COVERAGE,"profile-negative never changed real event counters");
  require(et_cpuv1_test_k1_partial_cleanup_v1()==ET_R_OK,"real partial cleanup helper");
  require(observe(&after)==ET_R_COVERAGE,"partial helper conservation");
  const auto &a=before.rows[ET_R_K1][3],&b=after.rows[ET_R_K1][3];
  require(b.created==a.created+1 && b.freed==a.freed+1,"array event conservation");
  require(b.allocated_requested==a.allocated_requested+8*sizeof(const char *) &&
          b.freed_requested==a.freed_requested+8*sizeof(const char *),"full original eight-pointer extent");
  const auto &c=before.rows[ET_R_K1][2],&d=after.rows[ET_R_K1][2];
  require(d.created==c.created+3 && d.freed==c.freed+3 &&
          d.freed_requested==c.freed_requested+3*sizeof("cpuv1-partial"),"only three initialized strings");
  require(after.rows[ET_R_K1][3].bases==0 && after.rows[ET_R_K1][2].bases==0,"no partial helper owner remains");

  uint64_t shape[]={2,3};et_f32_tensor *tensor=nullptr;et_f32_tensor_error error;
  require(et_f32_tensor_create_v1(2,shape,&tensor,&error)==0,"real F32 create");
  require(observe(&before)==ET_R_COVERAGE,"F32 owned census");
  require(before.rows[ET_R_F32][6].requested==6*sizeof(float) &&
          before.rows[ET_R_F32][7].requested==2*sizeof(uint64_t) &&
          before.rows[ET_R_F32][8].requested==2*sizeof(size_t),"payload plus separate shape and strides");
  et_f32_test_tensor_metadata_v1 metadata={};metadata.struct_size=sizeof(metadata);
  require(et_f32_tensor_test_metadata_snapshot_v1(tensor,&metadata)==0,"save exact metadata");
  require(et_f32_tensor_test_metadata_corrupt_v1(tensor,ET_F32_TEST_TENSOR_METADATA_STRIDE,0)==0,"real stride mutation");
  require(observe(&after)==ET_R_MISMATCH,"census rejects malformed strides");
  require(et_f32_tensor_test_metadata_restore_v1(tensor,&metadata)==0,"restore exact metadata");
  require(observe(&after)==ET_R_COVERAGE,"restored metadata accepted");
  require(et_f32_tensor_destroy_v1(&tensor,&error)==0 && tensor==nullptr,"real F32 release");
  require(observe(&after)==ET_R_COVERAGE,"retired control and provenance conserved");
  require(after.rows[ET_R_F32][0].dead==1 && !after.rows[ET_R_F32][6].bases &&
          !after.rows[ET_R_F32][7].bases && !after.rows[ET_R_F32][8].bases,"retained control versus freed payload");

  /* Genuine fe9 block transfer: the source arena control/header/mutex must be
   * freed before the observer traverses root current AND adopted chains. */
  arena_t *source=arena_create(4096);require(source!=nullptr,"real source arena");
  require(arena_allocate(source,64)!=nullptr,"real source used bytes");
  require(arena_adopt_blocks(eshkol_root_arena_v1(),source)>0,"actual block adoption");
  arena_destroy(source);
  require(observe(&after)==ET_R_COVERAGE,"all arena capacity reconciles");
  require(after.runtime[ET_R_ADOPTED_BLOCKS]>=1 && after.runtime[ET_R_ADOPTED_USED]>=64,
          "adopted chain counted, not just current chain");
  require(after.runtime[ET_R_TRACKED_CURRENT]==after.runtime[ET_R_SCHEME_CAPACITY],"tracker reconciliation");
  require(after.runtime[ET_R_KNOWN_USABLE]<=after.runtime[ET_R_ALLOCATOR_INUSE],"nonnegative unclamped residual");

  /* Generic scalar/spine writers are not a concrete Scheme census. */
  after.struct_size=sizeof(after);after.version=ET_R_PACKET_VERSION;
  require(et_cpuv1_test_resource_begin_v1(&after)==0,"generic-only negative begin");
  for(int64_t i=0;i<64;i++)require(et_cpuv1_test_scheme_value_v1(&after,i,0,1,0)==0,"generic root fact");
  for(int64_t i=0;i<51;i++)require(et_cpuv1_test_scheme_value_v1(&after,i,1,4,1)==0,"generic lexical fact");
  require(after.scheme_classified==0 && !after.scheme_candidate_ready,"generic facts never classify");
  require(et_cpuv1_test_scheme_finish_v1(&after)==ET_R_COVERAGE,"generic facts never COMPLETE");
  require(et_cpuv1_test_resource_finish_v1(&after)==ET_R_COVERAGE,"generic observation rejected");

  void *base=ET_R_ALLOC(ET_R_RUNTIME,6,std::malloc(128),128,1u);
  require(base!=nullptr,"allocated observer-buffer fixture");
  auto open=[](){after.struct_size=sizeof(after);after.version=ET_R_PACKET_VERSION;
                require(et_cpuv1_test_resource_begin_v1(&after)==0,"negative fixture begin");};
  open();
  require(et_cpuv1_test_resource_add_v1(&after,ET_R_RUNTIME,6,base,128,1,ET_R_LIVE,3)==ET_R_ARGUMENT,
          "non-power-of-two alignment rejected before usable query");
  require(et_cpuv1_test_resource_finish_v1(&after)!=ET_R_OK,"bad alignment never COMPLETE");
  open();
  require(et_cpuv1_test_resource_add_v1(&after,ET_R_RUNTIME,6,base,SIZE_MAX,2,ET_R_LIVE,1)==ET_R_ARGUMENT,
          "requested extent overflow rejected");
  require(et_cpuv1_test_resource_finish_v1(&after)!=ET_R_OK,"overflow never COMPLETE");
  open();
  require(et_cpuv1_test_resource_add_v1(&after,ET_R_RUNTIME,6,base,128,1,ET_R_LIVE,1)==ET_R_OK,"first base");
  require(et_cpuv1_test_resource_add_v1(&after,ET_R_F32,6,base,32,sizeof(float),ET_R_LIVE,1)==ET_R_MISMATCH,
          "cross-domain duplicate allocation base rejected");
  require(et_cpuv1_test_resource_finish_v1(&after)!=ET_R_OK,"alias never COMPLETE");
  et_cpuv1_test_resource_mutation_v1(3);open();
  require(et_cpuv1_test_resource_add_v1(&after,ET_R_RUNTIME,6,base,128,1,ET_R_LIVE,1)==ET_R_SCRATCH,
          "fixed scratch saturation rejects without allocation");
  require(et_cpuv1_test_resource_finish_v1(&after)!=ET_R_OK,"saturation never COMPLETE");
  et_cpuv1_test_resource_mutation_v1(0);
  ET_R_FREE(ET_R_RUNTIME,6,base,128,1u);
  errno=EDOM;const char *absent=nullptr;
  ET_R_FREE(ET_R_K1,2,const_cast<char *>(absent),std::strlen(absent)+1u,1u);
  require(errno==EDOM,"NULL metadata never evaluated, original errno retained");
  require(observe(&after)==ET_R_COVERAGE,"all actual event/census rows conserved after negatives");
  require(after.rows[ET_R_K1][0].control_size>0 && after.rows[ET_R_C2_IO][0].control_alignment>0 &&
          after.details[ET_R_C2_IO][1]>0,"empty kinds still report compiled layout/prefix");
  require(et_cpuv1_test_runtime_static_bytes_v1()==sizeof(void *)+sizeof(long)+sizeof(bool),
          "runtime root/page/warmed declared static extents included");
  require(after.observer_static_bytes>=et_cpuv1_test_runtime_static_bytes_v1(),"full reported static extents");
  p1_callback_and_dead_checks();
  require(observe(&after)==ET_R_COVERAGE,"final actual native/runtime facts after P1 controls");
  std::printf("CPUV1-NATIVE-PRODUCER checks=%u Scheme=UNAVAILABLE native-requested-peak=UNOBSERVED\n",checks);
  std::printf("LAYOUT packet=%zu pointer=%zu F32=%llu K1=%llu C2prefix=%llu arena=%llu block=%llu scope=%llu mutex=%llu reflection=%llu entry=%llu\n",
    sizeof(after),sizeof(void *),(unsigned long long)after.rows[ET_R_F32][0].control_size,
    (unsigned long long)after.rows[ET_R_K1][0].control_size,
    (unsigned long long)after.details[ET_R_C2_IO][1],
    (unsigned long long)after.runtime[ET_R_SIZEOF_ARENA],(unsigned long long)after.runtime[ET_R_SIZEOF_BLOCK],
    (unsigned long long)after.runtime[ET_R_SIZEOF_SCOPE],(unsigned long long)after.runtime[ET_R_SIZEOF_MUTEX],
    (unsigned long long)after.runtime[ET_R_SIZEOF_REFLECTION],(unsigned long long)after.runtime[ET_R_SIZEOF_REFLECTION_ENTRY]);
  return 0;
}
