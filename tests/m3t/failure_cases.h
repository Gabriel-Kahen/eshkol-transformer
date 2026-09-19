extern void et_m3t_test_k1_fail_after(size_t);
extern void et_m3t_test_k1_allocator_reset(void);
static void discovery_failures(initializer *s) {
 size_t n;int64_t original[4];memcpy(original,s->words,sizeof(original));
 for(n=0;n<4096;n++) {
  et_m3t_test_k1_fail_after(n);owner *o=et_m3t_private_owner_create_v1(s);
  et_m3t_test_k1_allocator_reset();
  CHECK(!memcmp(original,s->words,sizeof(original)));
  if(o){OK(et_m3t_private_owner_abort_v1(o));break;}
  CHECK(error_domain==1 && error_category==ET_KERNEL_ERROR_INTERNAL && error_code==ET_KERNEL_CODE_ALLOCATION_FAILED);
  CHECK(!runtimes[0]&&!runtimes[1]&&!runtimes[2]&&!staged_owner);
 }
 CHECK(n<4096);printf("three-provider K1 discovery allocation failpoints: %zu\n",n);
}
extern et_f32_tensor *et_m3t_test_parameter_gradient(et_f32_parameter *);
static et_f32_test_live_counts_v1 live_counts(void) {
 et_f32_test_live_counts_v1 c={.struct_size=sizeof(c)};et_f32_test_live_counts_snapshot_v1(&c);return c;
}
static void same_live(et_f32_test_live_counts_v1 before) {
 et_f32_test_live_counts_v1 after=live_counts();CHECK(!memcmp(&before,&after,sizeof(before)));
}
static void construction_failures(initializer *s) {
 et_f32_test_live_counts_v1 baseline=live_counts();size_t limit;
 for(limit=0;limit<256;limit++) {
  et_f32_tensor_test_fail_alloc_after_v1(limit);owner *o=et_m3t_private_owner_create_v1(s);
  et_f32_tensor_test_reset_allocator_v1();
  if(o){OK(et_m3t_private_owner_abort_v1(o));same_live(baseline);break;}
  CHECK(error_domain==3 && error_code==ET_F32_TENSOR_CODE_ALLOCATION_FAILED);same_live(baseline);CHECK(s->words[2]==0);
 }
 CHECK(limit<256);printf("owner I2 allocation failpoints: %zu\n",limit);
 for(size_t n=0;n<2;n++){et_m3t_test_fail_alloc_after(n);CHECK(!et_m3t_private_owner_create_v1(s));same_live(baseline);}
 et_m3t_test_fail_alloc_after(SIZE_MAX);
 owner *o=et_m3t_private_owner_create_v1(s);CHECK(o);static unsigned char h[14];et_f32_tensor_error e;
 CHECK(et_m3t_construction_parameter_preflight_internal(o->p[0],&h[0],&e));
 OK(et_m3t_construction_parameter_preflight_internal(o->p[0],NULL,&e));
 /* Registration can bind before owner_bind's bookkeeping. Exact readback is
  * sufficient; NULL and a different canonical handle are never accepted. */
 OK(et_f32_parameter_bind_identity_v1(o->p[0],&h[0],&e));
 CHECK(et_m3t_construction_parameter_preflight_internal(o->p[0],NULL,&e));
 CHECK(et_m3t_construction_parameter_preflight_internal(o->p[0],&h[1],&e));
 OK(et_m3t_construction_parameter_preflight_internal(o->p[0],&h[0],&e));
 for(int n=0;n<14;n++)OK(et_m3t_private_owner_bind_v1(o,n,&h[n]));
 baseline=live_counts();
 for(size_t n=0;n<14;n++) {
  for(int kind=0;kind<7;kind++) {
   et_f32_tensor *v=value(o,n),*g=et_m3t_test_parameter_gradient(o->p[n]);
   et_f32_tensor_borrow *lease=NULL;et_f32_scoped_guard_internal guard={0};
   et_f32_gradient_reset_plan *reset=NULL;et_f32_tensor_copy_plan *copy=NULL;et_f32_gradient_plan *gradient=NULL;et_f32_tensor *numerator=NULL;
   et_f32_gradient_contribution_v1 contribution={sizeof(contribution),o->p[n],NULL,0};
   et_f32_tensor_copy_assignment_v1 assignment={sizeof(assignment),v,g};
   if(kind==0)OK(et_f32_tensor_borrow_begin_v1(v,&lease,&e));
   if(kind==1)OK(et_f32_tensor_borrow_begin_v1(g,&lease,&e));
   if(kind==2)OK(et_f32_tensor_scoped_begin_internal(v,&guard,&e));
   if(kind==3)OK(et_f32_tensor_scoped_begin_internal(g,&guard,&e));
   if(kind==4)OK(et_f32_tensor_copy_plan_prepare_v1(1,&assignment,&copy,&e));
   if(kind==5)OK(et_f32_gradient_reset_plan_prepare_v1(1,&o->p[n],&reset,&e));
   if(kind==6){OK(et_f32_tensor_clone_v1(v,&numerator,&e));contribution.weighted_numerator=numerator;
    OK(et_f32_gradient_plan_prepare_v1(1,&contribution,UINT32_C(0x3f800000),&gradient,&e));}
   et_f32_test_live_counts_v1 held=live_counts();
   et_f32_tensor_test_fail_alloc_after_v1(0);et_m3t_test_fail_alloc_after(0);
   CHECK(et_m3t_construction_parameter_preflight_internal(o->p[n],&h[n],&e));
   CHECK(et_m3t_construction_parameter_preflight_internal(o->p[0],&h[0],&e));
   CHECK(et_m3t_private_owner_abort_v1(o));same_live(held);
   for(size_t j=0;j<14;j++)CHECK(et_f32_parameter_is_live_v1(o->p[j]));
   OK(et_f32_tensor_borrow_end_v1(&lease,&e));OK(et_f32_tensor_scoped_end_internal(&guard));
   OK(et_f32_tensor_copy_plan_release_v1(&copy,&e));OK(et_f32_gradient_reset_plan_release_v1(&reset,&e));OK(et_f32_gradient_plan_release_v1(&gradient,&e));OK(et_f32_tensor_destroy_v1(&numerator,&e));
   OK(et_m3t_construction_parameter_preflight_internal(o->p[n],&h[n],&e));
   et_f32_tensor_test_reset_allocator_v1();et_m3t_test_fail_alloc_after(SIZE_MAX);same_live(baseline);
  }
 }
 et_f32_tensor_test_fail_alloc_after_v1(0);et_m3t_test_fail_alloc_after(0);
 OK(et_m3t_private_owner_abort_v1(o));OK(et_m3t_private_owner_abort_v1(o));
 et_f32_tensor_test_reset_allocator_v1();et_m3t_test_fail_alloc_after(SIZE_MAX);
 puts("all 14 construction positions: value/gradient ordinary/scoped borrow and plan rejection preserve all carriers");
}
static void workspace_failures(owner *o) {
 et_f32_test_live_counts_v1 baseline=live_counts();size_t limit;
 for(limit=0;limit<512;limit++) {
  et_f32_tensor_test_fail_alloc_after_v1(limit);workspace *w=et_m3t_private_workspace_create_v1(o);
  et_f32_tensor_test_reset_allocator_v1();
  if(w){OK(et_m3t_private_workspace_release_v1(w));same_live(baseline);break;}
  CHECK(error_domain==3 && error_code==ET_F32_TENSOR_CODE_ALLOCATION_FAILED);same_live(baseline);
 }
 CHECK(limit<512);printf("workspace I2 allocation failpoints: %zu\n",limit);
 for(limit=0;limit<16;limit++) {
  et_i64_tensor_test_fail_alloc_after_v1(limit);workspace *w=et_m3t_private_workspace_create_v1(o);
  et_i64_tensor_test_reset_allocator_v1();
  if(w){OK(et_m3t_private_workspace_release_v1(w));same_live(baseline);break;}
  CHECK(error_domain==2 && error_code==ET_I64_TENSOR_CODE_ALLOCATION_FAILED);same_live(baseline);
 }
 CHECK(limit<16);printf("workspace I1 allocation failpoints: %zu\n",limit);
}
typedef struct frame_copy {uint32_t bits[SLOT_COUNT][1024];unsigned char ready[SLOT_COUNT];} frame_copy;
static void capture(workspace *w,frame_copy *f) {
 memset(f,0,sizeof(*f));memcpy(f->ready,w->ready,sizeof(w->ready));
 for(size_t i=0;i<SLOT_COUNT;i++) {
  size_t n;et_f32_tensor_error e;OK(et_f32_tensor_element_count_v1(w->slots[i],&n,&e));
  OK(et_f32_tensor_copy_bits_to_v1(w->slots[i],f->bits[i],n,&e));
 }
}
static void unchanged(workspace *w,frame_copy *before) {
 frame_copy *after=malloc(sizeof(*after));CHECK(after);capture(w,after);CHECK(!memcmp(before,after,sizeof(*after)));free(after);
}
static void lifecycle_negatives(owner *o,input *i,logits *u,logits *l) {
 workspace *w=et_m3t_private_workspace_create_v1(o),*other=et_m3t_private_workspace_create_v1(o);CHECK(w&&other);
 frame_copy *before=malloc(sizeof(*before));CHECK(before);
 CHECK(et_m3t_private_linear_v1(i,0,0)==ET_M3T_INVALID_ARGUMENT);
 CHECK(et_m3t_private_workspace_begin_v1(w,u,0)==ET_M3T_INVALID_ARGUMENT);
 CHECK(et_m3t_private_workspace_begin_v1(w,i,2)==ET_M3T_INVALID_ARGUMENT);
 CHECK(et_m3t_private_input_copy_v1(i,-1,2)==ET_M3T_SHAPE_MISMATCH);
 CHECK(et_m3t_private_input_copy_v1(i,1,256)==ET_M3T_SHAPE_MISMATCH);
 int64_t ids[2];et_i64_tensor_error ie;OK(et_i64_tensor_copy_to_v1(i->tensor,ids,2,&ie));CHECK(ids[0]==65&&ids[1]==66);
 et_f32_scoped_guard_internal g={0};et_f32_tensor_error fe;
 OK(et_f32_tensor_scoped_begin_internal(value(o,13),&g,&fe));
 CHECK(et_m3t_private_workspace_begin_v1(w,i,0));CHECK(w->r.state==0 && o->active==NULL);OK(et_f32_tensor_scoped_end_internal(&g));
 OK(et_m3t_private_workspace_begin_v1(w,i,0));
 CHECK(et_m3t_private_workspace_begin_v1(other,i,0)==ET_M3T_INVALID_STATE);
 w->busy=1;CHECK(et_m3t_private_workspace_reset_v1(w));CHECK(et_m3t_private_workspace_release_v1(w));CHECK(et_m3t_private_embedding_v1(w,0,0));w->busy=0;
 capture(w,before);
 et_i64_tensor_test_fail_alloc_after_v1(0);CHECK(et_m3t_private_embedding_v1(w,0,0));et_i64_tensor_test_reset_allocator_v1();unchanged(w,before);
 for(size_t n=0;n<sizeof(roles)/sizeof(*roles);n++) {
  const role *r=&roles[n];if(r->family==EMBEDDING&&!r->reverse)continue;
  CHECK(primitive(w,r->family,r->index,r->reverse));unchanged(w,before);
 }
 CHECK(et_m3t_private_workspace_copy_logits_v1(w,l));CHECK(et_m3t_private_workspace_vjp_begin_v1(w,u));
 CHECK(et_m3t_private_linear_v1(w,-1,0)==ET_M3T_INVALID_ARGUMENT);
 CHECK(et_m3t_private_linear_v1(w,0,2)==ET_M3T_INVALID_ARGUMENT);unchanged(w,before);
 forward(w);capture(w,before);
 for(size_t n=0;n<sizeof(roles)/sizeof(*roles);n++)if(!roles[n].reverse){CHECK(primitive(w,roles[n].family,roles[n].index,0));unchanged(w,before);}
 CHECK(et_m3t_private_workspace_check_primals_v1(w,1)==ET_M3T_INVALID_STATE);
 uint32_t original[8],modified[8];et_f32_tensor *v=value(o,13);
 OK(et_f32_tensor_copy_bits_to_v1(v,original,8,&fe));memcpy(modified,original,sizeof(original));modified[0]^=UINT32_C(0x80000000);
 OK(et_f32_tensor_copy_bits_from_v1(v,modified,8,&fe));CHECK(et_m3t_private_workspace_check_primals_v1(w,0)==ET_M3T_INVALID_STATE);
 OK(et_f32_tensor_copy_bits_from_v1(v,original,8,&fe));OK(et_m3t_private_workspace_check_primals_v1(w,0));
 OK(et_m3t_private_workspace_vjp_begin_v1(w,u));capture(w,before);
 CHECK(et_m3t_private_workspace_vjp_begin_v1(w,u));CHECK(et_m3t_private_linear_v1(w,0,0));unchanged(w,before);
 uint32_t dz[512],nan[512];OK(et_f32_tensor_copy_bits_to_v1(w->slots[DZ],dz,512,&fe));memcpy(nan,dz,sizeof(nan));nan[0]=UINT32_C(0x7fc00000);
 OK(et_f32_tensor_copy_bits_from_v1(w->slots[DZ],nan,512,&fe));capture(w,before);
 CHECK(et_m3t_private_linear_v1(w,6,1));CHECK(error_domain==1);unchanged(w,before);
 OK(et_f32_tensor_copy_bits_from_v1(w->slots[DZ],dz,512,&fe));
 reverse(w);capture(w,before);
 for(size_t n=0;n<sizeof(roles)/sizeof(*roles);n++)if(roles[n].reverse){CHECK(primitive(w,roles[n].family,roles[n].index,1));unchanged(w,before);}
 OK(et_f32_tensor_scoped_begin_internal(w->slots[GTOKEN],&g,&fe));
 CHECK(et_m3t_private_workspace_reset_v1(w));CHECK(w->r.state==2&&o->active==w);
 CHECK(et_m3t_private_workspace_release_v1(w));CHECK(w->r.state==2&&o->active==w);OK(et_f32_tensor_scoped_end_internal(&g));
 OK(et_m3t_private_workspace_reset_v1(w));OK(et_m3t_private_workspace_reset_v1(w));
 CHECK(!et_m3t_private_workspace_gradient_v1(w,0));OK(et_m3t_private_workspace_begin_v1(other,i,1));
 OK(et_m3t_private_workspace_release_v1(other));OK(et_m3t_private_workspace_release_v1(w));
 CHECK(et_m3t_private_workspace_begin_v1(w,i,0)==ET_M3T_INVALID_STATE);free(before);
 puts("lifecycle, every role readiness/duplicate, exact stale restoration, failed primitive byte preservation passed");
}
static size_t registry_bytes(void) {
 size_t bytes=0;for(record *r=registry;r;r=r->next) {
  if(r->kind==INITIALIZER)bytes+=sizeof(initializer);else if(r->kind==OWNER)bytes+=sizeof(owner);
  else if(r->kind==INPUT)bytes+=sizeof(input);else if(r->kind==LOGITS)bytes+=sizeof(logits);else bytes+=sizeof(workspace);
 }return bytes;
}
static void memory_loop(owner *o,input *i,logits *u,logits *l) {
 workspace *w=et_m3t_private_workspace_create_v1(o);CHECK(w);
 const void *addresses[SLOT_COUNT+14];
 for(size_t n=0;n<SLOT_COUNT;n++)addresses[n]=et_f32_tensor_test_data_storage_v1(w->slots[n]);
 for(size_t n=0;n<14;n++)addresses[SLOT_COUNT+n]=et_f32_tensor_test_data_storage_v1(w->saved[n]);
 et_f32_test_live_counts_v1 live=live_counts();
 et_f32_test_retired_counts_v1 retired={.struct_size=sizeof(retired)},after=retired;
 et_f32_test_retired_counts_snapshot_v1(&retired);size_t shells=registry_bytes();
 et_f32_tensor_test_fail_alloc_after_v1(0);et_m3t_test_fail_alloc_after(0);
 const size_t batches[]={10,100,1000};
 for(size_t batch=0;batch<3;batch++) {
  for(size_t n=0;n<batches[batch];n++) {
   OK(et_m3t_private_workspace_begin_v1(w,i,1));forward(w);OK(et_m3t_private_workspace_copy_logits_v1(w,l));
   OK(et_m3t_private_workspace_vjp_begin_v1(w,u));reverse(w);OK(et_m3t_private_workspace_reset_v1(w));
  }
  same_live(live);et_f32_test_retired_counts_snapshot_v1(&after);CHECK(!memcmp(&retired,&after,sizeof(after)));
  CHECK(shells==registry_bytes());for(size_t n=0;n<SLOT_COUNT;n++)CHECK(addresses[n]==et_f32_tensor_test_data_storage_v1(w->slots[n]));
  for(size_t n=0;n<14;n++)CHECK(addresses[SLOT_COUNT+n]==et_f32_tensor_test_data_storage_v1(w->saved[n]));
  printf("native frame reuse batch %zu: %zu live I2 tensors, %zu M3T control bytes, %zu retained I2 control bytes; zero growth\n",batches[batch],live.tensors,shells,after.retained_control_bytes);
 }
 et_f32_tensor_test_reset_allocator_v1();et_m3t_test_fail_alloc_after(SIZE_MAX);OK(et_m3t_private_workspace_release_v1(w));
}
static void ingress_and_initializer(initializer *s,owner *o,input *i) {
 CHECK(!et_m3t_private_initializer_create_v1(-1));CHECK(error_category==ET_M3T_INVALID_ARGUMENT);
 initializer *staged=et_m3t_private_initializer_create_v1(17);CHECK(staged);
 CHECK(!et_m3t_private_owner_create_v1(staged));OK(et_m3t_private_initializer_abort_v1(staged));OK(et_m3t_private_initializer_abort_v1(staged));
 CHECK(et_m3t_private_initializer_seal_v1(staged));CHECK(et_m3t_private_initializer_abort_v1(s));
 initializer *near=et_m3t_private_initializer_create_v1(19);CHECK(near);OK(et_m3t_private_initializer_seal_v1(near));
 uint64_t low=UINT64_MAX-289,high=UINT64_MAX;memcpy(&near->words[2],&low,8);memcpy(&near->words[3],&high,8);
 int64_t old[4];memcpy(old,near->words,sizeof(old));CHECK(!et_m3t_private_owner_create_v1(near));CHECK(!memcmp(old,near->words,sizeof(old)));
 owner *next=et_m3t_private_owner_create_v1(o->successor);CHECK(next);static unsigned char continuation_handles[14];
 for(int n=0;n<14;n++)OK(et_m3t_private_owner_bind_v1(next,n,&continuation_handles[n]));
 for(int n=0;n<14;n++)OK(et_m3t_private_owner_initialize_v1(next,n));
 CHECK(next->successor->words[2]==580);CHECK(o->successor->words[2]==290);OK(et_m3t_private_owner_abort_v1(next));
 /* The direct N3K initializer is independently threaded over canonical shapes. */
 int64_t state[4]={1,1729,0,0},successor[4];const uint64_t four[]={4};et_f32_tensor_error e;
 for(int n=0;n<14;n++) {
  uint32_t actual[1024],expected[1024]={0};size_t count;
  OK(et_f32_tensor_element_count_v1(value(o,(size_t)n),&count,&e));OK(et_f32_tensor_copy_bits_to_v1(value(o,(size_t)n),actual,count,&e));
  if(n==6||n==8||n==11){}else if(n==7||n==9||n==12){for(size_t j=0;j<count;j++)expected[j]=UINT32_C(0x3f800000);}
  else {
   et_kernel_tensor_view_v1 in[]={view(state,32,"i64",1,four)};
   et_kernel_tensor_view_v1 out[]={view(expected,count*4,"f32",2,parameter_shape[n]),view(successor,32,"i64",1,four)};
   OK(dispatch(0,"n3k.matrix-init","n3k.matrix-init.uniform",2,parameter_shape[n],in,1,out,2));memcpy(state,successor,32);
  }
  CHECK(!memcmp(actual,expected,count*4));
 }
 CHECK(state[2]==290);
 void *tokenizer=et_t1_i64_shell_create_v1(2);CHECK(tokenizer);OK(et_t1_i64_shell_write_v1(tokenizer,0,65));OK(et_t1_i64_shell_write_v1(tokenizer,1,66));
 CHECK(et_m3t_private_input_copy_tokenizer_v1(i,tokenizer));OK(et_t1_i64_shell_seal_v1(tokenizer));OK(et_m3t_private_input_copy_tokenizer_v1(i,tokenizer));
 void *bad_tokenizer=et_t1_i64_shell_create_v1(2);CHECK(bad_tokenizer);OK(et_t1_i64_shell_write_v1(bad_tokenizer,0,256));OK(et_t1_i64_shell_write_v1(bad_tokenizer,1,2));OK(et_t1_i64_shell_seal_v1(bad_tokenizer));
 CHECK(et_m3t_private_input_copy_tokenizer_v1(i,bad_tokenizer)==ET_M3T_SHAPE_MISMATCH);
 int64_t ids[2];et_i64_tensor_error ie;OK(et_i64_tensor_copy_to_v1(i->tensor,ids,2,&ie));CHECK(ids[0]==65&&ids[1]==66);
 for(size_t n=0;n<4;n++){et_i64_tensor_test_fail_alloc_after_v1(n);CHECK(!et_m3t_private_input_create_v1());et_i64_tensor_test_reset_allocator_v1();}
 for(size_t n=0;n<4;n++){et_f32_test_live_counts_v1 before=live_counts();et_f32_tensor_test_fail_alloc_after_v1(n);CHECK(!et_m3t_private_logits_create_v1());et_f32_tensor_test_reset_allocator_v1();same_live(before);}
 puts("typed initializer continuation/exhaustion, exact initializer oracle, ingress and sealed T1 copy passed");
}
