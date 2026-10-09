#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_M3T_TESTING 1
#include "../../src/eshkol_transformer/m3t_transport.c"
#include <stdio.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"candidate retire %s:%d: %s (%lld/%lld)\n", \
 __FILE__,__LINE__,#x,(long long)error_category,(long long)error_code); return 1; } } while(0)
#define OK(x) CHECK((x)==0)
static unsigned char handles[3][14];
static owner *sealed_owner(initializer *original,int group) {
 owner *o=et_m3t_private_owner_create_v1(original);if(!o)return NULL;
 for(int i=0;i<14;i++)if(et_m3t_private_owner_bind_v1(o,i,&handles[group][i]))return NULL;
 for(int i=0;i<14;i++)if(et_m3t_private_owner_initialize_v1(o,i))return NULL;
 if(et_m3t_private_owner_seal_v1(o) || et_m3t_private_initializer_seal_v1(o->successor))return NULL;
 return o;
}
static int reject(int64_t rc,int64_t category) {
 return rc==category && error_category==category && error_code!=0;
}
int main(void) {
 initializer *original=et_m3t_private_initializer_create_v1(1729);CHECK(original);
 OK(et_m3t_private_initializer_seal_v1(original));
 owner *o=sealed_owner(original,0);CHECK(o);
 owner *shared=sealed_owner(original,1);CHECK(shared);
 CHECK(reject(et_m3t_private_candidate_retire_preflight_v1(o,original,o->successor),ET_M3T_INVALID_STATE));
 CHECK(!o->retire_ticket && !o->successor_ticket);
 CHECK(reject(et_m3t_private_candidate_binding_preflight_v1(o,0,o->p[1],o->handles[0]),ET_M3T_INVALID_ARGUMENT));
 CHECK(reject(et_m3t_private_candidate_binding_preflight_v1(o,14,o->p[0],o->handles[0]),ET_M3T_INVALID_ARGUMENT));
 et_f32_parameter *p=o->p[0];void *h=o->handles[0];
 o->p[0]=o->p[1];o->p[1]=p;o->handles[0]=o->handles[1];o->handles[1]=h;
 CHECK(reject(et_m3t_private_candidate_retire_preflight_v1(o,original,o->successor),ET_M3T_INVALID_STATE));
 o->p[1]=o->p[0];o->p[0]=p;o->handles[1]=o->handles[0];o->handles[0]=h;
 workspace *w=et_m3t_private_workspace_create_v1(o);CHECK(w);
 CHECK(reject(et_m3t_private_candidate_retire_preflight_v1(o,original,o->successor),ET_M3T_INVALID_STATE));
 OK(et_m3t_private_workspace_release_v1(w));
 owner *consumer=et_m3t_private_owner_create_v1(o->successor);CHECK(consumer);
 CHECK(reject(et_m3t_private_candidate_retire_preflight_v1(o,original,o->successor),ET_M3T_INVALID_STATE));
 OK(et_m3t_private_owner_abort_v1(consumer));
 et_f32_tensor_borrow *borrow=NULL;et_f32_tensor_error e;
 OK(et_f32_tensor_borrow_begin_v1(value(o,0),&borrow,&e));
 CHECK(et_m3t_private_candidate_retire_preflight_v1(o,original,o->successor)!=0);
 OK(et_f32_tensor_borrow_end_v1(&borrow,&e));
 for(int i=0;i<14;i++)OK(et_m3t_private_candidate_binding_preflight_v1(o,i,o->p[i],o->handles[i]));
 OK(et_m3t_private_candidate_initializer_retire_preflight_v1(o->successor));
 CHECK(reject(et_m3t_private_candidate_initializer_retire_preflight_v1(o->successor),ET_M3T_INVALID_STATE));
 CHECK(reject(et_m3t_private_candidate_retire_preflight_v1(o,shared->successor,o->successor),ET_M3T_INVALID_ARGUMENT));
 OK(et_m3t_private_candidate_retire_preflight_v1(o,original,o->successor));
 CHECK(reject(et_m3t_private_candidate_retire_preflight_v1(o,original,o->successor),ET_M3T_INVALID_STATE));
 pid_t child=fork();CHECK(child>=0);
 if(child==0) { o->handles[0]=o->handles[1];(void)et_m3t_private_candidate_retire_commit_v1(o);_exit(9); }
 int status=0;CHECK(waitpid(child,&status,0)==child);
 CHECK(WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
 et_f32_parameter *saved[14];memcpy(saved,o->p,sizeof(saved));
 OK(et_m3t_private_candidate_retire_commit_v1(o));
 CHECK(o->r.state==-1 && o->successor_pending==1 && original->r.state==1);
 for(int i=0;i<14;i++)CHECK(!o->p[i] && !et_f32_parameter_is_live_v1(saved[i]));
 CHECK(reject(et_m3t_private_candidate_retire_commit_v1(o),ET_M3T_INVALID_STATE));
 OK(et_m3t_private_candidate_initializer_retire_commit_v1(o->successor));
 CHECK(o->successor->r.state==-1 && original->r.state==1);
 CHECK(reject(et_m3t_private_candidate_initializer_retire_commit_v1(o->successor),ET_M3T_INVALID_STATE));
 OK(et_m3t_private_candidate_initializer_retire_preflight_v1(shared->successor));
 OK(et_m3t_private_candidate_retire_preflight_v1(shared,original,shared->successor));
 OK(et_m3t_private_candidate_retire_commit_v1(shared));
 OK(et_m3t_private_candidate_initializer_retire_commit_v1(shared->successor));
 puts("native sealed owner retirement passed; P1/I2 producer remains separate");return 0;
}
