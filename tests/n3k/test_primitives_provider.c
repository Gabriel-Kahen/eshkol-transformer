#include "test_common.h"
#include "finite_difference.h"
#include "reference_vectors.h"

#include <fenv.h>
#include <float.h>

static size_t derivatives;
static double maximum_reference_absolute_error;
static const uint64_t embedding_rows[][4] = {{1,2,256,4}, {1,2,2,4}};
static const uint64_t linear_rows[][4] = {{1,2,4,4}, {1,2,4,8}, {1,2,8,4}, {1,2,4,256}};

static void test_metadata(et_kernel_runtime *runtime) {
  static const struct {
    const char *name;
    const char *operations[4];
    size_t operation_count, rank, row_count;
    uint64_t rows[5][4];
  } expected[] = {
    {"n3k.embedding-forward", {"n3k.embedding.forward"}, 1,4,2, {{1,2,256,4},{1,2,2,4}}},
    {"n3k.embedding-backward", {"n3k.embedding.backward"}, 1,4,2, {{1,2,256,4},{1,2,2,4}}},
    {"n3k.linear", {"n3k.linear.backward-no-bias","n3k.linear.forward-no-bias"}, 2,4,4,
      {{1,2,4,4},{1,2,4,8},{1,2,8,4},{1,2,4,256}}},
    {"n3k.gelu", {"n3k.gelu.backward","n3k.gelu.forward"}, 2,3,1, {{1,2,8}}},
    {"n3k.residual", {"n3k.residual.backward","n3k.residual.forward"}, 2,3,1, {{1,2,4}}},
    {"n3k.head-layout", {"n3k.heads.merge.backward","n3k.heads.merge.forward",
      "n3k.heads.split.backward","n3k.heads.split.forward"}, 4,4,1, {{1,2,2,2}}},
    {"n3k.activation-sum", {"n3k.sum2.backward","n3k.sum2.forward","n3k.sum3.backward","n3k.sum3.forward"},
      4,3,1, {{1,2,4}}},
    {"n3k.tied-sum", {"n3k.sum2.backward","n3k.sum2.forward"}, 2,2,1, {{256,4}}},
    {"n3k.matrix-init", {"n3k.matrix-init.uniform"}, 1,2,5, {{256,4},{2,4},{4,4},{8,4},{4,8}}},
  };
  const et_kernel_provider_v1 *provider = et_n3k_kernel_provider_v1();
  CHECK(ET_N3K_PRIMITIVES_ABI_MAJOR == 1u && ET_N3K_PRIMITIVES_ABI_MINOR == 0u);
  CHECK(provider && provider == et_n3k_kernel_provider_v1());
  CHECK(provider->struct_size == sizeof(*provider));
  CHECK(provider->abi_major == 1u && provider->abi_minor == 0u && provider->required_features == 0u);
  CHECK(strcmp(provider->name,"n3k.cpu-f32.serial") == 0);
  CHECK(strcmp(provider->version,"1.0") == 0);
  CHECK(strcmp(provider->evidence,"N3K:cpu-f32-diagnostic-v1") == 0);
  CHECK(provider->capability_count == COUNT(expected));
  CHECK(provider->capability_stride == sizeof(et_kernel_capability_v1));
  CHECK(provider->capability_bytes == COUNT(expected) * sizeof(et_kernel_capability_v1));
  CHECK(provider->validate_call && provider->invoke_call);
  size_t combinations = 0;
  for (size_t index = 0; index < COUNT(expected); index++) {
    const et_kernel_capability_v1 *cap = et_kernel_runtime_capability_find(runtime,expected[index].name);
    CHECK(cap && cap->status == ET_KERNEL_CAPABILITY_VERIFIED && cap->deterministic == 1u);
    CHECK(strcmp(cap->implementation,"n3k.cpu-f32.serial") == 0);
    CHECK(strcmp(cap->version,"1.0") == 0 && strcmp(cap->evidence,"N3K:cpu-f32-diagnostic-v1") == 0);
    CHECK(cap->operation_count == expected[index].operation_count);
    CHECK(cap->dtype_count == 1u && cap->device_count == 1u);
    CHECK(strcmp(cap->dtypes[0],"f32") == 0 && strcmp(cap->devices[0],"cpu") == 0);
    CHECK(cap->shape_range_count == expected[index].row_count);
    for (size_t op = 0; op < cap->operation_count; op++)
      CHECK(strcmp(cap->operations[op],expected[index].operations[op]) == 0);
    for (size_t row = 0; row < cap->shape_range_count; row++) {
      CHECK(cap->shape_ranges[row].rank == expected[index].rank);
      size_t matches = 0;
      for (size_t dimension = 0; dimension < expected[index].rank; dimension++) {
        const et_kernel_dimension_range_v1 *range = &cap->shape_ranges[row].dimensions[dimension];
        CHECK(range->maximum == range->minimum && range->maximum_unbounded == 0u);
      }
      for (size_t candidate = 0; candidate < expected[index].row_count; candidate++) {
        int same = 1;
        for (size_t dimension = 0; dimension < expected[index].rank; dimension++)
          if (cap->shape_ranges[row].dimensions[dimension].minimum != expected[index].rows[candidate][dimension]) same = 0;
        matches += (size_t)same;
      }
      CHECK(matches == 1u);
      for (size_t op = 0; op < cap->operation_count; op++) {
        uint64_t shape[5] = {0};
        memcpy(shape,expected[index].rows[row],expected[index].rank * sizeof(*shape));
        et_kernel_request_v1 request = rq(cap->operations[op],expected[index].rank,shape);
        const et_kernel_capability_v1 *found = NULL;
        et_kernel_error error;
        CHECK(et_kernel_runtime_capability_require(runtime,cap->name,&request,&found,&error) == 0);
        CHECK(found == cap);
        combinations++;
        for (size_t dimension = 0; dimension < request.rank; dimension++) {
          const uint64_t saved = shape[dimension];
          const uint64_t neighbors[3] = {0,saved - 1u,saved + 1u};
          for (size_t neighbor = 0; neighbor < COUNT(neighbors); neighbor++) {
            shape[dimension] = neighbors[neighbor];
            CHECK(et_kernel_runtime_capability_require(runtime,cap->name,&request,&found,&error) != 0);
            CHECK(error.category == ET_KERNEL_ERROR_UNSUPPORTED && error.code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
          }
          shape[dimension] = saved;
        }
        request.rank--;
        CHECK(et_kernel_runtime_capability_require(runtime,cap->name,&request,&found,&error) != 0);
        request.rank += 2;
        shape[request.rank - 1] = 1;
        CHECK(et_kernel_runtime_capability_require(runtime,cap->name,&request,&found,&error) != 0);
        request.rank--;
        request.dtype = "f64";
        CHECK(et_kernel_runtime_capability_require(runtime,cap->name,&request,&found,&error) != 0);
        request.dtype = "f32";
        request.device = "cuda";
        CHECK(et_kernel_runtime_capability_require(runtime,cap->name,&request,&found,&error) != 0);
        request.device = "cpu";
        request.operation = "n3k.unadvertised";
        CHECK(et_kernel_runtime_capability_require(runtime,cap->name,&request,&found,&error) != 0);
      }
    }
  }
  CHECK(combinations == 31u);
}

static const char *reference_capability(const char *operation) {
  if (strcmp(operation,"embedding.forward") == 0) return "n3k.embedding-forward";
  if (strcmp(operation,"embedding.backward") == 0) return "n3k.embedding-backward";
  if (strncmp(operation,"linear.",7) == 0) return "n3k.linear";
  if (strncmp(operation,"gelu.",5) == 0) return "n3k.gelu";
  if (strncmp(operation,"residual.",9) == 0) return "n3k.residual";
  CHECK(0);
  return NULL;
}

static void test_independent_references(et_kernel_runtime *runtime) {
  CHECK(ET_N3K_REFERENCE_CASE_COUNT == 20u);
  for (size_t index = 0; index < ET_N3K_REFERENCE_CASE_COUNT; index++) {
    const struct et_n3k_ref_case *record = &et_n3k_reference_cases[index];
    float output[2][1024], first[2][1024];
    et_kernel_tensor_view_v1 inputs[3], outputs[2];
    unsigned char input_snapshot[3][4096];
    char operation[80];
    CHECK(snprintf(operation,sizeof(operation),"n3k.%s",record->operation) > 0);
    for (size_t i = 0; i < record->input_count; i++) {
      const struct et_n3k_ref_tensor *tensor = &record->inputs[i];
      const size_t bytes = tensor->count * (strcmp(tensor->dtype,"f32") == 0 ? 4u : 8u);
      CHECK(bytes <= sizeof(input_snapshot[i]));
      memcpy(input_snapshot[i],tensor->data,bytes);
      inputs[i] = tv((void *)tensor->data,bytes,tensor->dtype,tensor->rank,tensor->shape);
    }
    for (size_t i = 0; i < record->output_count; i++) {
      const struct et_n3k_ref_tensor *tensor = &record->outputs[i];
      CHECK(tensor->count <= COUNT(output[i]));
      outputs[i] = tv(output[i],tensor->count * sizeof(float),"f32",tensor->rank,tensor->shape);
    }
    et_kernel_request_v1 request = rq(operation,record->rank,record->row);
    et_kernel_call_v1 call = kc(reference_capability(record->operation),&request,
                                inputs,record->input_count,outputs,record->output_count);
    for (size_t repetition = 0; repetition < 2; repetition++) {
      memset(output,repetition == 0u ? 0xa5 : 0x5a,sizeof(output));
      run(runtime,&call);
      for (size_t i = 0; i < record->output_count; i++) {
        const float *expected = record->outputs[i].data;
        for (size_t element = 0; element < record->outputs[i].count; element++) {
          const double actual = output[i][element], ideal = expected[element];
          const double difference = fabs(actual - ideal);
          const double tolerance = 4e-6 + 3e-5 * fmax(fabs(actual),fabs(ideal));
          if (!(isfinite(actual) && difference <= tolerance))
            fprintf(stderr,"reference %s output %zu element %zu actual %.9g ideal %.9g tolerance %.9g\n",
                    record->name,i,element,actual,ideal,tolerance);
          CHECK(isfinite(actual) && difference <= tolerance);
          if (difference > maximum_reference_absolute_error) maximum_reference_absolute_error = difference;
          if (strncmp(record->operation,"embedding.",10) == 0 ||
              strcmp(record->operation,"residual.backward") == 0)
            CHECK(fbits(output[i][element]) == fbits(expected[element]));
        }
        if (repetition == 0u) memcpy(first[i],output[i],outputs[i].byte_length);
        else CHECK(memcmp(first[i],output[i],outputs[i].byte_length) == 0);
      }
      for (size_t i = 0; i < record->input_count; i++)
        CHECK(memcmp(input_snapshot[i],inputs[i].data,inputs[i].byte_length) == 0);
    }
  }
}

typedef struct forward_context {
  et_kernel_runtime *runtime;
  et_kernel_call_v1 call;
  et_kernel_request_v1 request;
  et_kernel_tensor_view_v1 inputs[3], output;
} forward_context;

static void prepare_forward(forward_context *context, et_kernel_runtime *runtime,
                            const char *capability, const char *operation,
                            size_t rank, const uint64_t *shape,
                            const et_kernel_tensor_view_v1 *inputs, size_t input_count,
                            et_kernel_tensor_view_v1 output) {
  context->runtime = runtime;
  context->request = rq(operation,rank,shape);
  memcpy(context->inputs,inputs,input_count * sizeof(*inputs));
  context->output = output;
  context->call = kc(capability,&context->request,context->inputs,input_count,&context->output,1);
}

static int actual_forward(void *opaque, float *output) {
  forward_context *context = opaque;
  et_kernel_error error;
  context->output.data = output;
  const int status = et_kernel_runtime_dispatch(context->runtime,&context->call,&error);
  if (status != 0) fprintf(stderr,"native perturbed forward failed: %s\n",error.message);
  return status;
}

static void check_derivatives(const char *name, forward_context *context, float *primal,
                              size_t count, const float *upstream, size_t output_count,
                              const float *analytic, float step, double absolute, double relative) {
  float saved[1024];
  CHECK(count <= COUNT(saved));
  memcpy(saved,primal,count * sizeof(float));
  CHECK(n3k_test_central_vjp(name,actual_forward,context,primal,count,upstream,
                            output_count,analytic,step,absolute,relative));
  CHECK(memcmp(saved,primal,count * sizeof(float)) == 0);
  derivatives += count;
}

static void test_embedding_gradients(et_kernel_runtime *runtime) {
  const uint64_t ids_shape[2] = {1,2}, y_shape[3] = {1,2,4};
  for (size_t row = 0; row < COUNT(embedding_rows); row++) {
    const size_t vocabulary = embedding_rows[row][2];
    const uint64_t weight_shape[2] = {vocabulary,4};
    float weight[1024], upstream[8], dweight[1024];
    for (size_t i = 0; i < vocabulary * 4; i++) weight[i] = (float)((int)(i % 31) - 15) * 0x1p-5f;
    for (size_t i = 0; i < 8; i++) upstream[i] = (float)((int)i - 4) * 0x1p-4f;
    for (size_t repeated = 0; repeated < 2; repeated++) {
      int64_t ids[2] = {0,(int64_t)vocabulary - 1};
      if (repeated) ids[0] = ids[1] = vocabulary == 256u ? 7 : 1;
      et_kernel_tensor_view_v1 inputs[2] = {
        tv(ids,sizeof(ids),"i64",2,ids_shape), tv(weight,vocabulary * 4 * sizeof(float),"f32",2,weight_shape)};
      forward_context context;
      prepare_forward(&context,runtime,"n3k.embedding-forward","n3k.embedding.forward",4,embedding_rows[row],
                      inputs,2,tv(NULL,sizeof(upstream),"f32",3,y_shape));
      inputs[1] = tv(upstream,sizeof(upstream),"f32",3,y_shape);
      et_kernel_tensor_view_v1 output = tv(dweight,vocabulary * 4 * sizeof(float),"f32",2,weight_shape);
      et_kernel_request_v1 request = rq("n3k.embedding.backward",4,embedding_rows[row]);
      et_kernel_call_v1 call = kc("n3k.embedding-backward",&request,inputs,2,&output,1);
      run(runtime,&call);
      check_derivatives("embedding.weight",&context,weight,vocabulary * 4,upstream,8,dweight,0x1p-6f,0,0);
      for (size_t v = 0; v < vocabulary; v++)
        if (v != (size_t)ids[0] && v != (size_t)ids[1])
          for (size_t d = 0; d < 4; d++) CHECK(fbits(dweight[v * 4 + d]) == 0u);
    }
  }
}

static void test_linear_gradients_and_order(et_kernel_runtime *runtime) {
  for (size_t row = 0; row < COUNT(linear_rows); row++) {
    const size_t din = linear_rows[row][2], dout = linear_rows[row][3];
    const uint64_t x_shape[3] = {1,2,din}, weight_shape[2] = {dout,din}, y_shape[3] = {1,2,dout};
    float x[16], weight[1024], upstream[512], dx[16], dweight[1024], y[512];
    for (size_t i = 0; i < 2 * din; i++) x[i] = (float)((int)(i * 13 % 31) - 15) * 0x1p-4f;
    for (size_t i = 0; i < din * dout; i++) weight[i] = (float)((int)(i * 7 % 31) - 15) * 0x1p-5f;
    for (size_t i = 0; i < 2 * dout; i++) upstream[i] = (float)((int)(i * 5 % 23) - 11) * 0x1p-4f;
    et_kernel_tensor_view_v1 inputs[3] = {
      tv(x,2 * din * sizeof(float),"f32",3,x_shape),
      tv(weight,din * dout * sizeof(float),"f32",2,weight_shape),
      tv(upstream,2 * dout * sizeof(float),"f32",3,y_shape)};
    et_kernel_tensor_view_v1 outputs[2] = {
      tv(dx,2 * din * sizeof(float),"f32",3,x_shape), tv(dweight,din * dout * sizeof(float),"f32",2,weight_shape)};
    et_kernel_request_v1 request = rq("n3k.linear.backward-no-bias",4,linear_rows[row]);
    et_kernel_call_v1 backward = kc("n3k.linear",&request,inputs,3,outputs,2);
    run(runtime,&backward);
    forward_context context;
    prepare_forward(&context,runtime,"n3k.linear","n3k.linear.forward-no-bias",4,linear_rows[row],
                    inputs,2,tv(NULL,2 * dout * sizeof(float),"f32",3,y_shape));
    check_derivatives("linear.x",&context,x,2 * din,upstream,2 * dout,dx,0x1p-6f,0,0);
    check_derivatives("linear.weight",&context,weight,din * dout,upstream,2 * dout,dweight,0x1p-6f,0,0);
    memset(x,0,sizeof(x)); memset(weight,0,sizeof(weight)); memset(upstream,0,sizeof(upstream));
    x[0] = 0x1p24f; x[1] = 1.0f; x[2] = -0x1p24f; x[3] = 2.0f;
    for (size_t i = 0; i < din; i++) weight[i] = 1.0f;
    CHECK(actual_forward(&context,y) == 0);
    CHECK(fbits(y[0]) == fbits(2.0f)); /* Increasing i with separate f32 adds, not pairwise/f64. */
    memset(weight,0,sizeof(weight));
    for (size_t i = 0; i < dout; i++) weight[i * din] = 1.0f;
    upstream[0] = 0x1p24f; upstream[1] = 1.0f; upstream[2] = -0x1p24f; upstream[3] = 2.0f;
    run(runtime,&backward);
    CHECK(fbits(dx[0]) == fbits(2.0f));
  }
}

static void test_gelu_and_residual_gradients(et_kernel_runtime *runtime) {
  const uint64_t gelu_shape[3] = {1,2,8}, residual_shape[3] = {1,2,4};
  float x[16] = {-3,-2,-1.75f,-0.75f,-0.25f,-0.0f,0,0.125f,0.25f,0.5f,0.75f,1,1.25f,1.75f,2,3};
  float upstream[16], dx[16], branch[8], dbranch[8], y[16];
  for (size_t i = 0; i < 16; i++) upstream[i] = (float)((int)i - 8) * 0x1p-4f;
  et_kernel_tensor_view_v1 inputs[2] = {tv(x,sizeof(x),"f32",3,gelu_shape),tv(upstream,sizeof(upstream),"f32",3,gelu_shape)};
  et_kernel_tensor_view_v1 output = tv(dx,sizeof(dx),"f32",3,gelu_shape);
  et_kernel_request_v1 request = rq("n3k.gelu.backward",3,gelu_shape);
  et_kernel_call_v1 call = kc("n3k.gelu",&request,inputs,2,&output,1);
  run(runtime,&call);
  forward_context context;
  prepare_forward(&context,runtime,"n3k.gelu","n3k.gelu.forward",3,gelu_shape,inputs,1,
                  tv(NULL,sizeof(x),"f32",3,gelu_shape));
  check_derivatives("gelu.x",&context,x,16,upstream,16,dx,0x1p-7f,2e-4,2e-3);
  CHECK(actual_forward(&context,y) == 0);
  CHECK(fbits(y[5]) == UINT32_C(0x80000000));
  CHECK(fbits(y[6]) == 0u);
  for (size_t i = 0; i < 8; i++) {
    x[i] = (float)((int)i - 4) * 0x1p-4f;
    branch[i] = (float)((int)(3 * i) - 8) * 0x1p-5f;
  }
  inputs[0] = tv(x,sizeof(branch),"f32",3,residual_shape);
  inputs[1] = tv(branch,sizeof(branch),"f32",3,residual_shape);
  prepare_forward(&context,runtime,"n3k.residual","n3k.residual.forward",3,residual_shape,inputs,2,
                  tv(NULL,sizeof(branch),"f32",3,residual_shape));
  et_kernel_tensor_view_v1 back_input = tv(upstream,sizeof(branch),"f32",3,residual_shape);
  et_kernel_tensor_view_v1 back_outputs[2] = {
    tv(dx,sizeof(branch),"f32",3,residual_shape),tv(dbranch,sizeof(branch),"f32",3,residual_shape)};
  request = rq("n3k.residual.backward",3,residual_shape);
  call = kc("n3k.residual",&request,&back_input,1,back_outputs,2);
  run(runtime,&call);
  CHECK(memcmp(dx,upstream,sizeof(branch)) == 0 && memcmp(dbranch,upstream,sizeof(branch)) == 0);
  check_derivatives("residual.skip",&context,x,8,upstream,8,dx,0x1p-6f,0,0);
  check_derivatives("residual.branch",&context,branch,8,upstream,8,dbranch,0x1p-6f,0,0);
  context.inputs[1] = context.inputs[0];
  for (size_t i = 0; i < 8; i++) dx[i] += dbranch[i];
  check_derivatives("residual.shared-primal",&context,x,8,upstream,8,dx,0x1p-6f,0,0);
}

int main(void) {
  CHECK(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && sizeof(float) == 4u);
  CHECK(fegetround() == FE_TONEAREST);
  et_kernel_runtime *runtime = runtime_new();
  test_metadata(runtime);
  test_independent_references(runtime);
  test_embedding_gradients(runtime);
  test_linear_gradients_and_order(runtime);
  test_gelu_and_residual_gradients(runtime);
  CHECK(derivatives == 3248u);
  et_kernel_runtime_destroy(runtime);
  printf("N3K primitives: %zu checks, %zu actual-native central derivatives; max reference absolute error %.9g\n",
         checks,derivatives,maximum_reference_absolute_error);
  return 0;
}
