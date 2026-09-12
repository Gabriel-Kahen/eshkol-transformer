/* Reusable K2 wrapper extension.  The including aggregate must first provide
 * the E1B box helpers and the trusted K2 Scheme C-ABI definitions. */
#define ET_K2_DECLARE_NULLARY(name)                                        \
  extern eshkol_tagged_value_t name(void)
#define ET_K2_DECLARE_UNARY(name)                                          \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t value)
#define ET_K2_DECLARE_TERNARY(name)                                        \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t first,            \
                                     eshkol_tagged_value_t second,           \
                                     eshkol_tagged_value_t third)

ET_K2_DECLARE_NULLARY(et_e1b_private_k2_capability_discover_cabi_v1);
extern eshkol_tagged_value_t et_e1b_private_k2_capability_request_cabi_v1(
    eshkol_tagged_value_t operation, eshkol_tagged_value_t dtype,
    eshkol_tagged_value_t device, eshkol_tagged_value_t shape,
    eshkol_tagged_value_t deterministic);
ET_K2_DECLARE_TERNARY(et_e1b_private_k2_capability_verified_cabi_v1);
ET_K2_DECLARE_TERNARY(et_e1b_private_k2_capability_require_cabi_v1);
ET_K2_DECLARE_UNARY(et_e1b_private_k2_capability_report_entries_cabi_v1);
ET_K2_DECLARE_UNARY(et_e1b_private_k2_capability_entry_name_cabi_v1);
ET_K2_DECLARE_UNARY(et_e1b_private_k2_capability_entry_status_cabi_v1);
ET_K2_DECLARE_UNARY(et_e1b_private_k2_capability_entry_implementation_cabi_v1);
ET_K2_DECLARE_UNARY(et_e1b_private_k2_capability_entry_version_cabi_v1);
ET_K2_DECLARE_UNARY(et_e1b_private_k2_capability_entry_constraints_cabi_v1);
ET_K2_DECLARE_UNARY(et_e1b_private_k2_capability_entry_deterministic_cabi_v1);
ET_K2_DECLARE_UNARY(et_e1b_private_k2_capability_entry_evidence_cabi_v1);

#define ET_K2_PUBLIC_UNARY(name, target)                                   \
  void name(void *input, void *output) {                                   \
    et_e1b_ensure_private_initialized_v1();                                \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));     \
  }

void et_e1b_public_k2_capability_discover_v1(void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_k2_capability_discover_cabi_v1();
}

void et_e1b_public_k2_capability_request_v1(
    void *operation, void *dtype, void *device, void *shape,
    void *deterministic, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_k2_capability_request_cabi_v1(
          *et_e1b_box_value_v1(operation), *et_e1b_box_value_v1(dtype),
          *et_e1b_box_value_v1(device), *et_e1b_box_value_v1(shape),
          *et_e1b_box_value_v1(deterministic));
}

void et_e1b_public_k2_capability_verified_v1(
    void *report, void *capability, void *request, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_k2_capability_verified_cabi_v1(
          *et_e1b_box_value_v1(report), *et_e1b_box_value_v1(capability),
          *et_e1b_box_value_v1(request));
}

void et_e1b_public_k2_capability_require_v1(
    void *report, void *capability, void *request, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_k2_capability_require_cabi_v1(
          *et_e1b_box_value_v1(report), *et_e1b_box_value_v1(capability),
          *et_e1b_box_value_v1(request));
}

ET_K2_PUBLIC_UNARY(et_e1b_public_k2_capability_report_entries_v1,
                   et_e1b_private_k2_capability_report_entries_cabi_v1)
ET_K2_PUBLIC_UNARY(et_e1b_public_k2_capability_entry_name_v1,
                   et_e1b_private_k2_capability_entry_name_cabi_v1)
ET_K2_PUBLIC_UNARY(et_e1b_public_k2_capability_entry_status_v1,
                   et_e1b_private_k2_capability_entry_status_cabi_v1)
ET_K2_PUBLIC_UNARY(et_e1b_public_k2_capability_entry_implementation_v1,
                   et_e1b_private_k2_capability_entry_implementation_cabi_v1)
ET_K2_PUBLIC_UNARY(et_e1b_public_k2_capability_entry_version_v1,
                   et_e1b_private_k2_capability_entry_version_cabi_v1)
ET_K2_PUBLIC_UNARY(et_e1b_public_k2_capability_entry_constraints_v1,
                   et_e1b_private_k2_capability_entry_constraints_cabi_v1)
ET_K2_PUBLIC_UNARY(et_e1b_public_k2_capability_entry_deterministic_v1,
                   et_e1b_private_k2_capability_entry_deterministic_cabi_v1)
ET_K2_PUBLIC_UNARY(et_e1b_public_k2_capability_entry_evidence_v1,
                   et_e1b_private_k2_capability_entry_evidence_cabi_v1)

#undef ET_K2_PUBLIC_UNARY
#undef ET_K2_DECLARE_TERNARY
#undef ET_K2_DECLARE_UNARY
#undef ET_K2_DECLARE_NULLARY
