#!/usr/bin/env bash
# Sourced only by E1B. This admits the base private E3 tuple, its exact
# source-private diagnostic successor, the installed diagnostic tuple, or the
# separate noninstalled retention, fault, overflow, mode/graph, numerical, and
# uniform-tie, min-subnormal, and whole-call-fenv successors.
# Mixed and partial tuples are rejected.
e3_base_prefix="${PROJECT_ROOT}/native/e3_private_package"
e3_diagnostic_prefix="${PROJECT_ROOT}/native/e3_diagnostic_private_package"
e3_public_prefix="${PROJECT_ROOT}/native/e3_diagnostic_public_package"
e3_retention_prefix="${PROJECT_ROOT}/tests/e3_private/retention_package"
e3_fault_prefix="${PROJECT_ROOT}/tests/e3_private/fault_package"
e3_first_eval_mutant_prefix="${PROJECT_ROOT}/tests/e3_private/retention_first_eval_mutant_package"
e3_overflow_prefix="${PROJECT_ROOT}/tests/e3_private/overflow_package"
e3_mode_graph_prefix="${PROJECT_ROOT}/tests/e3_mode_graph/package"
e3_numerical_prefix="${PROJECT_ROOT}/tests/e3_private/numerical_package"
e3_uniform_prefix="${PROJECT_ROOT}/tests/e3_private/uniform_package"
e3_minsubnormal_prefix="${PROJECT_ROOT}/tests/e3_private/minsubnormal_package"
e3_fenv_prefix="${PROJECT_ROOT}/tests/e3_private/fenv_package"
e3_zero_prefix="${PROJECT_ROOT}/tests/e3_private/zero_control_package"
e3_base_inputs=(
  "${PROJECT_ROOT}/native/e3_private_driver_root.esk"
  "${PROJECT_ROOT}/native/e3_private_bridge.c"
  "${e3_base_prefix}_private_renames.txt"
  "${e3_base_prefix}_public_exports.txt"
)
e3_diagnostic_inputs=(
  "${PROJECT_ROOT}/native/e3_diagnostic_private_driver_root.esk"
  "${PROJECT_ROOT}/native/e3_diagnostic_private_bridge.c"
  "${e3_diagnostic_prefix}_private_renames.txt"
  "${e3_diagnostic_prefix}_public_exports.txt"
)
e3_public_inputs=(
  "${PROJECT_ROOT}/native/e3_diagnostic_public_root.esk"
  "${PROJECT_ROOT}/native/e3_diagnostic_public_bridge.c"
  "${e3_public_prefix}_private_renames.txt"
  "${e3_public_prefix}_public_exports.txt"
)
e3_retention_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/retention_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/retention_bridge.c"
  "${e3_retention_prefix}_private_renames.txt"
  "${e3_retention_prefix}_public_exports.txt"
)
e3_fault_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/fault_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/fault_bridge.c"
  "${e3_fault_prefix}_private_renames.txt"
  "${e3_fault_prefix}_public_exports.txt"
)
e3_first_eval_mutant_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/retention_first_eval_mutant_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/retention_first_eval_mutant_bridge.c"
  "${e3_first_eval_mutant_prefix}_private_renames.txt"
  "${e3_first_eval_mutant_prefix}_public_exports.txt"
)
e3_overflow_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/overflow_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/overflow_bridge.c"
  "${e3_overflow_prefix}_private_renames.txt"
  "${e3_overflow_prefix}_public_exports.txt"
)
e3_mode_graph_inputs=(
  "${PROJECT_ROOT}/tests/e3_mode_graph/driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_mode_graph/bridge.c"
  "${e3_mode_graph_prefix}_private_renames.txt"
  "${e3_mode_graph_prefix}_public_exports.txt"
)
e3_numerical_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/numerical_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/numerical_bridge.c"
  "${e3_numerical_prefix}_private_renames.txt"
  "${e3_numerical_prefix}_public_exports.txt"
)
e3_uniform_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/uniform_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/uniform_bridge.c"
  "${e3_uniform_prefix}_private_renames.txt"
  "${e3_uniform_prefix}_public_exports.txt"
)
e3_minsubnormal_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/minsubnormal_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/minsubnormal_bridge.c"
  "${e3_minsubnormal_prefix}_private_renames.txt"
  "${e3_minsubnormal_prefix}_public_exports.txt"
)
e3_fenv_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/fenv_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/fenv_bridge.c"
  "${e3_fenv_prefix}_private_renames.txt"
  "${e3_fenv_prefix}_public_exports.txt"
)
e3_zero_inputs=(
  "${PROJECT_ROOT}/tests/e3_private/zero_control_driver_root.esk"
  "${PROJECT_ROOT}/tests/e3_private/zero_control_bridge.c"
  "${e3_zero_prefix}_private_renames.txt"
  "${e3_zero_prefix}_public_exports.txt"
)
e3_prefix="${e3_base_prefix}"
e3_inputs=("${e3_base_inputs[@]}")
e3_tuple_requested=0
e3_tuple_kind=
e3_diagnostic_tuple=0
e3_public_tuple=0
for e3_raw in "${raw_private_root}" "${raw_package_bridge}" \
    "${raw_package_renames}" "${raw_public_exports}"; do
  for e3_expected in "${e3_base_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == base ]] || \
        die "E3 policy rejects mixed base and diagnostic tuples"
      e3_tuple_kind=base
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_diagnostic_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == diagnostic ]] || \
        die "E3 policy rejects mixed base and diagnostic tuples"
      e3_tuple_kind=diagnostic
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_public_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == public ]] || \
        die "E3 policy rejects mixed private and installed tuples"
      e3_tuple_kind=public
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_retention_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == retention ]] || \
        die "E3 policy rejects mixed retention and canonical tuples"
      e3_tuple_kind=retention
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_fault_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == fault ]] ||
        die "E3 policy rejects mixed fault and other tuples"
      e3_tuple_kind=fault
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_first_eval_mutant_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" ||
         "${e3_tuple_kind}" == retention-first-eval-mutant ]] ||
        die "E3 policy rejects mixed first-eval mutant tuples"
      e3_tuple_kind=retention-first-eval-mutant
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_overflow_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == retention-overflow ]] || \
        die "E3 policy rejects mixed overflow and other tuples"
      e3_tuple_kind=retention-overflow
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_mode_graph_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == mode-graph ]] || \
        die "E3 policy rejects mixed mode/graph and canonical tuples"
      e3_tuple_kind=mode-graph
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_numerical_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == numerical ]] || \
        die "E3 policy rejects mixed numerical and canonical tuples"
      e3_tuple_kind=numerical
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_uniform_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == uniform ]] || \
        die "E3 policy rejects mixed uniform and other tuples"
      e3_tuple_kind=uniform
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_minsubnormal_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == minsubnormal ]] ||
        die "E3 policy rejects mixed min-subnormal and other tuples"
      e3_tuple_kind=minsubnormal
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_fenv_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == fenv ]] || \
        die "E3 policy rejects mixed fenv and other tuples"
      e3_tuple_kind=fenv
      e3_tuple_requested=1
    fi
  done
  for e3_expected in "${e3_zero_inputs[@]}"; do
    if [[ "$(realpath -m -- "${e3_raw}")" == "${e3_expected}" ]]; then
      [[ -z "${e3_tuple_kind}" || "${e3_tuple_kind}" == zero ]] || \
        die "E3 policy rejects mixed zero-control and other tuples"
      e3_tuple_kind=zero
      e3_tuple_requested=1
    fi
  done
done

if [[ "${e3_tuple_kind}" == diagnostic ]]; then
  e3_prefix="${e3_diagnostic_prefix}"
  e3_inputs=("${e3_diagnostic_inputs[@]}")
  e3_diagnostic_tuple=1
elif [[ "${e3_tuple_kind}" == public ]]; then
  e3_prefix="${e3_public_prefix}"
  e3_inputs=("${e3_public_inputs[@]}")
  e3_public_tuple=1
elif [[ "${e3_tuple_kind}" == retention ]]; then
  e3_prefix="${e3_retention_prefix}"
  e3_inputs=("${e3_retention_inputs[@]}")
elif [[ "${e3_tuple_kind}" == fault ]]; then
  e3_prefix="${e3_fault_prefix}"
  e3_inputs=("${e3_fault_inputs[@]}")
elif [[ "${e3_tuple_kind}" == retention-first-eval-mutant ]]; then
  e3_prefix="${e3_first_eval_mutant_prefix}"
  e3_inputs=("${e3_first_eval_mutant_inputs[@]}")
elif [[ "${e3_tuple_kind}" == retention-overflow ]]; then
  e3_prefix="${e3_overflow_prefix}"
  e3_inputs=("${e3_overflow_inputs[@]}")
elif [[ "${e3_tuple_kind}" == mode-graph ]]; then
  e3_prefix="${e3_mode_graph_prefix}"
  e3_inputs=("${e3_mode_graph_inputs[@]}")
elif [[ "${e3_tuple_kind}" == numerical ]]; then
  e3_prefix="${e3_numerical_prefix}"
  e3_inputs=("${e3_numerical_inputs[@]}")
elif [[ "${e3_tuple_kind}" == uniform ]]; then
  e3_prefix="${e3_uniform_prefix}"
  e3_inputs=("${e3_uniform_inputs[@]}")
elif [[ "${e3_tuple_kind}" == minsubnormal ]]; then
  e3_prefix="${e3_minsubnormal_prefix}"
  e3_inputs=("${e3_minsubnormal_inputs[@]}")
elif [[ "${e3_tuple_kind}" == fenv ]]; then
  e3_prefix="${e3_fenv_prefix}"
  e3_inputs=("${e3_fenv_inputs[@]}")
elif [[ "${e3_tuple_kind}" == zero ]]; then
  e3_prefix="${e3_zero_prefix}"
  e3_inputs=("${e3_zero_inputs[@]}")
fi

e3_check_repository_path() {
  local path=$1 part current="${PROJECT_ROOT}"
  [[ "${path}" == "${PROJECT_ROOT}/"* && "${path}" != *'/../'* && \
     "${path}" != *'/./'* && "$(realpath -m -- "${path}")" == "${path}" ]] || \
    die "E3 policy requires exact canonical repository paths"
  local relative=${path#"${PROJECT_ROOT}/"}
  local -a parts
  IFS=/ read -r -a parts <<<"${relative}"
  for part in "${parts[@]}"; do
    current+="/${part}"
    [[ ! -L "${current}" ]] || die "E3 policy rejects symlinked repository inputs"
  done
}

e3_check_native_dependency_path() {
  local raw=$1 current=/ part
  local -a parts
  [[ "${raw}" == /* ]] || die "E3 native depfile path must be absolute"
  IFS=/ read -r -a parts <<<"${raw}"
  for part in "${parts[@]}"; do
    case "${part}" in
      ''|.) continue ;;
      ..) current="${current%/*}"; [[ -n "${current}" ]] || current=/ ;;
      *) current="${current%/}/${part}"
         [[ ! -L "${current}" ]] || die "E3 native depfile uses a symlink alias" ;;
    esac
  done
}

e3_check_zero_generated_dependency() {
  local raw=$1 expected="${e3_generated_source_root}/transformer/module.esk"
  local current=/ part
  local -a parts
  [[ "${e3_tuple_kind}" == zero && "${raw}" == "${expected}" &&
     "${raw}" == /* ]] || die "E3 zero P1 overlay requires its exact lexical path"
  IFS=/ read -r -a parts <<<"${raw}"
  for part in "${parts[@]}"; do
    [[ -n "${part}" ]] || continue
    [[ "${part}" != . && "${part}" != .. ]] ||
      die "E3 zero P1 overlay rejects path aliases"
    current="${current%/}/${part}"
    [[ ! -L "${current}" ]] ||
      die "E3 zero P1 overlay rejects symlink aliases"
  done
  [[ -f "${expected}" ]] || die "E3 zero P1 overlay is absent"
  [[ "$(sha256sum -- "${expected}" | cut -d' ' -f1)" == \
     06f0d533f2ea1c1188ba939709ab814ba36e01391f91b434f7ca80a058c680c8 ]] ||
    die "E3 zero P1 overlay hash differs from the reviewed insertion"
}

e3_normalize_source_dependencies() {
  local dependency generated zero_p1
  [[ -n "${e3_generated_source_root:-}" ]] || \
    die "E3 private generated source was not initialized"
  generated="${e3_generated_source_root}/e3_d2_dataset.esk"
  zero_p1="${e3_generated_source_root}/transformer/module.esk"
  while IFS= read -r dependency; do
    [[ -n "${dependency}" ]] || continue
    if [[ "$(realpath -m -- "${dependency}")" == "${generated}" ]]; then
      printf '@BUILD@/source/e3_d2_dataset.esk\n'
    elif [[ "${e3_tuple_kind}" == zero && "${dependency}" == "${zero_p1}" ]]; then
      e3_check_zero_generated_dependency "${dependency}"
      printf '@BUILD@/source/transformer/module.esk\n'
    elif [[ "${e3_tuple_kind}" == zero &&
            "$(realpath -m -- "${dependency}")" == "${zero_p1}" ]]; then
      die "E3 zero P1 overlay rejects aliased generated dependencies"
    elif [[ ( "${e3_tuple_kind}" == mode-graph ||
              "${e3_tuple_kind}" == retention ||
              "${e3_tuple_kind}" == fault ||
              "${e3_tuple_kind}" == retention-first-eval-mutant ||
              "${e3_tuple_kind}" == retention-overflow ) && \
            "$(realpath -m -- "${dependency}")" == \
              "${e3_generated_source_root}/transformer/module.esk" ]]; then
      [[ "$(sha256sum -- "${dependency}" 2>/dev/null | cut -d' ' -f1)" != \
         06f0d533f2ea1c1188ba939709ab814ba36e01391f91b434f7ca80a058c680c8 ]] ||
        die "E3 zero P1 overlay is forbidden for another tuple"
      printf '@BUILD@/source/transformer/module.esk\n'
    elif [[ "${e3_tuple_kind}" == mode-graph && \
            "$(realpath -m -- "${dependency}")" == \
              "${e3_generated_source_root}/e3_mode_graph_extension.esk" ]]; then
      printf '@BUILD@/source/e3_mode_graph_extension.esk\n'
    else
      e3_check_repository_path "${dependency}"
      printf '%s\n' "${dependency#"${PROJECT_ROOT}/"}"
    fi
  done
}

if [[ "${e3_tuple_requested}" == 1 ]]; then
  [[ "${raw_private_root}" == "${e3_inputs[0]}" && \
     "${raw_package_bridge}" == "${e3_inputs[1]}" && \
     "${raw_package_renames}" == "${e3_inputs[2]}" && \
     "${raw_public_exports}" == "${e3_inputs[3]}" ]] || \
    die "E3 policy requires the exact lexical repository tuple"
  [[ "${#raw_include_dirs[@]}" == 6 && \
     "${raw_include_dirs[0]}" == "${PROJECT_ROOT}/internal/p1/lib" && \
     "${raw_include_dirs[1]}" == "${PROJECT_ROOT}/internal/c1/lib" && \
     "${raw_include_dirs[2]}" == "${PROJECT_ROOT}/internal/t1/lib" && \
     "${raw_include_dirs[3]}" == "${PROJECT_ROOT}/src" && \
     "${raw_include_dirs[4]}" == "${PROJECT_ROOT}/internal/d2/lib" && \
     "${raw_include_dirs[5]}" == "${PROJECT_ROOT}/internal/e3/lib" ]] || \
    die "E3 policy requires exact ordered trusted include roots"
  for e3_path in "${e3_inputs[@]}" "${raw_include_dirs[@]}" \
      "${e3_prefix}_defined_symbols.txt" \
      "${e3_prefix}_undefined_symbols.txt" \
      "${e3_prefix}_public_strings.txt" \
      "${e3_prefix}_native_objects.txt" \
      "${e3_prefix}_archive_members.txt" \
      "${e3_prefix}_source_closure.txt" \
      "${e3_prefix}_native_source_closure.txt"; do
    e3_check_repository_path "${e3_path}"
  done
  if [[ "${e3_public_tuple}" == 1 ]]; then
    e3_check_repository_path "${e3_prefix}_facades.txt"
  fi
  if [[ "${e3_tuple_kind}" == zero &&
        "${E3_ZERO_CONTROL_SANITIZE:-0}" == 1 ]]; then
    e3_check_repository_path \
      "${e3_prefix}_sanitized_noninstrumentation_undefined_symbols.txt"
    e3_check_repository_path \
      "${e3_prefix}_sanitized_instrumentation_undefined_symbols.txt"
  fi
fi
