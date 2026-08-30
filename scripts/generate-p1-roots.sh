#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
template="${project_root}/templates/p1/module_roots.esk.tmpl"
manifest="${project_root}/tools/p1/module_surface.tsv"
public_output="${project_root}/lib/transformer/module.esk"
trusted_output="${project_root}/internal/p1/lib/transformer/module.esk"
mode="${1:---write}"

case "${mode}" in
  --write|--check) ;;
  *) printf 'usage: %s [--write|--check]\n' "$0" >&2; exit 2 ;;
esac

for marker in P1_PUBLIC_BEGIN P1_PUBLIC_END P1_TRUSTED_BEGIN P1_TRUSTED_END; do
  count="$(grep -c "^@@${marker}@@$" "${template}")"
  [[ "${count}" == 1 ]] || {
    printf 'P1 template marker %s count is %s, expected 1\n' \
      "${marker}" "${count}" >&2
    exit 1
  }
done

public_begin="$(grep -n '^@@P1_PUBLIC_BEGIN@@$' "${template}" | cut -d: -f1)"
public_end="$(grep -n '^@@P1_PUBLIC_END@@$' "${template}" | cut -d: -f1)"
trusted_begin="$(grep -n '^@@P1_TRUSTED_BEGIN@@$' "${template}" | cut -d: -f1)"
trusted_end="$(grep -n '^@@P1_TRUSTED_END@@$' "${template}" | cut -d: -f1)"
if ! (( public_begin < public_end && public_end < trusted_begin && \
       trusted_begin < trusted_end )); then
  printf 'P1 template roots overlap or have noncanonical marker order\n' >&2
  exit 1
fi

if awk -F '\t' 'NF != 2 || ($1 != "public" && $1 != "internal") || \
                    $2 == "" || seen[$2]++ { exit 1 }
                 $1 == "public" { public_count++ }
                 $1 == "internal" { internal_count++ }
                 END { if (public_count != 17 || internal_count != 26) exit 1 }' \
    "${manifest}"; then
  :
else
  printf 'P1 surface manifest is malformed, duplicated, or changed in count\n' >&2
  exit 1
fi

temporary="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-p1-roots.XXXXXX")"
trap 'rm -rf -- "${temporary}"' EXIT

awk '/^@@P1_PUBLIC_BEGIN@@$/ { active=1; next }
     /^@@P1_PUBLIC_END@@$/ { active=0; next }
     active { print }' "${template}" >"${temporary}/module.esk"
awk '/^@@P1_TRUSTED_BEGIN@@$/ { active=1; next }
     /^@@P1_TRUSTED_END@@$/ { active=0; next }
     active { print }' "${template}" >"${temporary}/module_internal.esk"

[[ -s "${temporary}/module.esk" && -s "${temporary}/module_internal.esk" ]] || {
  printf 'P1 root generation produced an empty output\n' >&2
  exit 1
}

extract_provides() {
  awk '
    /^\(provide([[:space:]]|$)/ { active=1 }
    active {
      line=$0
      gsub(/[()]/, " ", line)
      count=split(line, fields, /[[:space:]]+/)
      for (index=1; index<=count; index++) {
        if (fields[index] != "" && fields[index] != "provide") {
          print fields[index]
        }
      }
      if ($0 ~ /\)[[:space:]]*$/) { exit }
    }
  ' "$1"
}

awk -F '\t' '$1 == "public" { print $2 }' "${manifest}" \
  >"${temporary}/expected-public"
awk -F '\t' '{ print $2 }' "${manifest}" >"${temporary}/expected-trusted"
extract_provides "${temporary}/module.esk" >"${temporary}/actual-public"
extract_provides "${temporary}/module_internal.esk" >"${temporary}/actual-trusted"
if ! cmp --silent "${temporary}/expected-public" "${temporary}/actual-public"; then
  printf 'generated public provide surface differs from the exact manifest\n' >&2
  exit 1
fi
if ! cmp --silent "${temporary}/expected-trusted" "${temporary}/actual-trusted"; then
  printf 'generated trusted provide surface differs from the exact manifest\n' >&2
  exit 1
fi

while IFS=$'\t' read -r visibility name; do
  [[ -n "${visibility}" && -n "${name}" ]] || {
    printf 'malformed P1 surface manifest\n' >&2
    exit 1
  }
  case "${visibility}" in
    public)
      grep -F "${name}" "${temporary}/module.esk" >/dev/null || {
        printf 'generated public root omits %s\n' "${name}" >&2
        exit 1
      }
      grep -F "${name}" "${temporary}/module_internal.esk" >/dev/null || {
        printf 'generated trusted root omits public name %s\n' "${name}" >&2
        exit 1
      }
      ;;
    internal)
      if grep -F "${name}" "${temporary}/module.esk" >/dev/null; then
        printf 'generated public root contains trusted name %s\n' "${name}" >&2
        exit 1
      fi
      grep -F "${name}" "${temporary}/module_internal.esk" >/dev/null || {
        printf 'generated trusted root omits internal name %s\n' "${name}" >&2
        exit 1
      }
      ;;
    *) printf 'unknown P1 manifest visibility %s\n' "${visibility}" >&2; exit 1 ;;
  esac
done <"${manifest}"

if grep -E 'p1-native-|module_core|p1-core-dispatch|bootstrap-internal' \
    "${temporary}/module.esk" >/dev/null; then
  printf 'generated public root contains a private bridge identifier\n' >&2
  exit 1
fi

if grep -E '\(require[[:space:]]+transformer\.module(_internal)?\)' \
    "${temporary}/module.esk" "${temporary}/module_internal.esk" >/dev/null; then
  printf 'generated P1 roots import one another\n' >&2
  exit 1
fi

if [[ "${mode}" == --check ]]; then
  cmp --silent "${temporary}/module.esk" "${public_output}" || {
    printf 'generated public P1 root is stale\n' >&2
    exit 1
  }
  cmp --silent "${temporary}/module_internal.esk" "${trusted_output}" || {
    printf 'generated trusted P1 root is stale\n' >&2
    exit 1
  }
else
  mkdir -p "$(dirname -- "${trusted_output}")"
  mv -f "${temporary}/module.esk" "${public_output}"
  mv -f "${temporary}/module_internal.esk" "${trusted_output}"
fi
