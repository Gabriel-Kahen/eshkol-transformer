#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/cli3-native-closure.XXXXXX")"
trap 'rm -rf -- "${scratch}"' EXIT
PROJECT_ROOT="${scratch}/repo"
e1b_source="${scratch}/eshkol-src"
mkdir -p "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/native" \
  "${e1b_source}/inc" "${scratch}/outside"
touch "${PROJECT_ROOT}/native/header.h" "${e1b_source}/inc/header.h" \
  "${scratch}/outside/header.h"
ln -s native "${PROJECT_ROOT}/alias"
ln -s inc "${e1b_source}/alias"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/cli3-native-closure-policy.sh"

accepted="$(cli3_canonical_native_dependency_path \
  "${PROJECT_ROOT}/src/../native/header.h")"
[[ "${accepted}" == "${PROJECT_ROOT}/native/header.h" ]]
accepted="$(cli3_canonical_native_dependency_path \
  "${e1b_source}/inc/../inc/header.h")"
[[ "${accepted}" == "${e1b_source}/inc/header.h" ]]

reject() {
  local name=$1 path=$2 expected=$3
  if ( cli3_canonical_native_dependency_path "${path}" ) \
      >"${scratch}/${name}.stdout" 2>"${scratch}/${name}.stderr"; then
    die "${name}: unsafe native dependency was accepted"
  fi
  [[ ! -s "${scratch}/${name}.stdout" ]]
  grep -Fx "error: ${expected}" "${scratch}/${name}.stderr" >/dev/null
}
reject escape "${PROJECT_ROOT}/native/../../outside/header.h" \
  'CLI3 native depfile path escapes its reviewed source root'
reject cross-root "${PROJECT_ROOT}/native/../../eshkol-src/inc/header.h" \
  'CLI3 native depfile path escapes its reviewed source root'
reject alias "${PROJECT_ROOT}/src/../alias/header.h" \
  'CLI3 native depfile uses a symlink alias'
reject runtime-escape "${e1b_source}/inc/../../outside/header.h" \
  'CLI3 native depfile path escapes its reviewed source root'
reject runtime-alias "${e1b_source}/inc/../alias/header.h" \
  'CLI3 native depfile uses a symlink alias'
reject outside "${scratch}/outside/header.h" \
  'CLI3 native depfile path is outside reviewed source roots'
printf 'CLI3-NATIVE-CLOSURE-POLICY-PASS\n'
