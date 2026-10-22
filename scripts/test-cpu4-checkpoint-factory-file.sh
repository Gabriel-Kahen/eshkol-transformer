#!/usr/bin/env bash
# Direct normal/repeat only; the reviewed one-use outer launcher owns heavy.lock.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ $# == 1 && "$1" == /* && ! -e "$1" && ! -L "$1" ]] || die 'expected fresh absolute output'
cd "$PROJECT_ROOT"
verify_toolchain
cc= cxx=
resolve_provenance_compilers cc cxx "$(lock_value supported_cc)" "$(lock_value supported_cxx)"
python3 -B -m tests.cpu4_checkpoint_factory.file_driver "$1" \
  "$(eshkol_build_dir)/eshkol-run" "$cc" "$cxx" "$(eshkol_source_dir)"
