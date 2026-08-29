/usr/bin/bash -c 'set +e
compiler=/home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-build-minimal/eshkol-run
"$compiler" --features
features_status=$?
printf "features_exit=%s\n" "$features_status"
"$compiler" --abi-fingerprint
abi_status=$?
printf "abi_exit=%s\n" "$abi_status"
exit $((features_status != 0 || abi_status != 0))'
