#!/usr/bin/env python3
"""Check the real source dependencies of the proposed public G2 facade."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text()


terminal = source("native/g3c4_p2g2_terminal_close_extension.esk")
public = source("native/g3c4_public_generation_dispatch_extension.esk")
adapter = source("native/g3c4_public_output_adapter_extension.esk")
g2_adapter = source("native/g3c4_public_g2_output_adapter_extension.esk")
g2_owner = source("native/g3c4_public_g2_result_owner_extension.esk")
native = source("src/eshkol_transformer/g3c4_model_owner.c")
manual_root = source("native/g3g_manual_package_root.esk")
installed_root = source("native/g3g_g0_package_root.esk")
manual_facade = source("native/g3g_manual_public_extension.esk")
facade = source("lib/transformer/generation.esk")
manual_generation = source("native/g3g_manual_facades/transformer/generation.esk")
manual_transport = source(
    "native/g3g_manual_facades/transformer/diagnostic_transport.esk"
)
contract = source("docs/g3/G3_C4_P2_G2_PUBLIC_FACADE_CONTRACT.md")
roadmap = source("docs/ROADMAP.md")

private_operations = (
    "g3c4-p2g2-terminal-generate-internal",
    "g3c4-p2g2-terminal-snapshot-internal",
    "g3c4-p2g2-terminal-release-internal!",
    "g3c4-p2g2-terminal-ids-clone-internal",
    "g3c4-p2g2-terminal-ids-copy-internal",
    "g3c4-p2g2-terminal-ids-release-internal!",
)
for operation in private_operations:
    assert re.search(rf"\(define \({re.escape(operation)}\s", terminal), operation
    assert operation in contract, operation

for symbol in (
    "et_g3c4_private_p2g2_terminal_ids_clone_v1",
    "et_g3c4_private_p2g2_terminal_ids_clone_copy_v1",
    "et_g3c4_private_p2g2_terminal_ids_clone_release_v1",
):
    assert f":real {symbol})" in terminal, symbol
    assert symbol in native and symbol in contract, symbol

# The P2/G1 route remains fixed while only the explicit G2 profile reaches
# the terminal producer and rank-one result owner.
assert "diagnostic-c4-p2g1" in public
assert "diagnostic-c4-p2g2" in public
assert "(= (vector-ref policy 5) 1)" in public
assert "(eq? (vector-ref policy 6) #f)" in public
assert "(g3c4-p2g1-generate-internal generator input)" in public
assert "(g3c4-p2g2-terminal-generate-internal generator input)" in public
assert "(g3c4-public-g2-generator? generator)" in public
constructor = public[public.index("(define (g3c4-public-generator-create "):]
assert constructor.index("g3c4-model-entry-live model") < constructor.index("g3c4-public-profile-config config")
assert constructor.index("g3c4-public-profile-config config") < constructor.index("g3c4-generator-create-with-source")
assert "(= (vector-ref copy 2) 1) (= (vector-ref copy 3) 3)" in adapter
assert "const uint64_t shape[1] = {1u};" in native
assert "(kind == ET_G3C4_LENGTHS_KIND && value == 1)" in native
assert "(kind == ET_G3C4_CACHE_LENGTHS_KIND && value == 3)" in native
assert "i64[2]" in contract and "kind-9" in contract

# The diagnostic manual aggregate now contains the candidate. The older C2
# package remains separate and the public arities are unchanged.
assert '(load "g3c4_public_generation_dispatch_extension.esk")' in manual_root
assert '(load "g3c4_public_output_adapter_extension.esk")' in manual_root
for dependency in (
    "g3c4_p2g2_ownership_extension.esk",
    "g3c4_p2g2_protected_scope_extension.esk",
    "g3c4_p2g2_t1_prefix_coordinator_extension.esk",
    "g3c4_p2g2_terminal_close_extension.esk",
    "g3c4_public_g2_result_owner_extension.esk",
    "g3c4_public_g2_output_adapter_extension.esk",
):
    assert f'(load "{dependency}")' in manual_root, dependency
assert manual_root.index('g3c4_p2g2_terminal_close_extension.esk') < manual_root.index('g3c4_public_g2_output_adapter_extension.esk')
assert manual_root.index('g3c4_public_output_adapter_extension.esk') < manual_root.index('g3c4_public_g2_result_owner_extension.esk')
assert manual_root.index('g3c4_public_g2_output_adapter_extension.esk') < manual_root.index('g3c4_public_generation_dispatch_extension.esk')
assert "g3c4_p2g2_terminal_close_extension.esk" not in installed_root
assert "g3c4-public-g2-output-access shell kind operation" in adapter
assert "g3c4-p2g2-terminal-release-internal! output" in adapter
for operation in (
    "g3c4-p2g2-terminal-snapshot-internal",
    "g3c4-p2g2-terminal-ids-clone-internal",
    "g3c4-p2g2-terminal-ids-copy-internal",
    "g3c4-p2g2-terminal-ids-release-internal!",
):
    assert operation in g2_adapter
assert "(g3c4-public-g2-result-publish" in g2_adapter
assert "(g3c4-native-result-rng-create" in g2_adapter
assert "et_g3c4_private_result_tensor_create_rank1_v1" in g2_owner
assert "g3c4-public-g2-check-source source operation" in g2_owner
assert "g3c4-public-g2-source-checker 0" in g2_adapter
assert g2_adapter.index("g3c4-p2g2-terminal-ids-copy-internal") < g2_adapter.rindex("g3c4-p2g2-terminal-ids-release-internal!")
assert "g3c4-p2g2-terminal-entry-live source operation" in g2_adapter
assert "public_g2_runtime.esk" in source("scripts/test-g3g-manual-public.sh")
assert "G3G-MANUAL-PUBLIC-G2-PASS" in source("tests/g3g_manual/public_g2_runtime.esk")
assert "second EOS remains a logical G2 result" in source("tests/g3g_manual/public_g2_runtime.esk")
assert "g3c4-public-transport-member generator" in manual_facade
assert "g3c4-public-model-member generator" in manual_facade
assert "(g3c4-public-generator-generate generator input)" in manual_facade
assert "(g3c4-public-generator-create model tokenizer config)" in manual_facade
for operation in (
    "generation-output-ids",
    "generation-output-lengths",
    "generation-output-text",
    "generation-output-rng",
    "generation-output-cache-lengths",
    "generation-output-release!",
    "generation-rng-release!",
    "generator-close",
):
    assert f"(g3c4-public-{operation}" in manual_facade, operation
assert "(g3c4-public-generation-tensor-release! value)" in manual_facade
assert "(provide generation-c4-input-create" in manual_generation
assert "(provide diagnostic-c4-model-create-seeded" in manual_transport
assert "generation-c4-input-create" not in facade
assert "(define (generator-generate! arg0 arg1)" in facade
assert "(define (generation-output-ids arg0)" in facade
assert "C4 registries" in contract and "public E1 error boundary" in contract

assert "G3_C4_P2_G2_PUBLIC_FACADE_CONTRACT.md" in roadmap
print("public P2/G2 facade source composition: PASS (source only)")
