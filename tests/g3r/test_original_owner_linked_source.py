"""Causal source gate for the private linked original-owner runtime fixture."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"
fixture = (ROOT / "tests/g3r/original_owner_linked_runtime.esk").read_text()
runner = (ROOT / "scripts/test-g3r-original-owner-linked.sh").read_text()


def balanced(source):
    depth = 0
    quoted = escaped = comment = False
    for char in source:
        if comment:
            if char == "\n":
                comment = False
        elif quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == ";":
            comment = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            assert depth >= 0
    assert depth == 0 and not quoted


balanced(fixture)


def closure(path):
    yield path.name
    for name in re.findall(r'^\(load "([^"]+)"\)', path.read_text(), re.M):
        child = NATIVE / name
        if child.is_file():
            yield from closure(child)


names = list(closure(NATIVE / "g3r_original_owner_c2_claim_root.esk"))
for name in (
    "g3r_original_owner_preclaim_root.esk",
    "g3r_original_owner_coordinator_extension.esk",
    "g3r_held_o2_adopter_extension.esk",
    "g3r_held_o2_c2_claim_extension.esk",
    "c2_training_state_extension.esk",
    "o2_wave2_extension.esk",
):
    assert names.count(name) == 1, (name, names.count(name))
assert fixture.count('(load "g3r_original_owner_c2_claim_root.esk")') == 1
assert fixture.count("(load ") == 1


def section(text, begin, end):
    return text.split(begin, 1)[1].split(end, 1)[0]


def ordered(text, *items):
    positions = [text.index(item) for item in items]
    assert positions == sorted(positions), items


def check(source):
    setup = section(source, "(define (make-tuple directory)",
                    "(define (controls tuple)")
    ordered(setup, "(module-eval! model)", "(g3t-generator-create",
            "(g3m-generate-p1-g1!", "(g3t-output-release! output)",
            "(module-train! model)", "(o2-public-optimizer-create",
            "(tr3-lease-create-internal")
    transfer = section(source, "(define (prepare-transfer tuple)",
                       "(define (finish-tuple tuple)")
    ordered(transfer, "(g3r-original-owner-begin!",
            "(g3r-held-o2-adopt!", "(module-state-dict",
            "(vector c2-training-state-compose-tag")
    normal = section(source, "(define (normal tuple transfer)",
                     "(define (partial-claim tuple transfer raised-false?)")
    ordered(normal, "(c2-training-state-compose-internal request)",
            "(g3r-held-o2-adopter-c2-claim!", "(release-claimed transfer shell)")
    cleanup = section(source, "(define (release-claimed transfer shell)",
                      "(define (normal tuple transfer)")
    ordered(cleanup, "(g3r-original-owner-end!", "(c2-training-state-release-internal!")
    partial = section(source, "(define (partial-claim tuple transfer raised-false?)",
                      "(define (post-adoption-cleanup tuple transfer raised-false?)")
    for item in ("(saved state token)", "(raise #f)",
                 "(c2-training-state-fail 'corrupt-data 'trainer-state",
                 "(eq? c2-training-state-states before)",
                 "(eq? (vector-ref request 5) #f)",
                 "(o2-require-live-state", "(state-dict-paths p1)"):
        assert item in partial, item
    ordered(partial, "(state-dict-release! p1)",
            "(g3r-held-o2-adopter-release! adopter)",
            "(g3r-original-owner-end! (vector-ref transfer 0))")
    post = section(source, "(define (post-adoption-cleanup tuple transfer raised-false?)",
                   "(define (post-compose-forgery transfer)")
    assert "(if shell (release-claimed transfer shell))" in post
    assert "(raise #f)" in post
    assert "(c2-training-state-fail 'internal 'trainer-state" in post
    forge = section(source, "(define (post-compose-forgery transfer)",
                    "(define args (command-line))")
    ordered(forge, "(c2-training-state-compose-internal request)",
            "(vector-set! request 1", "(g3r-held-o2-adopter-c2-claim!")


check(fixture)
for cut in (
    "(g3m-generate-p1-g1!", "(module-train! model)",
    "(g3r-original-owner-begin!", "(g3r-held-o2-adopt!",
    "(g3r-held-o2-adopter-c2-claim!", "(raise #f)",
    "(o2-require-live-state", "(g3r-held-o2-adopter-release! adopter)",
    "(if shell (release-claimed transfer shell))",
):
    try:
        check(fixture.replace(cut, "(REMOVED", 1))
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError(f"linked owner mutation escaped: {cut}")

for item in ("verify_toolchain", "resolve_provenance_compilers",
             "-DET_G3R_SAME_OWNER_SAVE_PRIVATE",
             "-DET_G3R_CANDIDATE_RETIRE_PRIVATE",
             "-DET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE",
             "-DET_C2_O2_RECONSTRUCT_BRIDGE",
             "compile src/eshkol_transformer/g3t_transport.c",
             "compile native/g3r_trainer_g3_save_bridge.c",
             "original_owner_linked_runtime.esk", "post-cleanup-typed",
             "post-cleanup-false", "post-forgery", '"$status" == 134',
             "G3R LINKED OWNER POST-COMPOSE FORGERY"):
    assert item in runner, item
for included_source in ("native/f32_tensor.c", "native/i64_tensor.c",
                        "src/eshkol_transformer/m3t_transport.c",
                        "src/eshkol_transformer/m3t_f32_integration.c",
                        "src/eshkol_transformer/m3_model.c"):
    assert f"compile {included_source}" not in runner, included_source
assert "docker" not in runner
print("G3R linked owner source PASS (one registry, 9 causal mutations)")
