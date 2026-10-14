#!/usr/bin/env python3
"""Freeze the installed G1/C2 boundary while private G2 lacks a public contract."""

from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "native/g3g_manual_package_root.esk": "c96cd342849066a46ec1fbd75a0ef60fcfc4e44e140e4eac35f994baf4cd283f",
    "native/g3g_manual_package_source_closure.txt": "a524bd25875c4a42be996c78d31288dc71949b884f7ab54af95278e3431b1857",
    "native/g3g_manual_package_public_exports.txt": "4186a699293e70ebf062f7e27252f5dfa184b69d0520877d0d4748cf0b1b3824",
    "native/g3g_manual_package_public_strings.txt": "714e467b12338ab9291acceedb1e04608d770bf3c42bbb7ed6c163998c97010b",
    "native/g3g_manual_package_bridge.c": "6e47f9d1e992163f6cef5ac4af6f039c3dd79e9b7223eeab839692326fef4ce5",
    "native/g3g_manual_package_private_renames.txt": "9ddeeb5cf35242155b1c911b35ea48a8d0ac247eb5df3549e2ef0ef1d38544a5",
    "native/g3g_manual_facades/transformer/generation.esk": "02f63d95bee7282376b363240e7ca59dba024b0cfe0094cfb29017df3e2b6529",
    "native/g3g_manual_public_extension.esk": "88522753e7830d759781b5434472d2388df9e8673319ff3e9dcdf9b553f29a61",
    "native/g3c4_public_generation_dispatch_extension.esk": "eb2e8e06da0da0cda882475a49a071e87bbf5d94bb8a639fdcf4e80654d793a4",
    "native/g3c4_public_output_adapter_extension.esk": "3d55f8737b6fcfc280a9c4094362d69eaa1d005002cf1b604f2464d6f40d2139",
    "native/g3c4_p2g1_shell_publication_extension.esk": "28f7e8a28e095b9b323032daa4775c9fee77aaacf9e01d632540c4591ff2afb1",
}


def current_sources():
    return {name: (ROOT / name).read_text() for name in PINS}


def form(source, name):
    start = source.find(f"(define ({name} ")
    if start < 0:
        return ""
    # These installed top-level definitions contain no nested definitions.
    end = source.find("\n(define ", start + 1)
    return source[start:end if end >= 0 else None]


def ordered(source, *fragments):
    offsets = [source.find(fragment) for fragment in fragments]
    return -1 not in offsets and offsets == sorted(offsets)


def issues(sources):
    problems = set()
    for name, expected in PINS.items():
        value = sources.get(name)
        if value is None or sha256(value.encode()).hexdigest() != expected:
            problems.add(f"review-pin:{name}")
    if any(name not in PINS for name in sources):
        problems.add("review-pin:unexpected-source")

    root = sources["native/g3g_manual_package_root.esk"]
    closure = sources["native/g3g_manual_package_source_closure.txt"]
    if ("g3c4_p2g2" in root or "g3c4_p2g2" in closure
            or root.count('(load "g3c4_public_generation_dispatch_extension.esk")') != 1
            or root.count('(load "g3c4_public_output_adapter_extension.esk")') != 1):
        problems.add("premature-terminal-package-load")

    dispatch = sources["native/g3c4_public_generation_dispatch_extension.esk"]
    public = sources["native/g3g_manual_public_extension.esk"]
    profile = form(dispatch, "g3c4-public-profile-config")
    create = form(dispatch, "g3c4-public-generator-create")
    generate = form(dispatch, "g3c4-public-generator-generate")
    c2_create = form(public, "g3g-public-generator-create")
    c2_generate = form(public, "g3g-public-generator-generate")
    if ("(not (eq? value 'diagnostic-c4-p2g1))" not in profile
            or "(= (vector-ref policy 5) 1)" not in create
            or "(eq? (vector-ref policy 6) #f)" not in create
            or "diagnostic-c4-p2g2" in dispatch
            or "g3c4-p2g2" in dispatch
            or not ordered(generate, "(g3c4-native-p2g1-preflight",
                           "(vector-set! g3t-p2g1-pending-gate 0 #t)",
                           "(g3c4-p2g1-generate-internal generator input)")):
        problems.add("p2g1-profile-budget-eos")
    if (not ordered(c2_create, "(if (g3c4-public-model-member model)",
                    "(g3c4-public-generator-create model tokenizer config)",
                    "(g3t-seed-config config 'generator-create)",
                    "(g3t-generator-create model tokenizer config)")
            or "(= (vector-ref (vector-ref normalized 0) 5) 0)" not in c2_create
            or "(= (vector-ref (vector-ref normalized 0) 5) 1)" not in c2_create
            or not ordered(c2_generate, "(g3c4-public-transport-member generator)",
                           "(if (g3c4-public-model-member generator)",
                           "(if c4-member", "(g3c4-public-generator-generate generator input)",
                           "(g3t-generator-find generator 'generator-generate!)")):
        problems.add("c4-c2-fallback-order")

    for name in ("ids", "lengths", "text", "rng", "cache-lengths"):
        public_name = "g3g-public-generation-output-" + name
        c4_name = "g3c4-public-generation-output-" + name
        c2_name = "g3t-generation-output-" + name
        body = form(public, public_name)
        if not ordered(body, "(g3c4-public-transport-member output)",
                       "(g3c4-public-model-member output)",
                       f"({c4_name} output)", f"({c2_name} output)"):
            problems.add("c4-c2-fallback-order")

    adapter = sources["native/g3c4_public_output_adapter_extension.esk"]
    copy = form(adapter, "g3c4-public-output-copy")
    valid = form(adapter, "g3c4-public-copy-valid")
    publish = form(adapter, "g3c4-public-result-publish")
    output_release = form(adapter, "g3c4-public-generation-output-release!")
    result_release = form(adapter, "g3c4-public-result-release!")
    if ("(g3c4-p2g1-output-copy-internal shell)" not in copy
            or "(g3c4-p2g1-output-entry-live shell operation)" not in copy
            or "(bytevector-length (vector-ref copy 0)) 1" not in valid
            or "(= (vector-ref copy 2) 1) (= (vector-ref copy 3) 3)" not in valid
            or "(g3c4-native-result-tensor-create" not in publish
            or "g3c4-p2g2" in adapter):
        problems.add("scalar-adapter-not-g2")
    if ("(g3c4-p2g1-output-release-internal! output)" not in output_release
            or not ordered(result_release, "(g3c4-native-rng-release",
                           "(g3c4-native-result-tensor-release",
                           "(g3c4-entry-dead! live)")
            or not ordered(form(public, "g3g-public-generation-output-release"),
                           "(g3c4-public-transport-member output)",
                           "(g3c4-public-generation-output-release! output)",
                           "(g3t-output-release! output)")):
        problems.add("exact-owner-native-first-release")

    p2g1_release = form(
        sources["native/g3c4_p2g1_shell_publication_extension.esk"],
        "g3c4-p2g1-output-release-internal!")
    if not ordered(p2g1_release, "(g3c4-p2g1-output-entry-live shell operation)",
                   "(g3c4-native-output-release (vector-ref live 3))",
                   "(g3c4-entry-dead! live)"):
        problems.add("exact-owner-native-first-release")

    exposed = "\n".join(sources[name] for name in (
        "native/g3g_manual_package_public_exports.txt",
        "native/g3g_manual_package_public_strings.txt",
        "native/g3g_manual_package_bridge.c",
        "native/g3g_manual_facades/transformer/generation.esk",
    ))
    if ("p2g2" in exposed.lower() or "terminal-ids" in exposed.lower()
            or "terminal_ids_clone" in exposed.lower()):
        problems.add("private-clone-public-exposure")
    return problems


if __name__ == "__main__":
    found = issues(current_sources())
    if found:
        raise SystemExit("G3-C4 public G2 frontier HOLD: " + ", ".join(sorted(found)))
    print("G3-C4 public G2 frontier source guard PASS: current C2/P2G1 only")
