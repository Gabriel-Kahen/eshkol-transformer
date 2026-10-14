#!/usr/bin/env python3
"""Check completeness and byte integrity of one noninstalled E3 N1 import receipt.

This checker does not decide whether a call ABI or provider is safe. That
requires independent review of the pinned raw evidence before source admission.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HEX = re.compile(r"^[0-9a-f]{64}$")
OBJECTS = {
    "private.o", "package-native-5.o", "package-native-6.o",
    "combined.o", "libprivate.so", "private.ll",
}
EVIDENCE = {
    "argv", "environment", "relocations", "disassembly", "optimized_ir",
    "source_mapping", "declarations", "provider_symbols", "resolved_link_map",
    "loader_search", "needed_closure", "depfile_inventory", "full_header_trace",
    "normal_imports", "noninstrumentation_imports", "sanitizer_imports",
    "depfile_anchor", "reviewed_claims",
}
DEPFILES = {
    "private.d", "bridge.d", "package-bridge.d",
    *(f"package-native-{index}.d" for index in range(11)),
    "providers/n2/n2_primitives_provider.d",
    "providers/n3k/n3k_primitives_provider.d",
    "providers/a2/a2_attention_provider.d",
}
PINS = {
    "source_head", "source_tree", "image_id", "compiler_commit",
    "builder_sha256", "target_triple",
}
F_PINS = {
    "source_head": "94b8fecd2d187ba990128c44884dc35a40d130e9",
    "source_tree": "d28cffc1a9a8de3c78b2a1cbd9ea27d1cb5fdd1e",
    "image_id": "sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6",
    "compiler_commit": "fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9",
    "builder_sha256": "9d39986addc9ab0cda28671b330116896e50927597b4f840919b0b697f83c698",
    "target_triple": "x86_64-pc-linux-gnu",
}
F_OBJECTS = {
    "combined.o": "3b65c611e4179851dc64285ba755c706718bd20eba1a1c7bcad521bdb5d21bc8",
    "libprivate.so": "b61fa27fc0034d9193b8dfb7f979f6ebfa4b2d59eea09c718e020adf4c31f972",
    "package-native-5.o": "6997483f01975defd030e394e43cc2edbff6dc852262c700d144fe1c0f573b35",
    "package-native-6.o": "bd85cace584cb0e1b8333c658c3508d1d83439ab710560805a375b9cd2c22458",
    "private.ll": "8538b6251cbad38c6ce84b8a2c39acdf68fa0153f223de401a05e91e0176dec9",
    "private.o": "ad1fd0ae2363a44fcb35092f5aa625d9831ee24c4fc82d53a5572abe86c27007",
}
F_CALLS_SHA256 = "e75cbc6316b90626709bea0016eebfd71cbcb2eff50f0d23989ea24e0241c3ef"
F_DEPFILE_ANCHOR_SHA256 = "0b47f72d89d770975d5583a26964e8a36424f59ebeb674d4d9acef5abae6ef16"
F_DEPFILE_ROOT = (
    "/home/gabe/e3-numerical-transcript-94b8fec-fe9-f31-20261002-prepared-f/"
    "scratch/eshkol-transformer-e1b.AWHfVe"
)
F_DEPFILE_PATHS = {name: f"{F_DEPFILE_ROOT}/{name}" for name in DEPFILES}
# No image-specific ABI/provider/declaration/loader claims have been accepted.
# A later independently reviewed source change must supply the exact anchor.
F_REVIEWED_CLAIMS_SHA256 = None
F_IMPORTS = {
    "normal_imports": "e9a317029e27da0a9e9a07d416727b33c837a8be04d6c912c97f47273f94bae2",
    "noninstrumentation_imports": "a84ba743537bc2d3c1cd50ad74b8195464338f80f15b591276b8dd58e112eb28",
    "sanitizer_imports": "957836a08fbe44b5bf4d9117ac7f6201f9153c6dc3527eb9a1cd62b3c3db65c7",
}


class ReceiptError(Exception):
    pass


def pairs_unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def need(test, message):
    if not test:
        raise ReceiptError(message)


def nonempty(value, name):
    need(isinstance(value, str) and bool(value.strip()), f"missing {name}")
    need(value.strip().lower() not in {"synthetic", "unknown", "unverified", "todo",
                                       "placeholder", "n/a"},
         f"placeholder {name}")
    return value


def digest(value, name):
    need(isinstance(value, str) and HEX.fullmatch(value), f"invalid {name} SHA-256")
    return value


def pinned_file(record, name):
    need(isinstance(record, dict) and set(record) == {"path", "sha256"},
         f"invalid {name} file record")
    raw = nonempty(record["path"], f"{name} path")
    path = Path(raw)
    need(path.is_absolute() and path.exists() and path.is_file(),
         f"missing {name} bytes: {raw}")
    need(os.path.realpath(raw) == raw, f"unresolved {name} symlink: {raw}")
    expected = digest(record["sha256"], name)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    actual = hasher.hexdigest()
    need(actual == expected, f"changed {name} bytes: {raw}")
    return actual


def call_roster_hash(calls):
    fields = ("object", "symbol", "section", "offset", "relocation", "addend")
    rows = [{name: call[name] for name in fields} for call in calls]
    rows.sort(key=lambda row: (row["object"], row["section"], row["offset"]))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reviewed_claims(data):
    """Projection checked against a separately reviewed immutable raw receipt."""
    return {
        "calls": [
            {name: call[name] for name in
             ("object", "section", "offset", "symbol", "producer", "abi",
              "declaration", "provider")}
            for call in data["calls"]
        ],
        "provider": data["provider"],
        "provider_files": data["provider_files"],
        "headers": data["headers"],
        "raw_evidence": {
            name: {field: data["evidence"][name][field] for field in ("path", "sha256")}
            for name in
            ("argv", "environment", "relocations", "disassembly", "optimized_ir",
             "source_mapping", "declarations", "provider_symbols",
             "resolved_link_map", "loader_search", "needed_closure",
             "full_header_trace")
        },
    }


def check_receipt(data, *, expected_pins=F_PINS, expected_objects=F_OBJECTS,
                  expected_calls=F_CALLS_SHA256, expected_imports=F_IMPORTS,
                  expected_depfile_anchor=F_DEPFILE_ANCHOR_SHA256,
                  expected_depfile_paths=F_DEPFILE_PATHS,
                  expected_claims=F_REVIEWED_CLAIMS_SHA256):
    need(isinstance(data, dict), "receipt must be an object")
    need(data.get("schema") == "e3-n1-import-receipt-v1", "wrong receipt schema")
    need(data.get("status") == "candidate-for-independent-review",
         "receipt is not a complete review candidate")
    pins = data.get("pins")
    need(isinstance(pins, dict) and set(pins) == PINS, "missing or extra candidate pins")
    for name in PINS:
        value = nonempty(pins[name], name)
        if name not in {"image_id", "target_triple"}:
            need(bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value)),
                 f"invalid {name}")
    need(re.fullmatch(r"sha256:[0-9a-f]{64}", pins["image_id"]), "invalid image ID")
    need(pins == expected_pins, "candidate source/image/compiler/flags pins changed")

    objects = data.get("objects")
    need(isinstance(objects, dict) and set(objects) == OBJECTS,
         "missing or extra object records")
    for name in sorted(OBJECTS):
        need(isinstance(objects[name], dict), f"invalid {name} object record")
        need(objects[name].get("sha256") == expected_objects[name],
             f"candidate {name} hash changed")
        pinned_file(objects[name], name)

    evidence = data.get("evidence")
    need(isinstance(evidence, dict) and set(evidence) == EVIDENCE,
         "missing or extra evidence records")
    for name in sorted(EVIDENCE):
        if name in expected_imports:
            need(isinstance(evidence[name], dict) and
                 evidence[name].get("sha256") == expected_imports[name],
                 f"candidate {name} hash changed")
        if name == "depfile_anchor":
            need(isinstance(evidence[name], dict) and
                 evidence[name].get("sha256") == expected_depfile_anchor,
                 "candidate depfile anchor changed")
        pinned_file(evidence[name], name)
    depfile_anchor = json.loads(Path(evidence["depfile_anchor"]["path"]).read_text(),
                                object_pairs_hook=pairs_unique)
    anchored_depfiles = depfile_anchor.get("dependencies", {}).get("depfile_sha256")
    need(isinstance(anchored_depfiles, dict) and set(anchored_depfiles) == DEPFILES,
         "incomplete anchored F depfile inventory")
    inventories = {}
    for name in expected_imports:
        lines = Path(evidence[name]["path"]).read_text().splitlines()
        need(bool(lines) and lines == sorted(set(lines)), f"invalid {name} inventory")
        inventories[name] = set(lines)
    normal = inventories["normal_imports"]
    sanitized = inventories["noninstrumentation_imports"]
    need(sanitized - normal == {"bcmp", "puts"},
         "unexpected sanitized noninstrumentation import")
    need(normal - sanitized == {"eshkol_ad_seed_flag", "eshkol_runtime_fatal",
                                "eshkol_taylor_lift_ad_node", "memcmp"},
         "unexpected absent normal import")
    need(not (inventories["sanitizer_imports"] & sanitized),
         "sanitizer and noninstrumentation inventories overlap")

    calls = data.get("calls")
    need(isinstance(calls, list) and len(calls) == 39, "expected 39 call rows")
    observed = set()
    counts = {"bcmp": 0, "puts": 0}
    for index, call in enumerate(calls):
        label = f"call {index}"
        need(isinstance(call, dict), f"invalid {label}")
        need(set(call) == {
            "object", "symbol", "section", "offset", "relocation", "addend",
            "producer", "abi", "declaration", "provider",
        }, f"incomplete {label}")
        obj = nonempty(call["object"], f"{label} object")
        symbol = nonempty(call["symbol"], f"{label} symbol")
        need((symbol == "puts" and obj == "private.o") or
             (symbol == "bcmp" and obj in {"package-native-5.o", "package-native-6.o"}),
             f"unexpected {label} object/symbol")
        need(call["relocation"] == "R_X86_64_PLT32" and call["addend"] == -4,
             f"changed {label} relocation")
        section = nonempty(call["section"], f"{label} section")
        offset = nonempty(call["offset"], f"{label} offset")
        need(bool(re.fullmatch(r"0x[0-9a-f]+", offset)), f"invalid {label} offset")
        key = (obj, section, offset)
        need(key not in observed, f"duplicate {label} relocation")
        observed.add(key)
        counts[symbol] += 1
        producer = call["producer"]
        need(isinstance(producer, dict) and set(producer) ==
             {"source", "final_mapping", "transformation"},
             f"missing {label} producer")
        for field in producer:
            nonempty(producer[field], f"{label} producer {field}")
        abi = call["abi"]
        need(isinstance(abi, dict) and set(abi) ==
             {"arguments", "return", "calling_convention", "widths",
              "pointer_semantics", "address_spaces", "visibility", "version"},
             f"missing {label} ABI")
        need(isinstance(abi["arguments"], list) and bool(abi["arguments"]),
             f"missing {label} ABI arguments")
        for field, value in abi.items():
            if field != "arguments":
                nonempty(value, f"{label} ABI {field}")
        for argument in abi["arguments"]:
            nonempty(argument, f"{label} ABI argument")
        nonempty(call["declaration"], f"{label} declaration")
        nonempty(call["provider"], f"{label} provider")
    need(counts == {"bcmp": 14, "puts": 25}, "wrong call counts")
    need(sum(c["object"] == "package-native-5.o" for c in calls) == 2,
         "wrong package-native-5 call count")
    need(call_roster_hash(calls) == expected_calls, "candidate call roster changed")

    for name in ("depfiles", "headers", "provider_files"):
        records = data.get(name)
        need(isinstance(records, list) and bool(records), f"missing {name}")
        names = set()
        for i, record in enumerate(records):
            need(isinstance(record, dict) and set(record) == {"name", "file"},
                 f"invalid {name} record")
            entry = nonempty(record["name"], f"{name} name")
            need(entry not in names, f"duplicate {name} name")
            names.add(entry)
            if name == "depfiles":
                need(isinstance(record["file"], dict) and
                     record["file"].get("path") == expected_depfile_paths.get(entry),
                     f"candidate {entry} path differs from sealed F depfile")
            pinned_file(record["file"], f"{name} {entry}")
        if name == "depfiles":
            need(names == DEPFILES, "missing or extra F depfiles")
            for record in records:
                need(record["file"]["sha256"] == anchored_depfiles[record["name"]],
                     f"candidate {record['name']} differs from sealed F depfile")
        if name == "headers":
            trace = Path(evidence["full_header_trace"]["path"]).read_text().splitlines()
            need(trace == sorted(set(trace)) and bool(trace),
                 "header trace must be sorted unique")
            need(all(Path(path).is_absolute() for path in trace),
                 "header trace contains nonabsolute path")
            need(names == set(trace), "missing or extra traced header bytes")
            need(all(record["name"] == record["file"]["path"] for record in records),
                 "header path and pinned file differ")
    provider = data.get("provider")
    need(isinstance(provider, dict) and set(provider) ==
         {"bcmp", "puts", "symlink_chain", "loader_path", "transitive_closure"},
         "missing provider closure")
    for symbol in ("bcmp", "puts"):
        need(isinstance(provider[symbol], dict) and set(provider[symbol]) ==
             {"file", "symbol_type", "symbol_version", "soname",
              "binding", "visibility"},
             f"incomplete {symbol} provider")
        for field in provider[symbol]:
            nonempty(provider[symbol][field], f"{symbol} provider {field}")
        need(provider[symbol]["file"] in
             {item["name"] for item in data["provider_files"]},
             f"unlisted {symbol} provider file")
    for call in calls:
        selected = provider[call["symbol"]]
        need(call["provider"] == selected["file"],
             "call provider differs from provider-symbol record")
        need(call["abi"]["version"] == selected["symbol_version"],
             "call ABI version differs from provider-symbol record")
    for field in ("symlink_chain", "loader_path", "transitive_closure"):
        need(isinstance(provider[field], list) and bool(provider[field]),
             f"missing provider {field}")
    provider_names = {record["name"] for record in data["provider_files"]}
    need(set(provider["transitive_closure"]) == provider_names,
         "provider transitive closure differs from pinned files")
    resolved_files = {record["file"]["path"] for record in data["provider_files"]}
    for directory in provider["loader_path"]:
        nonempty(directory, "loader search directory")
        need(Path(directory).is_absolute(), "nonabsolute loader search directory")
    for link in provider["symlink_chain"]:
        need(isinstance(link, dict) and set(link) == {"path", "target"},
             "invalid provider symlink chain")
        path = nonempty(link["path"], "provider symlink path")
        target = nonempty(link["target"], "provider symlink target")
        need(os.path.islink(path) and os.readlink(path) == target,
             f"changed provider symlink: {path}")
        need(os.path.realpath(path) in resolved_files,
             f"unlisted provider symlink target: {path}")
    need(expected_claims is not None,
         "no independently accepted ABI/provider/declaration/loader claims anchor")
    need(evidence["reviewed_claims"]["sha256"] == expected_claims,
         "reviewed claims anchor changed")
    claims = json.loads(Path(evidence["reviewed_claims"]["path"]).read_text(),
                        object_pairs_hook=pairs_unique)
    need(claims == reviewed_claims(data),
         "ABI/provider/declaration/loader claims differ from reviewed anchor")
    return "complete byte-integrity candidate; independent ABI/provider review required"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.receipt.read_text(), object_pairs_hook=pairs_unique)
        print(check_receipt(data))
    except (OSError, ValueError, ReceiptError) as error:
        print(f"E3 N1 receipt HOLD: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
