#!/usr/bin/env python3
"""Bounded data-only C1 model-name mapping and sharded safetensors interchange."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import os
from pathlib import Path
import re
import secrets
import stat
import sys

import gguf_interop as gg
import safetensors_interop as safe
from safetensors_interop import InteropError, Limits, State, Tensor, require

M3 = "eshkol-m3-v1"
DECODER = "eshkol-decoder-v1"
PROFILES = (M3, DECODER)
SCHEMES = ("huggingface", "gguf")

MAPPING_VERSION = "eshkol.io5.mapping_version"
PROFILE = "eshkol.io5.profile"
SCHEME = "eshkol.io5.scheme"
MAPPING_FIELDS = {MAPPING_VERSION, PROFILE, SCHEME}
SHARDED_VERSION = "eshkol.io5.sharded_version"
SHARD_SHA256 = "eshkol.io5.shard_sha256"
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARDS = 4096
SHARD_BASENAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,242}\.safetensors")
SHA256 = re.compile(r"[0-9a-f]{64}")
LAYER = re.compile(r"0|[1-9][0-9]{0,3}")

# Each row is canonical stem, Hugging Face stem, GGUF stem, and required matrix/vector shape for M3.
# None marks the decoder-only gate. Parameters remain individual C1 path segments.
TOP = (
    ("token_embedding", "model.embed_tokens", "token_embd", (256, 4)),
    ("position_embedding", "model.embed_positions", "position_embd", (2, 4)),
    ("norm_final", "model.norm", "output_norm", (4,)),
    ("head", "lm_head", "output", (256, 4)),
)
BLOCK = (
    ("norm1", "input_layernorm", "attn_norm", (4,)),
    ("attention.query", "self_attn.q_proj", "attn_q", (4, 4)),
    ("attention.key", "self_attn.k_proj", "attn_k", (4, 4)),
    ("attention.value", "self_attn.v_proj", "attn_v", (4, 4)),
    ("attention.output", "self_attn.o_proj", "attn_output", (4, 4)),
    ("norm2", "post_attention_layernorm", "ffn_norm", (4,)),
    ("ffn.up", "mlp.up_proj", "ffn_up", (8, 4)),
    ("ffn.down", "mlp.down_proj", "ffn_down", (4, 8)),
    ("ffn.gate", "mlp.gate_proj", "ffn_gate", None),
)
HEAD = ("head", "weight")
EMBEDDING = ("token_embedding", "weight")
TIE_PATHS = (HEAD, EMBEDDING)


def check_selection(profile: str | None, scheme: str | None) -> None:
    require(profile in PROFILES and scheme in SCHEMES, "explicit supported profile and scheme required")


def tables(rows: tuple, scheme: str, *, block: bool) -> tuple[dict, dict]:
    names, shapes = {}, {}
    for canonical, hf, gguf, shape in rows:
        stem = tuple(canonical.split("."))
        external = hf if scheme == "huggingface" else gguf
        names[stem + ("weight",)] = external + ".weight"
        shapes[stem + ("weight",)] = shape
        if block or canonical == "norm_final":
            names[stem + ("bias",)] = external + ".bias"
            if shape is not None and len(shape) == 1:
                shapes[stem + ("bias",)] = shape
    return names, shapes


def layer_index(segment: str) -> int:
    require(LAYER.fullmatch(segment) is not None, "invalid layer index spelling")
    index = int(segment)
    require(index < Limits().entries, "layer index limit")
    return index


def mapped_names(state: State, profile: str, scheme: str) -> tuple[str, ...]:
    check_selection(profile, scheme)
    top, m3_top = tables(TOP, scheme, block=False)
    block, m3_block = tables(BLOCK, scheme, block=True)
    prefix = "model.layers" if scheme == "huggingface" else "blk"
    names, roots, layers = [], set(), defaultdict(set)
    by_path = {tensor.path: tensor for tensor in state.tensors}
    for tensor in state.tensors:
        path = tensor.path
        require(tensor.kind == "parameter", "mapped tensor must be a parameter")
        if path[0] == "blocks" and len(path) >= 3:
            index = layer_index(path[1])
            suffix = path[2:]
            require(suffix in block, "unknown canonical block path")
            require(profile != M3 or index == 0, "M3 requires exactly layer 0")
            names.append(f"{prefix}.{index}.{block[suffix]}")
            layers[index].add(suffix)
            shape = m3_block.get(suffix)
            if suffix[-1] == "bias":
                require(path[:-1] + ("weight",) in by_path, "projection/norm bias requires weight")
        else:
            require(path in top, "unknown canonical path")
            names.append(top[path])
            roots.add(path)
            shape = m3_top.get(path)
        if profile == M3:
            require(shape is not None and tensor.shape == shape, "M3 tensor path/shape mismatch")
        else:
            rank = 1 if path[-1] == "bias" or path[-2] in ("norm1", "norm2", "norm_final") else 2
            require(len(tensor.shape) == rank and all(tensor.shape), "decoder tensor rank/extent mismatch")
    require(len(set(names)) == len(names), "duplicate mapped tensor name")
    require(set(layers) == set(range(len(layers))), "non-contiguous layer indices")
    if profile == M3:
        require(roots == m3_top.keys() and layers.get(0) == {p for p, s in m3_block.items() if s is not None},
                "missing M3 tensor")
    else:
        required = {path for path in top if path[-1] == "weight" and path[0] != "position_embedding"}
        require(required <= roots, "missing decoder root tensor")
        required = {path for path in block if path[-1] == "weight" and path[-2] != "gate"}
        require(all(required <= paths for paths in layers.values()), "missing decoder block tensor")
    groups = tuple(tuple(state.tensors[i].path for i in group) for group in state.aliases)
    require(groups in ((), (TIE_PATHS,)), "only the head/embedding tie is supported")
    require(profile != M3 or groups, "M3 requires a head/embedding tie")
    return tuple(names)


def plain_state(named: dict[str, Tensor], profile: str, scheme: str,
                provider: str, tie: bool) -> State:
    top, _ = tables(TOP, scheme, block=False)
    block, _ = tables(BLOCK, scheme, block=True)
    top_paths = {name: path for path, name in top.items()}
    block_paths = {name: path for path, name in block.items()}
    prefix = "model.layers." if scheme == "huggingface" else "blk."
    tensors = {}
    for name, tensor in named.items():
        if name.startswith(prefix):
            segment, separator, suffix = name[len(prefix):].partition(".")
            require(separator and suffix in block_paths, "unmapped external block name")
            layer_index(segment)
            path = ("blocks", segment) + block_paths[suffix]
        else:
            require(name in top_paths, "unmapped external tensor name")
            path = top_paths[name]
        require(path not in tensors, "duplicate mapped tensor path")
        tensors[path] = replace(tensor, path=path)
    paths = sorted(tensors)
    aliases = ()
    if tie or profile == M3:
        require(all(path in tensors for path in TIE_PATHS), "missing head/embedding tensor for tie")
        head, embedding = (tensors[path] for path in TIE_PATHS)
        require(head.shape == embedding.shape and head.data == embedding.data,
                "head/embedding tie requires equal shapes and exact bits")
        aliases = (tuple(paths.index(path) for path in TIE_PATHS),)
    return State(provider, tuple(tensors[path] for path in paths), aliases)


def synthetic_names(state: State) -> tuple[str, ...]:
    return tuple(f"t{index:06d}" for index in range(len(state.tensors)))


def mapping_fields(profile: str, scheme: str) -> dict[str, str]:
    return {MAPPING_VERSION: "1.0", PROFILE: profile, SCHEME: scheme}


def manifest_fields(raw: bytes, state: State, names: tuple[str, ...], digest_key: str) -> dict[str, str]:
    return {safe.MANIFEST_KEY: safe.make_manifest(state, names),
            digest_key: hashlib.sha256(raw).hexdigest()}


def selection_from_metadata(metadata: dict, profile: str | None,
                            scheme: str | None) -> tuple[str, str]:
    selected_profile, selected_scheme = metadata[PROFILE], metadata[SCHEME]
    require(metadata[MAPPING_VERSION] == "1.0", "unsupported IO5 mapping version")
    check_selection(selected_profile, selected_scheme)
    require(profile is None or profile == selected_profile, "IO5 profile mismatch")
    require(scheme is None or scheme == selected_scheme, "IO5 scheme mismatch")
    return selected_profile, selected_scheme


def io5_fields(metadata: dict) -> set[str]:
    return {key for key in metadata if key.startswith("eshkol.io5.")}


def restore_manifest(named: dict[str, Tensor], metadata: dict, digest_key: str,
                     profile: str | None, scheme: str | None, limits: Limits) -> bytes:
    expected = metadata[digest_key]
    require(type(expected) is str and SHA256.fullmatch(expected), "invalid IO5 C1 checksum")
    state, names = safe.state_from_manifest(metadata[safe.MANIFEST_KEY], named, limits)
    result = safe.encode_c1(state, limits)
    actual_names = synthetic_names(state) if profile is None else mapped_names(state, profile, scheme)
    require(names == actual_names, "IO5 manifest tensor names do not match selected mapping")
    require(hashlib.sha256(result).hexdigest() == expected, "IO5 C1 checksum mismatch")
    return result


def encode_mapped(raw: bytes, format_name: str, profile: str, scheme: str,
                  limits: Limits = Limits()) -> bytes:
    require(format_name in ("safetensors", "gguf"), "unsupported wire format")
    state = safe.decode_c1(raw, limits)
    names = mapped_names(state, profile, scheme)
    is_gguf = format_name == "gguf"
    fields = (mapping_fields(profile, scheme)
              | manifest_fields(raw, state, names, gg.DIGEST_KEY if is_gguf else safe.DIGEST_KEY))
    encode = gg.encode_gguf_tensors if is_gguf else safe.encode_safetensors_tensors
    return encode(dict(zip(names, state.tensors)), fields, limits)


def decode_mapped(raw: bytes, format_name: str, limits: Limits = Limits(), *,
                  profile: str | None = None, scheme: str | None = None,
                  external: bool = False, provider: str | None = None, tie: bool = False) -> bytes:
    require(format_name in ("safetensors", "gguf"), "unsupported wire format")
    is_gguf = format_name == "gguf"
    decode = gg.decode_gguf_tensors if is_gguf else safe.decode_safetensors_tensors
    named, metadata = decode(raw, limits)
    reserved = io5_fields(metadata)
    if external:
        require(not reserved, "external import cannot override IO5 metadata")
        check_selection(profile, scheme)
        safe.text_bytes(provider, 127, "explicit external provider")
        state = plain_state(named, profile, scheme, provider, tie)
        mapped_names(state, profile, scheme)
        return safe.encode_c1(state, limits)
    require(provider is None and not tie, "provider/tie options require --external")
    digest_key = gg.DIGEST_KEY if is_gguf else safe.DIGEST_KEY
    require(reserved == MAPPING_FIELDS | {safe.MANIFEST_KEY, digest_key}, "IO5 mapping metadata fields")
    if is_gguf:
        require(metadata["general.architecture"] == "eshkol", "IO5 GGUF architecture profile")
    profile, scheme = selection_from_metadata(metadata, profile, scheme)
    return restore_manifest(named, metadata, digest_key, profile, scheme, limits)


def shard_groups(named: dict[str, Tensor], limits: Limits, max_shard_bytes: int,
                 max_shards: int, allow_oversize: bool) -> list[dict[str, Tensor]]:
    groups = []
    group, header, offset = {}, 2, 0
    for name, tensor in named.items():
        size = len(tensor.data)
        entry = len(safe.json_bytes({name: safe.safetensors_descriptor(tensor, offset)})) - 2
        padded = safe.padded_safetensors_header_size(header + entry + bool(group))
        if group and (padded > limits.metadata or 8 + padded + offset + size > max_shard_bytes):
            groups.append(group)
            require(len(groups) < max_shards, "shard count limit")
            group, header, offset = {}, 2, 0
            entry = len(safe.json_bytes({name: safe.safetensors_descriptor(tensor, 0)})) - 2
            padded = safe.padded_safetensors_header_size(header + entry)
        require(padded <= limits.metadata, "shard header limit")
        file_size = 8 + padded + offset + size
        require(file_size <= limits.file, "shard file size limit")
        require(file_size <= max_shard_bytes or (allow_oversize and not group),
                "single tensor exceeds --max-shard-bytes including shard header")
        header += entry + bool(group)
        group[name] = tensor
        offset += size
    if group:
        groups.append(group)
    require(len(groups) <= max_shards, "shard count limit")
    return groups


def max_index_bytes(limits: Limits) -> int:
    return min(limits.metadata, limits.file)


def check_shard_count(max_shards: int) -> None:
    require(type(max_shards) is int and 0 <= max_shards <= MAX_SHARDS, "invalid shard count policy")


def build_sharded(raw: bytes, limits: Limits, *,
                  profile: str | None, scheme: str | None, max_shard_bytes: int,
                  max_shards: int, allow_oversize: bool) -> dict[str, bytes]:
    check_shard_count(max_shards)
    require(type(max_shard_bytes) is int and 0 < max_shard_bytes <= limits.file,
            "invalid shard byte policy")
    require((profile is None) == (scheme is None), "sharded mapping requires both profile and scheme")
    state = safe.decode_c1(raw, limits)
    names = synthetic_names(state) if profile is None else mapped_names(state, profile, scheme)
    groups = shard_groups(dict(zip(names, state.tensors)), limits, max_shard_bytes,
                          max_shards, allow_oversize)
    shards, hashes, weight_map = {}, {}, {}
    total_bytes = 0
    for number, group in enumerate(groups, 1):
        name = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        content = safe.encode_safetensors_tensors(group, {}, limits)
        total_bytes += len(content)
        require(total_bytes <= limits.file + limits.metadata, "aggregate shard file size limit")
        shards[name] = content
        hashes[name] = hashlib.sha256(content).hexdigest()
        weight_map.update(dict.fromkeys(group, name))
    metadata = {"total_size": sum(len(tensor.data) for tensor in state.tensors),
                SHARDED_VERSION: "1.0", SHARD_SHA256: hashes}
    metadata.update(manifest_fields(raw, state, names, safe.DIGEST_KEY))
    if profile is not None:
        metadata.update(mapping_fields(profile, scheme))
    index = safe.json_bytes({"metadata": metadata, "weight_map": weight_map})
    require(len(index) <= max_index_bytes(limits), "sharded index byte limit")
    shards[INDEX_NAME] = index
    return shards


@contextmanager
def index_directory(path: Path, *, writing: bool):
    require(path.name == INDEX_NAME, f"expected an explicit {INDEX_NAME} path")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX if writing else fcntl.LOCK_SH)
        yield directory
    finally:
        os.close(directory)


def stage(directory: int, raw: bytes) -> str:
    name = ".io5-" + secrets.token_hex(16)
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        os.unlink(name, dir_fd=directory)
        raise
    return name


def publish_sharded(path: Path, files: dict[str, bytes], overwrite: bool) -> None:
    require(bool(files) and next(reversed(files)) == INDEX_NAME, "sharded index must be published last")
    for name in files:
        if name != INDEX_NAME:
            require(type(name) is str and ".." not in name and SHARD_BASENAME.fullmatch(name),
                    "invalid shard basename")
    with index_directory(path, writing=True) as directory:
        # The caller constructs the index last. It is the publication point for a bundle.
        existing = set()
        for name in files:
            try:
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not overwrite:
                raise FileExistsError(name)
            require(stat.S_ISREG(info.st_mode), "shard output target must be a regular nonsymlink file")
            existing.add(name)

        staged, backups, published = {}, {}, []
        preserve_backups = False
        try:
            for name, raw in files.items():
                staged[name] = stage(directory, raw)
            for name in sorted(existing):
                backup = ".io5-backup-" + secrets.token_hex(16)
                os.link(name, backup, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                backups[name] = backup
            if backups:
                os.fsync(directory)
            try:
                for name, temporary in staged.items():
                    if name == INDEX_NAME:
                        os.fsync(directory)
                    if overwrite:
                        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                    else:
                        os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                    published.append(name)
            except BaseException:
                try:
                    for name in reversed(published):
                        if name in backups:
                            os.replace(backups[name], name, src_dir_fd=directory, dst_dir_fd=directory)
                        else:
                            os.unlink(name, dir_fd=directory)
                    os.fsync(directory)
                except OSError:
                    preserve_backups = True
                    raise
                raise
        finally:
            cleanup = staged.values() if preserve_backups else (*staged.values(), *backups.values())
            for temporary in cleanup:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
        os.fsync(directory)


def read_sharded(path: Path, limits: Limits = Limits(), *,
                 profile: str | None = None, scheme: str | None = None,
                 max_shards: int = MAX_SHARDS) -> bytes:
    check_shard_count(max_shards)
    with index_directory(path, writing=False) as directory:
        index_limit = max_index_bytes(limits)
        raw = safe.read_file(INDEX_NAME, replace(limits, file=index_limit), directory=directory)
        index = safe.parse_json(raw, index_limit)
        require(type(index) is dict and index.keys() == {"metadata", "weight_map"}, "sharded index fields")
        metadata, weight_map = index["metadata"], index["weight_map"]
        fields = {"total_size", SHARDED_VERSION, SHARD_SHA256, safe.MANIFEST_KEY, safe.DIGEST_KEY}
        require(type(metadata) is dict and metadata.keys() in (fields, fields | MAPPING_FIELDS),
                "sharded index metadata fields")
        require(metadata[SHARDED_VERSION] == "1.0", "unsupported IO5 sharded version")
        if MAPPING_VERSION in metadata:
            profile, scheme = selection_from_metadata(metadata, profile, scheme)
        else:
            require(profile is None and scheme is None, "index has no mapping metadata")
        total_size = safe.uint(metadata["total_size"])
        require(total_size <= limits.file, "total_size limit")
        hashes = metadata[SHARD_SHA256]
        require(type(weight_map) is dict and len(weight_map) <= limits.entries, "weight_map tensor count limit")
        require(type(hashes) is dict and len(hashes) <= max_shards, "shard count limit")
        for name, checksum in hashes.items():
            require(".." not in name and SHARD_BASENAME.fullmatch(name), "invalid shard basename")
            require(type(checksum) is str and SHA256.fullmatch(checksum), "invalid shard SHA-256")
        coverage = defaultdict(set)
        for name, shard in weight_map.items():
            require(type(shard) is str and shard in hashes, "weight_map shard missing from checksum map")
            coverage[shard].add(name)
        require(coverage.keys() == hashes.keys(), "shard coverage mismatch")
        tensors, payload_size, remaining = {}, 0, limits.file + limits.metadata
        for name, expected in coverage.items():
            shard_limits = replace(limits, file=min(limits.file, remaining))
            content = safe.read_file(name, shard_limits, directory=directory)
            remaining -= len(content)
            require(hashlib.sha256(content).hexdigest() == hashes[name], "shard SHA-256 mismatch")
            named, shard_metadata = safe.decode_safetensors_tensors(content, shard_limits)
            require(not io5_fields(shard_metadata), "unexpected IO5 metadata in shard")
            require(named.keys() == expected, "shard weight_map coverage mismatch")
            payload_size += sum(len(tensor.data) for tensor in named.values())
            require(payload_size <= total_size, "sharded total_size mismatch")
            tensors.update(named)
        require(payload_size == total_size, "sharded total_size mismatch")
    return restore_manifest(tensors, metadata, safe.DIGEST_KEY, profile, scheme, limits)


def add_selection(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--profile", choices=PROFILES, required=required)
    parser.add_argument("--scheme", choices=SCHEMES, required=required)


def add_limits(parser: argparse.ArgumentParser) -> None:
    for name, value in vars(Limits()).items():
        parser.add_argument("--max-" + name, type=int, default=value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for format_name in ("safetensors", "gguf", "sharded-safetensors"):
        for exporting in (True, False):
            label = format_name if format_name == "sharded-safetensors" else "mapped-" + format_name
            direction = f"c1-to-{label}" if exporting else f"{label}-to-c1"
            operation = commands.add_parser(direction)
            operation.add_argument("source", type=Path)
            operation.add_argument("destination", type=Path)
            operation.add_argument("--overwrite", action="store_true")
            add_selection(operation)
            add_limits(operation)
            operation.set_defaults(exporting=exporting, format_name=format_name)
            if format_name == "sharded-safetensors":
                operation.add_argument("--max-shards", type=int, default=MAX_SHARDS)
                if exporting:
                    operation.add_argument("--max-shard-bytes", type=int)
                    operation.add_argument("--allow-oversize-single-tensor", action="store_true")
            elif not exporting:
                operation.add_argument("--external", action="store_true")
                operation.add_argument("--provider")
                operation.add_argument("--tie-head-to-embedding", action="store_true")
    inspect = commands.add_parser("inspect-mapping")
    inspect.add_argument("source", type=Path)
    add_selection(inspect, required=True)
    add_limits(inspect)

    args = parser.parse_args()
    try:
        limits = Limits(**{name: getattr(args, "max_" + name) for name in vars(Limits())})
        if args.command == "inspect-mapping":
            state = safe.decode_c1(safe.read_file(args.source, limits), limits)
            names = mapped_names(state, args.profile, args.scheme)
            manifest = safe.parse_json(safe.make_manifest(state, names).encode("ascii"), limits.metadata)
            sys.stdout.buffer.write(safe.json_bytes(mapping_fields(args.profile, args.scheme)
                                                   | {safe.MANIFEST_KEY: manifest}) + b"\n")
            return 0
        if args.format_name == "sharded-safetensors" and not args.exporting:
            result = read_sharded(args.source, limits, profile=args.profile, scheme=args.scheme,
                                  max_shards=args.max_shards)
            safe.publish(args.destination, result, args.overwrite)
            return 0
        raw = safe.read_file(args.source, limits)
        if args.format_name == "sharded-safetensors":
            files = build_sharded(
                raw, limits, profile=args.profile, scheme=args.scheme, max_shards=args.max_shards,
                max_shard_bytes=limits.file if args.max_shard_bytes is None else args.max_shard_bytes,
                allow_oversize=args.allow_oversize_single_tensor)
            publish_sharded(args.destination, files, args.overwrite)
            return 0
        if args.exporting:
            scheme = args.scheme or ("gguf" if args.format_name == "gguf" else "huggingface")
            result = encode_mapped(raw, args.format_name, args.profile, scheme, limits)
        else:
            result = decode_mapped(raw, args.format_name, limits, profile=args.profile,
                                   scheme=args.scheme, external=args.external, provider=args.provider,
                                   tie=args.tie_head_to_embedding)
        safe.publish(args.destination, result, args.overwrite)
        return 0
    except (InteropError, OSError) as error:
        print(f"IO5 model mapping error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
