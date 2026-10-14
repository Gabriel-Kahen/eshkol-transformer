#!/usr/bin/env python3
"""Fast structural checks for the accepted G3-C4 Step 4a context/cache leaf."""

import ast
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DISABLED_STEP4 = frozenset({
    "ET_G3C4_ACTIVE_CALL_PRIVATE",
    "ET_G3C4_GENERATOR_PRIVATE",
    "ET_G3C4_P2_G2_TERMINAL_PRIVATE",
})
SAFE_STEP4_INCLUDES = frozenset({
    '"g3c4_context_internal.h"',
    '"eshkol_transformer/a2_kv_cache.h"',
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered(text, fragments, label):
    position = -1
    for fragment in fragments:
        found = text.find(fragment, position + 1)
        require(found >= 0, f"{label} missing or misordered: {fragment}")
        position = found


def _condition_value(expression, disabled):
    """Return True, False, or None for an unknown preprocessor condition."""
    tokens = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = re.match(r"[A-Za-z_]\w*|0[xX][0-9A-Fa-f]+|[0-9]+|&&|\|\||!|[()]",
                         expression[position:])
        require(match is not None,
                f"unsupported preprocessor condition: {expression}")
        tokens.append(match.group())
        position += len(match.group())

    translated = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "defined":
            index += 1
            parenthesized = index < len(tokens) and tokens[index] == "("
            if parenthesized:
                index += 1
            require(index < len(tokens) and
                    re.fullmatch(r"[A-Za-z_]\w*", tokens[index]) is not None,
                    f"unsupported preprocessor condition: {expression}")
            name = tokens[index]
            index += 1
            if parenthesized:
                require(index < len(tokens) and tokens[index] == ")",
                        f"unsupported preprocessor condition: {expression}")
                index += 1
            translated.append(f"defined({name})")
            continue
        if re.fullmatch(r"[A-Za-z_]\w*", token):
            require(token in disabled,
                    f"unsupported preprocessor macro expansion: {token}")
        translated.append({"&&": "and", "||": "or", "!": "not"}.get(token, token))
        index += 1
    try:
        parsed = ast.parse(" ".join(translated), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"unsupported preprocessor condition: {expression}") from error

    def negate(value):
        return None if value is None else not value

    def combine(values, is_and):
        if (False if is_and else True) in values:
            return not is_and
        if all(value is (True if is_and else False) for value in values):
            return is_and
        return None

    def evaluate(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            require(node.func.id == "defined" and len(node.args) == 1 and
                    isinstance(node.args[0], ast.Name) and not node.keywords,
                    f"unsupported preprocessor condition: {expression}")
            return False if node.args[0].id in disabled else None
        if isinstance(node, ast.Name):
            return False if node.id in disabled else None
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return bool(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return negate(evaluate(node.operand))
        if isinstance(node, ast.BoolOp):
            require(isinstance(node.op, (ast.And, ast.Or)),
                    f"unsupported preprocessor condition: {expression}")
            return combine([evaluate(value) for value in node.values],
                           isinstance(node.op, ast.And))
        raise ValueError(f"unsupported preprocessor condition: {expression}")

    return evaluate(parsed.body)


def _mask_comments(text):
    """Replace C comments with spaces before recognizing directives."""
    kept = []
    state = "code"
    index = 0
    while index < len(text):
        pair = text[index:index + 2]
        char = text[index]
        if state == "block":
            if pair == "*/":
                kept.extend("  ")
                state = "code"
                index += 2
            else:
                kept.append("\n" if char == "\n" else " ")
                index += 1
        elif state == "line":
            kept.append("\n" if char == "\n" else " ")
            if char == "\n":
                state = "code"
            index += 1
        elif state in ('"', "'"):
            kept.append(char)
            if char == "\\" and index + 1 < len(text):
                kept.append(text[index + 1])
                index += 2
            else:
                if char == state:
                    state = "code"
                index += 1
        elif pair in ("/*", "//"):
            kept.extend("  ")
            state = "block" if pair == "/*" else "line"
            index += 2
        else:
            kept.append(char)
            if char in ('"', "'"):
                state = char
            index += 1
    require(state not in ("block", '"', "'"), "unterminated C comment or quote")
    return "".join(kept)


def _logical_lines(text):
    """Apply C line splicing before comment masking and directive parsing."""
    require("??/" not in text, "unsupported C trigraph in Step 4a projection")
    spliced = text.replace(chr(92) + "\r\n", "").replace(chr(92) + "\n", "")
    yield from _mask_comments(spliced).splitlines(keepends=True)


def require_no_feature_redefinitions(source):
    """Reject source/header changes that invalidate the disabled-feature premise."""
    def inspect(text, label):
        for line in _logical_lines(text):
            match = re.match(r"\s*#\s*(?:define|undef)\s+([A-Za-z_]\w*)\b", line)
            require(not match or match.group(1) not in DISABLED_STEP4,
                    f"Step 4a feature redefined by {label}")

    inspect(source, "source")
    for directory in ("src", "include"):
        for path in (ROOT / directory).rglob("*.h"):
            inspect(path.read_text(), str(path))


def step4_projection(block):
    """Keep every source arm that can run with later features disabled."""
    disabled = DISABLED_STEP4
    kept = []
    frames = []  # [prior-arm truth, current-arm reachable]

    def visible():
        return all(frame[1] for frame in frames)

    for line in _logical_lines(block):
        match = re.match(r"\s*#\s*([A-Za-z_]\w*)\b(.*)",
                         line, re.DOTALL)
        if not match:
            if visible():
                kept.append(line)
            continue
        directive, rest = match.groups()
        if directive in ("define", "undef") and visible():
            name = re.match(r"\s*([A-Za-z_]\w*)\b", rest)
            require(name is not None, f"unsupported #{directive}: {rest}")
            require(name.group(1) not in disabled,
                    f"Step 4a feature redefined: {name.group(1)}")
        elif directive == "include" and visible():
            require(rest.strip() in SAFE_STEP4_INCLUDES,
                    f"unsupported Step 4a include: {rest.strip()}")
        elif directive == "pragma" and visible():
            raise ValueError("unsupported Step 4a pragma")
        if directive not in ("if", "ifdef", "ifndef", "elif", "else", "endif"):
            if visible():
                kept.append(line)
            continue
        if directive in ("if", "ifdef", "ifndef", "elif"):
            if directive == "if" or directive == "elif":
                value = _condition_value(rest, disabled) if visible() or directive == "elif" else None
            else:
                name = rest.strip()
                require(re.fullmatch(r"[A-Za-z_]\w*", name) is not None,
                        f"unsupported #{directive}: {rest}")
                value = False if name in disabled else None
                if directive == "ifndef":
                    value = None if value is None else not value
            if directive == "elif":
                require(frames, "unexpected #elif")
                prior = frames[-1][0]
                frames[-1] = [
                    _or(prior, value),
                    prior is not True and value is not False,
                ]
            else:
                frames.append([value, value is not False])
        elif directive == "else":
            require(frames, "unexpected #else")
            prior = frames[-1][0]
            frames[-1] = [True, prior is not True]
        else:
            require(frames, "unexpected #endif")
            frames.pop()
    require(not frames, "unterminated preprocessor conditional")
    return "".join(kept)


def _or(left, right):
    if left is True or right is True:
        return True
    if left is False and right is False:
        return False
    return None


def require_no_deferred_surface(block):
    step4_block = step4_projection(block)
    for forbidden in [
        "et_g3c4_model_pins", "generator_seed", "generator_rng",
        "policy", "sampler", "forward", "owner->active =",
    ]:
        require(forbidden not in step4_block,
                f"Step 4a reaches deferred surface: {forbidden}")


def check():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-g3c4-model-authority.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_context_cache.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_CONTEXT_CACHE_STEP4A_CONTRACT.md").read_text()

    require("root accepted" in contract.lower(),
            "Step 4a contract does not record root acceptance")
    require(source.count("#ifdef ET_G3C4_CONTEXT_PRIVATE") == 1,
            "context implementation feature block changed")
    block = source[source.index("#ifdef ET_G3C4_CONTEXT_PRIVATE"):]
    require_no_feature_redefinitions(source)
    require(source.index("#ifdef ET_G3C4_CONTEXT_PRIVATE") >
            source.index("et_g3c4_private_model_owner_abort_v1"),
            "context block no longer follows accepted owner implementation")
    for symbol in [
        "et_g3c4_private_context_create_v1",
        "et_g3c4_private_context_close_v1",
    ]:
        require(header.count(symbol) == 1, f"header boundary changed: {symbol}")
        require(block.count(symbol) == 1, f"source boundary changed: {symbol}")

    for field in [
        "registry_next", "magic", "kind", "state", "busy", "owner", "cache",
    ]:
        require(field in block, f"context record omits {field}")
    require_no_deferred_surface(block)

    ordered(block, [
        "owner = et_g3c4_admit_owner(candidate, 0)",
        "owner->state != ET_G3C4_OWNER_SEALED",
        "owner->active != NULL",
        "context = et_g3c4_context_allocate()",
        "et_a2_kv_cache_create_v1(",
        "1u, 1u, 2u, 4u, 2u, &cache, &error",
        "context->owner = owner",
        "context->cache = cache",
        "context->registry_next = et_g3c4_context_registry",
        "et_g3c4_context_registry = context",
    ], "context construction")
    ordered(block, [
        "context = et_g3c4_admit_context(candidate)",
        "ET_G3C4_CONTEXT_STATE(context) == ET_G3C4_CONTEXT_DEAD",
        "owner = et_g3c4_admit_owner(context->owner, 0)",
        "owner->state != ET_G3C4_OWNER_SEALED",
        "owner->active != NULL",
        "et_a2_kv_cache_destroy_v1(&context->cache, &error)",
        "context->owner = NULL",
        "ET_G3C4_CONTEXT_STATE(context) = ET_G3C4_CONTEXT_DEAD",
    ], "context close")
    require("free(context);\n    et_g3c4_error_restore_internal(first);" in block,
            "constructor failure does not preserve first error")

    for phrase in [
        "construction_failures", "cache_geometry_and_multiplicity",
        "transaction_busy", "view_busy",
        "borrow_busy", "wrong_lifecycle", "registry_1024",
        "registry_8192", "owner_unchanged",
    ]:
        require(phrase in test, f"focused test omits {phrase}")

    expected = [
        "src/eshkol_transformer/g3c4_context_internal.h",
        "src/eshkol_transformer/g3c4_model_owner_internal.h",
        "src/eshkol_transformer/g3c4_model_owner.c",
        "src/eshkol_transformer/m3t_f32_scoped.h",
        "src/eshkol_transformer/m3t_f32_integration.c",
        "native/f32_parameter_internal.h",
        "native/f32_tensor.c",
        "include/eshkol_transformer/f32_tensor.h",
        "include/eshkol_transformer/i64_tensor.h",
        "include/eshkol_transformer/kernel_abi.h",
        "include/eshkol_transformer/n3k_primitives_abi.h",
        "include/eshkol_transformer/a2_kv_cache.h",
        "native/a2_kv_cache.c",
        "native/kernel_abi.c",
        "native/n3k_primitives_provider.c",
        "tests/m3t/kernel_fail_allocator.c",
        "tests/g3c4/test_model_owner.c",
        "tests/g3c4/test_context_cache.c",
        "scripts/check-g3c4-context-cache.py",
        "scripts/test-g3c4-native-owner.sh",
        "scripts/test-g3c4-context-cache.sh",
        "docs/g3/G3_C4_CONTEXT_CACHE_STEP4A_CONTRACT.md",
        "docs/g3/G3_C4_CONTEXT_CACHE_STEP4A.md",
    ]
    actual = (ROOT / "native/g3c4_context_source_closure.txt").read_text().splitlines()
    require(actual == expected, "Step 4a source closure changed")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 context/cache static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 context/cache static contract: PASS")
