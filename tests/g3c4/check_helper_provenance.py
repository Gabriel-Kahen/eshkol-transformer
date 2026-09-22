"""Mechanically check copied arithmetic helpers and the two accepted adaptations."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
current_text = (root / "native/g3c4_primitives_provider.c").read_text(encoding="utf-8")


def function(source: str, name: str) -> str:
    match = re.search(r"\b" + re.escape(name) + r"\s*\(", source)
    assert match is not None, name
    start = match.start()
    position = source.index("{", match.start()) + 1
    depth = 1
    while depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    return re.sub(r"\s+", "", source[start:position])


def predecessor(name: str) -> str:
    return (root / "native" / name).read_text(encoding="utf-8")


exact = [
    ("n3k_primitives_provider.c", [
        "embedding_forward_run", "linear_forward_run", "index_linear_x", "index_linear_y"
    ]),
    ("n2_primitives_provider.c", [
        "index3", "layer_norm_statistics", "layer_norm_complete_statistics",
        "layer_norm_forward_run", "gelu_one",
    ]),
    ("a2_attention_provider.c", [
        "attention_scale", "q_index", "kv_index", "mask_index", "attention_score",
    ]),
    ("g3n_primitives_provider.c", ["attention_forward_run"]),
]
count = 0
for source_name, names in exact:
    source_text = predecessor(source_name)
    for name in names:
        assert function(current_text, name) == function(source_text, name), (source_name, name)
        count += 1

a2_admitted = function(predecessor("a2_attention_provider.c"), "admitted").replace(
    "admitted(", "attention_admitted("
)
assert function(current_text, "attention_admitted") == a2_admitted
count += 1

# Removing only the named shifted-finiteness checks must recover G3-N exactly.
row = function(current_text, "attention_row").replace(
    "constfloatshifted=score-row_max;if(!isfinite(shifted))return-1;weight=expf(shifted);",
    "weight=expf(score-row_max);",
)
probability = function(current_text, "attention_probability").replace(
    "constfloatshifted=score-maximum;if(!isfinite(shifted))return0;*result=expf(shifted)/denominator;",
    "*result=expf(score-maximum)/denominator;",
)
g3n = predecessor("g3n_primitives_provider.c")
assert row == function(g3n, "attention_row")
assert probability == function(g3n, "attention_probability")
assert function(current_text, "attention_row").count("!isfinite(shifted)") == 1
assert function(current_text, "attention_probability").count("!isfinite(shifted)") == 1

print(
    f"G3-C4-N helper provenance: {count} exact copied helpers; "
    "two mechanically isolated shifted-finiteness adaptations"
)
