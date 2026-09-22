"""Check reviewed copied helpers against hash-frozen predecessor sources."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
current = (root / "native/g3n_primitives_provider.c").read_text()


def function(source, name):
    start = re.search(r"\b" + name + r"\(", source).start()
    position = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    return re.sub(r"\s+", "", source[start:position]).replace("attention_admitted(", "admitted(")


for source, names in [
    ("n3k_primitives_provider.c", ["embedding_forward_run", "linear_forward_run", "index_linear_x", "index_linear_y"]),
    ("n2_primitives_provider.c", ["index3", "layer_norm_statistics", "layer_norm_complete_statistics", "layer_norm_forward_run"]),
    ("a2_attention_provider.c", ["attention_scale", "q_index", "kv_index", "mask_index", "attention_score", "attention_row", "attention_probability"]),
]:
    predecessor = (root / "native" / source).read_text()
    for name in names:
        assert function(current, name) == function(predecessor, name), (source, name)
assert function(current, "attention_admitted").replace("attention_admitted(", "admitted(") == function(
    (root / "native/a2_attention_provider.c").read_text(), "admitted")
print("G3-N helper provenance: sixteen unchanged arithmetic helpers (A2 admitted renamed)")
