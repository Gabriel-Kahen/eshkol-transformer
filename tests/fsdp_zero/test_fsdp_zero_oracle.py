#!/usr/bin/env python3
"""Independent binary64 AdamW + global L2 clipping oracle for FSDP/ZeRO-1/2/3.

Uses only the Python standard library (zero PyTorch/NumPy dependency) to verify:
  1. Analytical vs central finite-difference MLP gradients.
  2. Exact <= 1e-6 weight and loss parity for ALL (stage in {1,2,3}, W in {1,2,4})
     against an independent binary64 unsharded AdamW reference.
  3. Exact bitwise identity across ZeRO Stages 1, 2, and 3 for each W.
  4. Exact 1/W optimizer state reduction (Stages 1/2/3) and exact 1/W total
     persistent parameter + gradient + optimizer state reduction (Stage 3 FSDP).
  5. Optional verification of compiled Eshkol AOT binary weight dumps (W=1/2/4).
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

IN_DIM = 8
HIDDEN_DIM = 17
NUM_CLASSES = 5
NUM_PARAMS = IN_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * NUM_CLASSES + NUM_CLASSES
GLOBAL_BATCH = 8
TRAIN_STEPS = 12
MODEL_SEED = 20261124

LR = 0.05
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5

MASK64 = (1 << 64) - 1


def f32(v: float) -> float:
    x = struct.unpack("<f", struct.pack("<f", v))[0]
    return 0.0 if x == 0.0 else x


def splitmix64_next(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def splitmix64_sym_f32(state: int, scale: float) -> tuple[int, float]:
    state, r = splitmix64_next(state)
    u = r >> 40
    v = ((float(u) / 8388608.0) - 1.0) * scale
    return state, f32(v)


def init_mlp_params(seed: int) -> list[float]:
    rng = seed ^ 0xA5A5A5A55A5A5A5A
    s1 = 0.5 / math.sqrt(float(IN_DIM))
    s2 = 0.5 / math.sqrt(float(HIDDEN_DIM))
    params: list[float] = []
    for _ in range(IN_DIM * HIDDEN_DIM):
        rng, v = splitmix64_sym_f32(rng, s1)
        params.append(v)
    for _ in range(HIDDEN_DIM):
        rng, v = splitmix64_sym_f32(rng, 0.05)
        params.append(v)
    for _ in range(HIDDEN_DIM * NUM_CLASSES):
        rng, v = splitmix64_sym_f32(rng, s2)
        params.append(v)
    for _ in range(NUM_CLASSES):
        rng, v = splitmix64_sym_f32(rng, 0.05)
        params.append(v)
    assert len(params) == NUM_PARAMS
    return params


def make_global_batch() -> tuple[list[list[float]], list[int]]:
    xs: list[list[float]] = []
    ys: list[int] = []
    for s in range(GLOBAL_BATCH):
        cls = s % NUM_CLASSES
        ys.append(cls)
        rng = 0x517CC1B727220A95 ^ (((s + 1) * 0xBF58476D1CE4E5B9) & MASK64)
        row: list[float] = []
        for j in range(IN_DIM):
            proto = 0.9 * math.cos(0.75 * float(cls + 1) * float(j + 1))
            rng, jit = splitmix64_sym_f32(rng, 0.15)
            row.append(f32(proto + jit))
        xs.append(row)
    return xs, ys


def mlp_forward_backward_f64(
    params: list[float], xs: list[list[float]], ys: list[int], need_grads: bool = True
) -> tuple[float, list[float]]:
    w1_off = 0
    b1_off = IN_DIM * HIDDEN_DIM
    w2_off = b1_off + HIDDEN_DIM
    b2_off = w2_off + HIDDEN_DIM * NUM_CLASSES

    B = len(xs)
    inv_b = 1.0 / float(B)
    grads = [0.0] * NUM_PARAMS
    total_loss = 0.0

    for n in range(B):
        xn = xs[n]
        y = ys[n]
        h = [0.0] * HIDDEN_DIM
        for j in range(HIDDEN_DIM):
            acc = float(params[b1_off + j])
            for i in range(IN_DIM):
                acc += float(xn[i]) * float(params[w1_off + i * HIDDEN_DIM + j])
            h[j] = math.tanh(acc)

        logits = [0.0] * NUM_CLASSES
        max_l = -math.inf
        for c in range(NUM_CLASSES):
            acc = float(params[b2_off + c])
            for j in range(HIDDEN_DIM):
                acc += h[j] * float(params[w2_off + j * NUM_CLASSES + c])
            logits[c] = acc
            if acc > max_l:
                max_l = acc

        exps = [math.exp(l - max_l) for l in logits]
        sum_exp = sum(exps)
        log_z = max_l + math.log(sum_exp)
        total_loss += log_z - logits[y]

        if need_grads:
            probs = [e / sum_exp for e in exps]
            dh = [0.0] * HIDDEN_DIM
            for c in range(NUM_CLASSES):
                dl = (probs[c] - (1.0 if c == y else 0.0)) * inv_b
                grads[b2_off + c] += dl
                for j in range(HIDDEN_DIM):
                    grads[w2_off + j * NUM_CLASSES + c] += h[j] * dl
                    dh[j] += dl * float(params[w2_off + j * NUM_CLASSES + c])
            for j in range(HIDDEN_DIM):
                dh_pre = dh[j] * (1.0 - h[j] * h[j])
                grads[b1_off + j] += dh_pre
                for i in range(IN_DIM):
                    grads[w1_off + i * HIDDEN_DIM + j] += float(xn[i]) * dh_pre

    return total_loss * inv_b, grads


def run_python_reference() -> tuple[list[float], list[float], list[float]]:
    """Pure binary64 unsharded AdamW + global L2 gradient clipping reference."""
    params = [float(p) for p in init_mlp_params(MODEL_SEED)]
    xs, ys = make_global_batch()

    # Verify Python analytical gradients via central finite differences first.
    _, anal_g = mlp_forward_backward_f64(params, xs, ys, need_grads=True)
    fd_eps = 1e-6
    max_fd = 0.0
    for idx in (0, 17, 136, 152, 153, 200, 238, 242):
        orig = params[idx]
        params[idx] = orig + fd_eps
        lp, _ = mlp_forward_backward_f64(params, xs, ys, need_grads=False)
        params[idx] = orig - fd_eps
        lm, _ = mlp_forward_backward_f64(params, xs, ys, need_grads=False)
        params[idx] = orig
        max_fd = max(max_fd, abs(anal_g[idx] - (lp - lm) / (2.0 * fd_eps)))
    assert max_fd < 1e-8, f"Python oracle FD check failed: {max_fd}"

    m = [0.0] * NUM_PARAMS
    v = [0.0] * NUM_PARAMS
    losses: list[float] = []
    norms: list[float] = []

    for step in range(1, TRAIN_STEPS + 1):
        loss, g = mlp_forward_backward_f64(params, xs, ys, need_grads=True)
        losses.append(loss)
        gn = math.sqrt(sum(val * val for val in g))
        norms.append(gn)
        clip = (MAX_GRAD_NORM / gn) if (MAX_GRAD_NORM > 0.0 and gn > MAX_GRAD_NORM) else 1.0
        bc1 = 1.0 - (BETA1 ** step)
        bc2 = 1.0 - (BETA2 ** step)
        for i in range(NUM_PARAMS):
            gi = g[i] * clip
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * gi
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * gi * gi
            mh = m[i] / bc1
            vh = v[i] / bc2
            params[i] = params[i] * (1.0 - LR * WEIGHT_DECAY) - LR * (
                mh / (math.sqrt(vh) + EPS)
            )
    return params, losses, norms


def load_dump(path: Path) -> dict:
    raw = path.read_bytes()
    hdr = struct.unpack_from("<8Q", raw, 0)
    off = 64
    losses = list(struct.unpack_from(f"<{TRAIN_STEPS}d", raw, off))
    off += TRAIN_STEPS * 8
    norms = list(struct.unpack_from(f"<{TRAIN_STEPS}d", raw, off))
    off += TRAIN_STEPS * 8
    params = list(struct.unpack_from(f"<{NUM_PARAMS}f", raw, off))
    off += NUM_PARAMS * 4
    assert off == len(raw), f"unexpected dump size {len(raw)} vs {off}"
    return {
        "stage": hdr[0],
        "world": hdr[1],
        "num_params": hdr[2],
        "steps": hdr[3],
        "param_bytes": hdr[4],
        "grad_bytes": hdr[5],
        "opt_bytes": hdr[6],
        "total_bytes": hdr[7],
        "losses": losses,
        "norms": norms,
        "params": params,
    }


def check_aot_weights(ref_params: list[float], paths: list[Path]) -> None:
    assert len(paths) == 3, "expected W=1, W=2, W=4 AOT weight dump paths"
    w_arrays = []
    for p in paths:
        raw = p.read_bytes()
        assert len(raw) == NUM_PARAMS * 4, f"bad weight dump size {len(raw)}"
        w_arrays.append(list(struct.unpack(f"<{NUM_PARAMS}f", raw)))

    w1, w2, w4 = w_arrays
    for label, arr in (("AOT W=1", w1), ("AOT W=2", w2), ("AOT W=4", w4)):
        d_ref = max(abs(a - b) for a, b in zip(arr, ref_params))
        d_w1 = max(abs(a - b) for a, b in zip(arr, w1))
        print(
            f"FSDP ORACLE {label}: max_vs_py64={d_ref:.3e} max_vs_aot_w1={d_w1:.3e} PASS"
        )
        assert d_ref <= 1e-6, f"{label} diverged from Python binary64 oracle: {d_ref}"
        assert d_w1 <= 1e-6, f"{label} diverged from AOT W=1 baseline: {d_w1}"


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: test_fsdp_zero_oracle.py <native_bin> | --check-aot <w1.bin> <w2.bin> <w4.bin>",
            file=sys.stderr,
        )
        return 2

    ref_params, ref_losses, ref_norms = run_python_reference()

    if sys.argv[1] == "--check-aot":
        if len(sys.argv) != 5:
            print("expected 3 weight files after --check-aot", file=sys.stderr)
            return 2
        check_aot_weights(ref_params, [Path(x) for x in sys.argv[2:5]])
        return 0

    native_bin = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="fsdp_oracle_") as td:
        subprocess.run([str(native_bin), "--oracle-dump", td], check=True)
        td_path = Path(td)

        for W in (1, 2, 4):
            dumps = {}
            padded = ((NUM_PARAMS + W - 1) // W) * W
            shard = padded // W
            for stage in (1, 2, 3):
                d = load_dump(td_path / f"stage{stage}_w{W}.bin")
                dumps[stage] = d
                assert d["num_params"] == NUM_PARAMS
                assert d["steps"] == TRAIN_STEPS
                assert d["opt_bytes"] == 2 * shard * 4
                if stage == 1:
                    assert d["param_bytes"] == padded * 4
                    assert d["grad_bytes"] == padded * 4
                elif stage == 2:
                    assert d["param_bytes"] == padded * 4
                    assert d["grad_bytes"] == shard * 4
                else:
                    assert d["param_bytes"] == shard * 4
                    assert d["grad_bytes"] == shard * 4
                    assert d["total_bytes"] == 4 * shard * 4

                max_p = max(abs(a - b) for a, b in zip(d["params"], ref_params))
                max_l = max(abs(a - b) for a, b in zip(d["losses"], ref_losses))
                max_gn = max(abs(a - b) for a, b in zip(d["norms"], ref_norms))
                assert max_p <= 1e-6, f"stage={stage} W={W} param diff {max_p}"
                assert max_l <= 1e-6, f"stage={stage} W={W} loss diff {max_l}"
                assert max_gn <= 1e-5, f"stage={stage} W={W} norm diff {max_gn}"

            # Stages 1, 2, 3 must match bitwise for the same W.
            for stage in (2, 3):
                assert dumps[stage]["params"] == dumps[1]["params"]
                assert dumps[stage]["losses"] == dumps[1]["losses"]

            d3 = dumps[3]
            max_p3 = max(abs(a - b) for a, b in zip(d3["params"], ref_params))
            max_l3 = max(abs(a - b) for a, b in zip(d3["losses"], ref_losses))
            print(
                f"FSDP ORACLE W={W} (Stages 1/2/3 bitwise equal): "
                f"max_param_vs_py64={max_p3:.3e} max_loss_vs_py64={max_l3:.3e} "
                f"stage3_total={d3['total_bytes']}B PASS"
            )

    print("FSDP ORACLE ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
