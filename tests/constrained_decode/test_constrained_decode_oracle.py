#!/usr/bin/env python3
"""Independent Python oracle for Constrained Decoding FSA + JSON Schema Engine.

Verifies:
  1. Reverse-BFS co-accessibility dead-state pruning on a raw DFA with dead
     cycles, dead branches, and unreachable co-accessible states.
  2. Exact precomputed allowed_mask bitsets for EVERY state and EVERY token
     (byte + multi-byte + UTF-8 + EOS) against an independent byte-walk oracle.
  3. 100% json.loads() validity and exact schema key/type/enum compliance across
     64 randomized adversarial-logit trajectories, proving adversarial tokens
     are blocked on every trajectory and zero invalid JSON strings are emitted.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import deque


def verify_prune_oracle(native_bin: str) -> None:
    proc = subprocess.run(
        [native_bin, "--oracle-prune"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines[0].startswith("PRUNE_INFO ")
    _, ns, start, live_cnt, acc_cnt = lines[0].split()
    num_states = int(ns)
    assert num_states == 7 and int(start) == 0

    # Ground-truth raw transitions before pruning.
    raw_edges = [
        (0, ord("a"), 1),
        (0, ord("d"), 4),
        (1, ord("b"), 2),
        (1, ord("c"), 5),
        (2, ord("a"), 1),
        (2, ord("x"), 6),
        (4, ord("e"), 5),
        (5, ord("f"), 4),
        (3, ord("g"), 2),
    ]
    accepts = {2}
    rev: dict[int, set[int]] = {s: set() for s in range(num_states)}
    for u, _, v in raw_edges:
        rev[v].add(u)

    live = set(accepts)
    q = deque(accepts)
    while q:
        v = q.popleft()
        for u in rev[v]:
            if u not in live:
                live.add(u)
                q.append(u)

    assert int(live_cnt) == len(live) == 4  # {0, 1, 2, 3}
    assert int(acc_cnt) == 1

    expected_trans: dict[int, dict[int, int]] = {s: {} for s in range(num_states)}
    for u, b, v in raw_edges:
        if u in live and v in live:
            expected_trans[u][b] = v

    parsed_states = 0
    for ln in lines[1:]:
        parts = ln.split()
        assert parts[0] == "STATE"
        s = int(parts[1])
        s_live = int(parts[2].split("=")[1])
        s_acc = int(parts[3].split("=")[1])
        assert s_live == (1 if s in live else 0), f"state {s} live mismatch"
        assert s_acc == (1 if s in accepts else 0), f"state {s} accept mismatch"
        got_edges = {}
        for tok in parts[4:]:
            b_str, nxt_str = tok.split(":")
            got_edges[int(b_str)] = int(nxt_str)
        assert got_edges == expected_trans[s], (
            f"state {s} transitions mismatch: {got_edges} != {expected_trans[s]}"
        )
        parsed_states += 1
    assert parsed_states == num_states


def verify_json_oracle(native_bin: str) -> None:
    num_trajectories = 64
    proc = subprocess.run(
        [native_bin, "--oracle-json", "20261123", str(num_trajectories)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines[0].startswith("META ")
    _, ns_s, start_s, live_s, vocab_s, eos_s = lines[0].split()
    num_states = int(ns_s)
    start_state = int(start_s)
    live_count = int(live_s)
    vocab_size = int(vocab_s)
    eos_id = int(eos_s)
    assert live_count == num_states

    tokens: dict[int, bytes] = {}
    dfa_live: dict[int, bool] = {}
    dfa_acc: dict[int, bool] = {}
    dfa_trans: dict[int, dict[int, int]] = {}
    c_masks: dict[int, list[int]] = {}
    trajectories: list[tuple[int, int, int, bytes]] = []

    for ln in lines[1:]:
        parts = ln.split()
        tag = parts[0]
        if tag == "TOK":
            tid = int(parts[1])
            tlen = int(parts[2])
            hex_str = parts[3] if len(parts) > 3 else ""
            raw = bytes.fromhex(hex_str)
            assert len(raw) == tlen
            tokens[tid] = raw
        elif tag == "DFA_STATE":
            s = int(parts[1])
            dfa_live[s] = parts[2] == "1"
            dfa_acc[s] = parts[3] == "1"
            edges = {}
            for item in parts[4:]:
                b_str, nxt_str = item.split(":")
                edges[int(b_str)] = int(nxt_str)
            dfa_trans[s] = edges
        elif tag == "MASK":
            s = int(parts[1])
            c_masks[s] = [int(w, 16) for w in parts[2:]]
        elif tag == "TRAJ":
            traj_id = int(parts[1])
            steps = int(parts[2])
            blocks = int(parts[3])
            raw = bytes.fromhex(parts[4]) if len(parts) > 4 else b""
            trajectories.append((traj_id, steps, blocks, raw))
        else:
            raise AssertionError(f"unexpected oracle line: {ln}")

    assert len(tokens) == vocab_size
    assert len(dfa_trans) == num_states
    assert len(c_masks) == num_states
    assert len(trajectories) == num_trajectories

    # 1. Independently verify co-accessibility of the compiled JSON schema DFA.
    rev: dict[int, set[int]] = {s: set() for s in range(num_states)}
    for u, edges in dfa_trans.items():
        for v in edges.values():
            rev[v].add(u)
    acc_set = {s for s, is_a in dfa_acc.items() if is_a}
    live_oracle = set(acc_set)
    q = deque(acc_set)
    while q:
        v = q.popleft()
        for u in rev[v]:
            if u not in live_oracle:
                live_oracle.add(u)
                q.append(u)
    assert start_state in live_oracle
    assert len(live_oracle) == num_states

    # 2. Independently recompute allowed_mask bitsets for EVERY (state, token)
    #    and verify exact 64-bit word equality with C precomputation!
    mask_words = (vocab_size + 63) // 64
    for s in range(num_states):
        expected_words = [0] * mask_words
        if dfa_live[s]:
            for t in range(vocab_size):
                allowed = False
                if t == eos_id:
                    allowed = dfa_acc[s]
                elif len(tokens[t]) > 0:
                    cur = s
                    ok = True
                    for b in tokens[t]:
                        nxt = dfa_trans[cur].get(b)
                        if nxt is None or not dfa_live[nxt]:
                            ok = False
                            break
                        cur = nxt
                    allowed = ok
                if allowed:
                    expected_words[t // 64] |= 1 << (t % 64)
        assert c_masks[s] == expected_words, (
            f"state {s} mask mismatch: {c_masks[s]} != {expected_words}"
        )

    # 3. Verify 100% of adversarial-logit trajectories parse as valid UTF-8 JSON
    #    matching exact ordered keys, exact Python types, and enum membership.
    total_blocks = 0
    distinct_outputs: set[bytes] = set()
    for traj_id, steps, blocks, raw in trajectories:
        assert steps > 0
        assert blocks > 0, f"trajectory {traj_id} had zero adversarial blocks"
        total_blocks += blocks
        distinct_outputs.add(raw)

        text = raw.decode("utf-8")
        obj = json.loads(text)
        assert isinstance(obj, dict), f"traj {traj_id}: not an object: {text}"
        assert list(obj.keys()) == ["answer", "score", "ok", "note"], (
            f"traj {traj_id}: wrong key order: {list(obj.keys())}"
        )
        assert obj["answer"] in ("yes", "no", "maybe"), (
            f"traj {traj_id}: bad enum: {obj['answer']!r}"
        )
        assert type(obj["score"]) is int, (
            f"traj {traj_id}: score is not int: {obj['score']!r}"
        )
        assert type(obj["ok"]) is bool, (
            f"traj {traj_id}: ok is not bool: {obj['ok']!r}"
        )
        assert type(obj["note"]) is str, (
            f"traj {traj_id}: note is not str: {obj['note']!r}"
        )

    assert len(distinct_outputs) >= 16, "trajectories lacked diversity"
    print(
        f"CONSTRAINED_DECODE Python oracle: PASS "
        f"(states={num_states} vocab={vocab_size} trajectories={num_trajectories} "
        f"distinct={len(distinct_outputs)} adversarial_blocks={total_blocks})"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: test_constrained_decode_oracle.py <native_bin>", file=sys.stderr)
        return 2
    native_bin = sys.argv[1]
    verify_prune_oracle(native_bin)
    verify_json_oracle(native_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
