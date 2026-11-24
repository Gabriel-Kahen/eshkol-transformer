"""Independent Python reference RadixTree + LRU + binary64 GQA oracle."""
import math
import subprocess
import sys

OK = 0
ERR_OOM = -7
ERR_INVALID_HANDLE = -8
ERR_DOUBLE_UNPIN = -9


def read_vec(lines):
    return [float.fromhex(tok) for tok in next(lines).split()]


class Node:
    def __init__(self, node_id, parent=None, tokens=None, k_rows=None, v_rows=None):
        self.node_id = node_id
        self.parent = parent
        self.tokens = list(tokens or [])
        self.k_rows = list(k_rows or [])
        self.v_rows = list(v_rows or [])
        self.children = {}  # first_token -> Node
        self.self_pin = 0
        self.pin_count = 0
        self.last_tick = 0
        self.alive = True


class ReferenceRadixTree:
    def __init__(self, max_tokens, max_nodes):
        self.max_tokens = max_tokens
        self.max_nodes = max_nodes
        self.root = Node(0, parent=None)
        self.free_nodes = list(range(max_nodes - 1, 0, -1))
        self.used_tokens = 0
        self.pinned_tokens = 0
        self.active_nodes = 1
        self.lru_tick = 0
        self.total_evicted_tokens = 0
        self.all_nodes = {0: self.root}

    @property
    def free_tokens(self):
        return self.max_tokens - self.used_tokens

    @property
    def evictable_tokens(self):
        return self.used_tokens - self.pinned_tokens

    def _ancestors(self, node):
        res = set()
        cur = node
        while cur is not None:
            res.add(cur)
            cur = cur.parent
        return res

    def _touch(self, node):
        self.lru_tick += 1
        cur = node
        while cur is not None:
            cur.last_tick = self.lru_tick
            cur = cur.parent

    def _walk(self, tokens):
        cur = self.root
        pos = 0
        n = len(tokens)
        while pos < n:
            child = cur.children.get(tokens[pos])
            if child is None:
                return pos, cur, 0
            k = 0
            while k < len(child.tokens) and pos + k < n and child.tokens[k] == tokens[pos + k]:
                k += 1
            pos += k
            if k == len(child.tokens):
                cur = child
            else:
                return pos, child, k
        return pos, cur, 0

    def _select_victim(self, protected):
        candidates = [
            n for n in self.all_nodes.values()
            if n is not self.root and n.alive and not n.children
            and n.pin_count == 0 and n not in protected
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda n: (n.last_tick, n.node_id))
        return candidates[0]

    def _evict_leaf(self, leaf):
        assert leaf.alive and not leaf.children and leaf.pin_count == 0
        ev = len(leaf.tokens)
        self.used_tokens -= ev
        self.total_evicted_tokens += ev
        del leaf.parent.children[leaf.tokens[0]]
        leaf.alive = False
        del self.all_nodes[leaf.node_id]
        self.active_nodes -= 1
        self.free_nodes.append(leaf.node_id)

    def _split(self, child, offset):
        p_id = self.free_nodes.pop()
        parent = child.parent
        p = Node(
            p_id,
            parent=parent,
            tokens=child.tokens[:offset],
            k_rows=child.k_rows[:offset],
            v_rows=child.v_rows[:offset],
        )
        p.pin_count = child.pin_count
        p.self_pin = 0
        p.last_tick = child.last_tick
        child.tokens = child.tokens[offset:]
        child.k_rows = child.k_rows[offset:]
        child.v_rows = child.v_rows[offset:]
        child.parent = p
        parent.children[p.tokens[0]] = p
        p.children[child.tokens[0]] = child
        self.all_nodes[p_id] = p
        self.active_nodes += 1
        return p

    def match_prefix(self, tokens):
        matched, cur, off = self._walk(tokens)
        did_split = 0
        if off > 0:
            protected = self._ancestors(cur)
            if not self.free_nodes:
                victim = self._select_victim(protected)
                if victim is None:
                    return ERR_OOM, 0, 0, None
                self._evict_leaf(victim)
            cur = self._split(cur, off)
            did_split = 1
        self._touch(cur)
        return OK, matched, did_split, cur

    def insert(self, tokens, k_rows, v_rows):
        matched, cur, off = self._walk(tokens)
        suffix_len = len(tokens) - matched
        need_split = 1 if off > 0 else 0
        need_leaf = 1 if suffix_len > 0 else 0
        needed_nodes = need_split + need_leaf
        needed_tokens = suffix_len

        protected = self._ancestors(cur)
        outside = [
            n for n in self.all_nodes.values()
            if n is not self.root and n.alive and n.pin_count == 0 and n not in protected
        ]
        out_nodes = len(outside)
        out_tokens = sum(len(n.tokens) for n in outside)
        tail_nodes = 1 if (need_split and need_leaf and cur.pin_count == 0) else 0
        tail_tokens = (len(cur.tokens) - off) if tail_nodes else 0

        if (need_split and len(self.free_nodes) + out_nodes < 1) or \
           (len(self.free_nodes) + out_nodes + tail_nodes < needed_nodes) or \
           (self.free_tokens + out_tokens + tail_tokens < needed_tokens):
            return ERR_OOM, 0, None

        if need_split:
            while len(self.free_nodes) < 1:
                self._evict_leaf(self._select_victim(protected))
            cur = self._split(cur, off)
            protected = self._ancestors(cur)

        while self.free_tokens < needed_tokens or len(self.free_nodes) < need_leaf:
            self._evict_leaf(self._select_victim(protected))

        if need_leaf:
            lid = self.free_nodes.pop()
            leaf = Node(
                lid,
                parent=cur,
                tokens=tokens[matched:],
                k_rows=k_rows[matched:],
                v_rows=v_rows[matched:],
            )
            cur.children[leaf.tokens[0]] = leaf
            self.all_nodes[lid] = leaf
            self.active_nodes += 1
            self.used_tokens += suffix_len
            cur = leaf

        self._touch(cur)
        return OK, matched, cur

    def pin(self, node):
        if node is None or not node.alive:
            return ERR_INVALID_HANDLE
        node.self_pin += 1
        cur = node
        while cur is not None:
            if cur is not self.root and cur.pin_count == 0:
                self.pinned_tokens += len(cur.tokens)
            cur.pin_count += 1
            cur = cur.parent
        return OK

    def unpin(self, node):
        if node is None or not node.alive:
            return ERR_INVALID_HANDLE
        if node.self_pin == 0:
            return ERR_DOUBLE_UNPIN
        node.self_pin -= 1
        cur = node
        while cur is not None:
            cur.pin_count -= 1
            if cur is not self.root and cur.pin_count == 0:
                self.pinned_tokens -= len(cur.tokens)
            cur = cur.parent
        return OK

    def gather_kv(self, node):
        chain = []
        cur = node
        while cur is not self.root:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        ks, vs = [], []
        for n in chain:
            ks.extend(n.k_rows)
            vs.extend(n.v_rows)
        return ks, vs


def causal_gqa_reference(hq, hkv, hd, scale, q, k_all, v_all, q_len):
    total_len = len(k_all)
    group = hq // hkv
    q_base = total_len - q_len
    out = [0.0] * (q_len * hq * hd)
    for qi in range(q_len):
        abs_pos = q_base + qi
        for qh in range(hq):
            kvh = qh // group
            q_off = (qi * hq + qh) * hd
            scores = []
            for t in range(abs_pos + 1):
                k_row = k_all[t]
                dot = math.fsum(
                    q[q_off + d] * k_row[kvh * hd + d] for d in range(hd)
                )
                scores.append(dot * scale)
            m = max(scores)
            ws = [math.exp(s - m) for s in scores]
            denom = math.fsum(ws)
            for d in range(hd):
                out[q_off + d] = math.fsum(
                    (ws[t] / denom) * v_all[t][kvh * hd + d]
                    for t in range(abs_pos + 1)
                )
    return out


def verify(binary):
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = iter(proc.stdout.splitlines())
    cfg = next(lines).split()
    assert cfg[0] == "CONFIG"
    max_tok, max_nodes, hq, hkv, hd = (int(cfg[i]) for i in range(1, 6))
    scale = float.fromhex(cfg[6])
    lkv = hkv * hd

    ref = ReferenceRadixTree(max_tok, max_nodes)
    handles = []
    ops_checked = 0
    attn_checks = 0
    max_abs_err = 0.0

    for line in lines:
        parts = line.split()
        assert parts[0] == "OP"
        step = int(parts[1])
        kind = parts[2]

        if kind == "INSERT":
            st = int(parts[3])
            matched = int(parts[4])
            hid = int(parts[5])
            used, free_t, pinned, evictable, active = (int(parts[i]) for i in range(6, 11))
            ev_tot, tick = int(parts[11]), int(parts[12])
            assert parts[13] == "TOKENS"
            tlen = int(parts[14])
            toks = [int(x) for x in parts[15:15 + tlen]]
            k_flat = read_vec(lines)
            v_flat = read_vec(lines)
            k_rows = [k_flat[i * lkv:(i + 1) * lkv] for i in range(tlen)]
            v_rows = [v_flat[i * lkv:(i + 1) * lkv] for i in range(tlen)]

            r_st, r_matched, r_node = ref.insert(toks, k_rows, v_rows)
            assert r_st == st, f"step {step} INSERT status {st} != {r_st}"
            if r_st == OK:
                assert r_matched == matched, f"step {step} matched {matched} != {r_matched}"
                assert hid == len(handles)
                handles.append(r_node)
        elif kind == "MATCH":
            st = int(parts[3])
            matched, split = int(parts[4]), int(parts[5])
            hid = int(parts[6])
            used, free_t, pinned, evictable, active = (int(parts[i]) for i in range(7, 12))
            ev_tot, tick, do_attn = int(parts[12]), int(parts[13]), int(parts[14])
            assert parts[15] == "TOKENS"
            tlen = int(parts[16])
            toks = [int(x) for x in parts[17:17 + tlen]]

            r_st, r_matched, r_split, r_node = ref.match_prefix(toks)
            assert (r_st, r_matched, r_split) == (st, matched, split), (
                f"step {step} MATCH {(st, matched, split)} != {(r_st, r_matched, r_split)}"
            )
            if r_st == OK:
                assert hid == len(handles)
                handles.append(r_node)
            if do_attn:
                q = read_vec(lines)
                sk = read_vec(lines)
                sv = read_vec(lines)
                actual = read_vec(lines)
                pk, pv = ref.gather_kv(r_node)
                k_all = pk + [sk[i * lkv:(i + 1) * lkv] for i in range(2)]
                v_all = pv + [sv[i * lkv:(i + 1) * lkv] for i in range(2)]
                expected = causal_gqa_reference(hq, hkv, hd, scale, q, k_all, v_all, 2)
                for va, ve in zip(actual, expected):
                    err = abs(va - ve)
                    assert err <= 1e-6, f"step {step} GQA err={err:.3e}"
                    if err > max_abs_err:
                        max_abs_err = err
                attn_checks += 1
        elif kind == "PIN":
            st, hid = int(parts[3]), int(parts[4])
            used, free_t, pinned, evictable, active = (int(parts[i]) for i in range(5, 10))
            ev_tot, tick = int(parts[10]), int(parts[11])
            r_st = ref.pin(handles[hid])
            assert r_st == st, f"step {step} PIN {st} != {r_st}"
        elif kind == "UNPIN":
            st, hid = int(parts[3]), int(parts[4])
            used, free_t, pinned, evictable, active = (int(parts[i]) for i in range(5, 10))
            ev_tot, tick = int(parts[10]), int(parts[11])
            r_st = ref.unpin(handles[hid])
            assert r_st == st, f"step {step} UNPIN {st} != {r_st}"
        else:
            raise AssertionError(f"unknown op {kind}")

        assert (used, free_t, pinned, evictable, active, ev_tot, tick) == (
            ref.used_tokens,
            ref.free_tokens,
            ref.pinned_tokens,
            ref.evictable_tokens,
            ref.active_nodes,
            ref.total_evicted_tokens,
            ref.lru_tick,
        ), (
            f"step {step} ({kind}) state mismatch: "
            f"C={(used, free_t, pinned, evictable, active, ev_tot, tick)} "
            f"Py={(ref.used_tokens, ref.free_tokens, ref.pinned_tokens, ref.evictable_tokens, ref.active_nodes, ref.total_evicted_tokens, ref.lru_tick)}"
        )
        ops_checked += 1

    assert ops_checked == 50, f"expected 50 trace ops, saw {ops_checked}"
    assert attn_checks >= 1, "expected at least one GQA attention verification"
    print(
        f"RADIX_KV Python oracle: PASS ({ops_checked} trace ops, "
        f"{attn_checks} GQA checks, max_abs_err={max_abs_err:.3e})"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
