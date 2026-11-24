"""Independent Python reference scheduler oracle for Continuous Batching & Chunked Prefill."""
from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys

POLICY_DECODE_FIRST = 0
POLICY_PREFILL_FIRST = 1

STATE_WAITING = 0
STATE_PREFILLING = 1
STATE_DECODING = 2
STATE_PREEMPTED = 3
STATE_FINISHED = 4
STATE_CANCELLED = 5

STEP_NONE = 0xFFFFFFFF


def ceil_div(num: int, den: int) -> int:
    return 0 if num == 0 else (num - 1) // den + 1


@dataclass
class RefRequest:
    request_id: int
    prompt_len: int
    max_new_tokens: int
    priority: int
    arrival_seq: int
    enqueue_step: int
    computed_tokens: int = 0
    generated_tokens: int = 0
    allocated_blocks: int = 0
    first_token_step: int = STEP_NONE
    completion_step: int = STEP_NONE
    preempt_count: int = 0
    state: int = STATE_WAITING
    eos_stopped: int = 0
    active_in_plan: bool = False


class ReferenceScheduler:
    def __init__(
        self,
        max_running_seqs: int,
        max_waiting_seqs: int,
        max_tokens_per_step: int,
        total_kv_blocks: int,
        block_size: int,
        policy: int,
    ) -> None:
        self.max_running_seqs = max_running_seqs
        self.max_waiting_seqs = max_waiting_seqs
        self.max_tokens_per_step = max_tokens_per_step
        self.total_kv_blocks = total_kv_blocks
        self.block_size = block_size
        self.policy = policy

        self.current_step = 0
        self.next_arrival_seq = 0
        self.used_kv_blocks = 0
        self.free_kv_blocks = total_kv_blocks
        self.finished_requests = 0
        self.cancelled_requests = 0
        self.total_preemptions = 0
        self.total_generated_tokens = 0
        self.total_scheduled_tokens = 0

        self.requests: dict[int, RefRequest] = {}
        self.waiting_queue: list[RefRequest] = []
        self.running_list: list[RefRequest] = []
        self.pending_entries: list[dict[str, int]] = []
        self.pending_total_tokens = 0

    def _rank_key(self, req: RefRequest) -> tuple[int, int]:
        # Higher priority first, then smaller arrival_seq.
        return (-req.priority, req.arrival_seq)

    def _alloc(self, req: RefRequest, n: int) -> None:
        assert n <= self.free_kv_blocks
        self.free_kv_blocks -= n
        self.used_kv_blocks += n
        req.allocated_blocks += n
        assert self.used_kv_blocks + self.free_kv_blocks == self.total_kv_blocks

    def _release_all(self, req: RefRequest) -> None:
        self.used_kv_blocks -= req.allocated_blocks
        self.free_kv_blocks += req.allocated_blocks
        req.allocated_blocks = 0
        assert self.used_kv_blocks + self.free_kv_blocks == self.total_kv_blocks

    def enqueue(self, req_id: int, prompt_len: int, max_new: int, priority: int) -> None:
        assert prompt_len + max_new - 1 <= self.total_kv_blocks * self.block_size
        assert req_id not in self.requests
        assert len(self.waiting_queue) < self.max_waiting_seqs
        req = RefRequest(
            request_id=req_id,
            prompt_len=prompt_len,
            max_new_tokens=max_new,
            priority=priority,
            arrival_seq=self.next_arrival_seq,
            enqueue_step=self.current_step,
        )
        self.next_arrival_seq += 1
        self.requests[req_id] = req
        self.waiting_queue.append(req)

    def cancel(self, req_id: int) -> None:
        req = self.requests[req_id]
        assert req.state not in (STATE_FINISHED, STATE_CANCELLED)
        assert not req.active_in_plan
        if req.state in (STATE_WAITING, STATE_PREEMPTED):
            self.waiting_queue.remove(req)
        else:
            self._release_all(req)
            self.running_list.remove(req)
        req.state = STATE_CANCELLED
        req.completion_step = self.current_step
        self.cancelled_requests += 1

    def _find_victim(self) -> RefRequest | None:
        candidates = [r for r in self.running_list if not r.active_in_plan]
        if not candidates:
            return None
        # Lowest priority, newest arrival (maximum _rank_key).
        return max(candidates, key=self._rank_key)

    def _preempt(self, victim: RefRequest, preempted_ids: list[int]) -> None:
        assert not victim.active_in_plan
        self._release_all(victim)
        self.running_list.remove(victim)
        victim.computed_tokens = 0
        victim.state = STATE_PREEMPTED
        victim.preempt_count += 1
        self.total_preemptions += 1
        self.waiting_queue.insert(0, victim)
        preempted_ids.append(victim.request_id)

    def step_plan(self) -> dict[str, object]:
        entries: list[dict[str, int]] = []
        preempted_ids: list[int] = []
        budget = self.max_tokens_per_step

        def append_entry(req: RefRequest, chunk: int, is_prefill: int, is_final: int, new_b: int) -> None:
            produces = 1 if (not is_prefill or is_final) else 0
            entries.append({
                "request_id": req.request_id,
                "token_offset": req.computed_tokens,
                "chunk_tokens": chunk,
                "is_prefill": is_prefill,
                "is_final": is_final,
                "produces": produces,
                "new_blocks": new_b,
                "total_blocks": req.allocated_blocks,
            })
            req.active_in_plan = True

        def plan_decodes() -> None:
            nonlocal budget
            decodes = sorted(
                [r for r in self.running_list if r.state == STATE_DECODING],
                key=self._rank_key,
            )
            for req in decodes:
                if budget < 1:
                    break
                if req.state != STATE_DECODING:
                    continue
                new_b = 1 if (req.computed_tokens % self.block_size == 0) else 0
                if new_b > self.free_kv_blocks:
                    victim = self._find_victim()
                    if victim is None:
                        continue
                    if victim is req and len(entries) > 0:
                        continue
                    self._preempt(victim, preempted_ids)
                    if victim is req:
                        continue
                self._alloc(req, new_b)
                append_entry(req, 1, 0, 0, new_b)
                budget -= 1

        def plan_continuing_prefills() -> None:
            nonlocal budget
            prefills = sorted(
                [r for r in self.running_list if r.state == STATE_PREFILLING],
                key=self._rank_key,
            )
            for req in prefills:
                if budget == 0:
                    break
                if req.state != STATE_PREFILLING:
                    continue
                target = req.prompt_len + req.generated_tokens
                rem = target - req.computed_tokens
                slack = req.allocated_blocks * self.block_size - req.computed_tokens
                max_fit = slack + self.free_kv_blocks * self.block_size
                if max_fit == 0:
                    victim = self._find_victim()
                    if victim is None:
                        continue
                    if victim is req and len(entries) > 0:
                        continue
                    self._preempt(victim, preempted_ids)
                    if victim is req:
                        continue
                    max_fit = slack + self.free_kv_blocks * self.block_size
                if max_fit == 0:
                    continue
                chunk = min(rem, budget, max_fit)
                total_needed = ceil_div(req.computed_tokens + chunk, self.block_size)
                new_b = total_needed - req.allocated_blocks
                self._alloc(req, new_b)
                is_final = 1 if (req.computed_tokens + chunk == target) else 0
                append_entry(req, chunk, 1, is_final, new_b)
                budget -= chunk

        def admit_waiting() -> None:
            nonlocal budget
            if preempted_ids:
                return
            while (
                self.waiting_queue
                and len(self.running_list) < self.max_running_seqs
                and budget > 0
                and self.free_kv_blocks > 0
            ):
                req = self.waiting_queue.pop(0)
                target = req.prompt_len + req.generated_tokens
                max_fit = self.free_kv_blocks * self.block_size
                chunk = min(target, budget, max_fit)
                new_b = ceil_div(chunk, self.block_size)
                req.state = STATE_PREFILLING
                self.running_list.append(req)
                self._alloc(req, new_b)
                is_final = 1 if chunk == target else 0
                append_entry(req, chunk, 1, is_final, new_b)
                budget -= chunk

        if self.policy == POLICY_DECODE_FIRST:
            plan_decodes()
            plan_continuing_prefills()
            admit_waiting()
        else:
            plan_continuing_prefills()
            admit_waiting()
            plan_decodes()

        total_tokens = sum(e["chunk_tokens"] for e in entries)
        prefill_tokens = sum(e["chunk_tokens"] for e in entries if e["is_prefill"])
        decode_tokens = sum(e["chunk_tokens"] for e in entries if not e["is_prefill"])
        assert total_tokens <= self.max_tokens_per_step
        assert len(self.running_list) <= self.max_running_seqs
        assert self.used_kv_blocks + self.free_kv_blocks == self.total_kv_blocks

        self.pending_entries = entries
        self.pending_total_tokens = total_tokens
        return {
            "step_index": self.current_step,
            "entry_count": len(entries),
            "total_tokens": total_tokens,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "preempted_ids": preempted_ids,
            "used_kv_blocks": self.used_kv_blocks,
            "free_kv_blocks": self.free_kv_blocks,
            "entries": entries,
        }

    def step_commit(self, eos_flags: list[int]) -> None:
        assert len(eos_flags) == len(self.pending_entries)
        for e, eos in zip(self.pending_entries, eos_flags):
            req = self.requests[e["request_id"]]
            req.computed_tokens += e["chunk_tokens"]
            req.active_in_plan = False
            if e["produces"]:
                if req.generated_tokens == 0:
                    req.first_token_step = self.current_step
                req.generated_tokens += 1
                self.total_generated_tokens += 1
                if eos == 1 or req.generated_tokens >= req.max_new_tokens:
                    self._release_all(req)
                    self.running_list.remove(req)
                    req.state = STATE_FINISHED
                    req.eos_stopped = eos
                    req.completion_step = self.current_step
                    self.finished_requests += 1
                else:
                    req.state = STATE_DECODING
            else:
                req.state = STATE_PREFILLING
        self.total_scheduled_tokens += self.pending_total_tokens
        self.pending_entries = []
        self.pending_total_tokens = 0
        self.current_step += 1


def verify(binary: str) -> None:
    proc = subprocess.run(
        [binary, "--oracle"], check=True, capture_output=True, text=True
    )
    lines = proc.stdout.splitlines()
    idx = 0
    cases = 0
    total_steps = 0
    total_entries = 0
    total_preemptions = 0

    while idx < len(lines):
        parts = lines[idx].split()
        assert parts[0] == "CASE" and len(parts) == 8, f"bad CASE line: {lines[idx]!r}"
        case_name = parts[1]
        cfg = [int(x) for x in parts[2:]]
        ref = ReferenceScheduler(*cfg)
        idx += 1

        while idx < len(lines):
            p = lines[idx].split()
            tag = p[0]
            if tag == "ENQUEUE":
                step, rid, plen, mnew, prio = int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])
                assert step == ref.current_step
                ref.enqueue(rid, plen, mnew, prio)
                idx += 1
            elif tag == "CANCEL":
                step, rid = int(p[1]), int(p[2])
                assert step == ref.current_step
                ref.cancel(rid)
                idx += 1
            elif tag == "STEP":
                act_step = int(p[1])
                act_n = int(p[2])
                act_tot = int(p[3])
                act_pre = int(p[4])
                act_dec = int(p[5])
                act_pcnt = int(p[6])
                act_used = int(p[7])
                act_free = int(p[8])
                act_pids = [int(tok.split(":")[1]) for tok in p[9:]]
                exp = ref.step_plan()
                assert (
                    act_step == exp["step_index"]
                    and act_n == exp["entry_count"]
                    and act_tot == exp["total_tokens"]
                    and act_pre == exp["prefill_tokens"]
                    and act_dec == exp["decode_tokens"]
                    and act_pcnt == len(exp["preempted_ids"])
                    and act_pids == exp["preempted_ids"]
                    and act_used == exp["used_kv_blocks"]
                    and act_free == exp["free_kv_blocks"]
                ), f"{case_name} step {act_step} mismatch:\n  actual={p}\n  expected={exp}"
                idx += 1

                eos_flags: list[int] = []
                for e_idx in range(act_n):
                    ep = lines[idx].split()
                    assert ep[0] == "ENTRY" and len(ep) == 10
                    act_e = [int(x) for x in ep[1:]]
                    exp_e = exp["entries"][e_idx]
                    expected_tuple = [
                        exp_e["request_id"],
                        exp_e["token_offset"],
                        exp_e["chunk_tokens"],
                        exp_e["is_prefill"],
                        exp_e["is_final"],
                        exp_e["produces"],
                        exp_e["new_blocks"],
                        exp_e["total_blocks"],
                    ]
                    assert act_e[:8] == expected_tuple, (
                        f"{case_name} step {act_step} entry {e_idx} mismatch: "
                        f"actual={act_e[:8]} expected={expected_tuple}"
                    )
                    eos_flags.append(act_e[8])
                    idx += 1
                    total_entries += 1

                ref.step_commit(eos_flags)
                pp = lines[idx].split()
                assert pp[0] == "POST" and len(pp) == 7
                assert [int(x) for x in pp[1:]] == [
                    act_step,
                    len(ref.running_list),
                    len(ref.waiting_queue),
                    ref.finished_requests,
                    ref.used_kv_blocks,
                    ref.free_kv_blocks,
                ], f"{case_name} POST mismatch at step {act_step}"
                idx += 1
                total_steps += 1
                total_preemptions += act_pcnt
            elif tag == "REQ":
                vals = [int(x) for x in p[1:]]
                rid = vals[0]
                r = ref.requests[rid]
                ttft = (
                    r.first_token_step - r.enqueue_step + 1
                    if r.first_token_step != STEP_NONE
                    else 0
                )
                exp_vals = [
                    r.request_id,
                    r.state,
                    r.computed_tokens,
                    r.generated_tokens,
                    r.allocated_blocks,
                    r.first_token_step,
                    r.completion_step,
                    ttft,
                    r.preempt_count,
                    r.eos_stopped,
                ]
                assert vals == exp_vals, f"{case_name} REQ {rid} mismatch: {vals} != {exp_vals}"
                idx += 1
            elif tag == "END":
                vals = [int(x) for x in p[1:]]
                assert vals == [
                    ref.current_step,
                    ref.finished_requests,
                    ref.cancelled_requests,
                    ref.used_kv_blocks,
                    ref.free_kv_blocks,
                    ref.total_preemptions,
                    ref.total_generated_tokens,
                    ref.total_scheduled_tokens,
                ], f"{case_name} END mismatch: {vals}"
                assert ref.used_kv_blocks == 0 and ref.free_kv_blocks == ref.total_kv_blocks
                idx += 1
                cases += 1
                break
            else:
                raise AssertionError(f"unexpected tag: {tag!r}")

    assert cases == 4, f"expected 4 oracle cases, saw {cases}"
    assert total_preemptions >= 2, f"expected preemptions in oracle suite, saw {total_preemptions}"
    print(
        f"CONTINUOUS_BATCH Python oracle: PASS ({cases} cases, {total_steps} steps, "
        f"{total_entries} plan entries, {total_preemptions} preemptions)"
    )


if __name__ == "__main__":
    verify(sys.argv[1])
