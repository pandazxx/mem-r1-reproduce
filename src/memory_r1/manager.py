"""Memory Manager RL plumbing for M4: prompt, context splicing, reward.

Pure Python (no torch/trl imports) like grpo.py, so the episode semantics
and reward path are unit-testable offline. Episodes are precomputed by
scripts/build_manager_episodes.py (see docs/memory-manager-rl.md); the only
model call at reward time is the frozen Answer Agent generation, injected
as a plain LLMFn.
"""

from __future__ import annotations

import json
from pathlib import Path

from memory_r1.answer_agent import ANSWER_PROMPT, format_memories, parse_answer
from memory_r1.bootstrap import LLMFn
from memory_r1.grpo import REWARD_METRICS, _completion_text
from memory_r1.memory_bank import MemoryEntry, MemoryOperation, OperationError, parse_operations

RELATED_K = 6

MANAGER_PROMPT = """\
You maintain a long-term memory bank for a conversation between two people. \
New facts were just extracted from one dialogue turn. Decide how to integrate \
them into the memory bank using these operations:

- ADD: store a new fact as a new memory ("text" required)
- UPDATE: rewrite an existing memory, consolidating it with the new information ("id" and "text" required)
- DELETE: remove an existing memory that the new information contradicts ("id" required)
- NOOP: the new information is not worth storing

New facts (from the turn spoken at {date_time}):
{facts}

Related existing memories:
{related}

Respond with only JSON in this exact format:
{{"operations": [{{"operation": "ADD", "text": "..."}}, {{"operation": "UPDATE", "id": "...", "text": "..."}}]}}"""


def format_facts(facts: list[str]) -> str:
    return "\n".join(f"- {fact}" for fact in facts) if facts else "(none)"


def format_related(related: list[dict]) -> str:
    if not related:
        return "(the memory bank has no related memories)"
    return "\n".join(f"- id={m['id']} [{m['timestamp']}] {m['text']}" for m in related)


def build_manager_prompt(episode: dict) -> str:
    return MANAGER_PROMPT.format(
        date_time=episode["turn"]["date_time"],
        facts=format_facts(episode["facts"]),
        related=format_related(episode["related"]),
    )


def splice_context(
    context: list[dict],
    operations: list[MemoryOperation],
    *,
    valid_ids: set[str],
    default_timestamp: str | None = None,
    cap: int | None = None,
) -> list[dict]:
    """Apply memory operations to a retrieved context (list of id/text/timestamp
    dicts), approximating what post-op retrieval would see (docs/memory-manager-rl.md).

    UPDATE/DELETE must target a known bank id (``valid_ids``); targeting a valid
    id that was not retrieved into the context is allowed (UPDATE appends, DELETE
    is a no-op). Raises OperationError on unknown ids or missing fields.

    With ``cap``, the result is limited to that many memories: new entries
    (ADDs and appended UPDATEs) displace the lowest-ranked retrieved entries
    instead of extending the context — so an ADD competes for a slot rather
    than getting free extra visibility the real top-k retrieval would not
    grant (issue #17).
    """
    originals = [dict(m) for m in context]
    appended: list[dict] = []
    for op in operations:
        if op.op == "NOOP":
            continue
        if op.op == "ADD":
            if not op.text:
                raise OperationError("ADD requires text")
            appended.append(
                {"id": None, "text": op.text, "timestamp": op.timestamp or default_timestamp}
            )
            continue
        # UPDATE / DELETE target an existing entry
        if op.id is None or op.id not in valid_ids:
            raise OperationError(f"{op.op}: unknown memory id {op.id!r}")
        if op.op == "UPDATE":
            if not op.text:
                raise OperationError("UPDATE requires text")
            for entry in originals + appended:
                if entry["id"] == op.id:
                    entry["text"] = op.text
                    entry["timestamp"] = op.timestamp or entry["timestamp"]
                    break
            else:
                appended.append(
                    {"id": op.id, "text": op.text, "timestamp": op.timestamp or default_timestamp}
                )
        elif op.op == "DELETE":
            originals = [entry for entry in originals if entry["id"] != op.id]
            appended = [entry for entry in appended if entry["id"] != op.id]
    if cap is not None and len(originals) + len(appended) > cap:
        appended = appended[:cap]
        originals = originals[: cap - len(appended)]  # context is relevance-ordered: drop the tail
    return originals + appended


def episode_valid_ids(episode: dict) -> set[str]:
    ids = {m["id"] for m in episode["related"]}
    ids.update(m["id"] for m in episode["context"] if m["id"] is not None)
    return ids


def _context_entries(context: list[dict]) -> list[MemoryEntry]:
    return [
        MemoryEntry(id=m["id"] or "", text=m["text"], timestamp=m["timestamp"]) for m in context
    ]


def _spliced_answer_prompt(
    completion: str, episode: dict, *, cap: int | None = None
) -> tuple[str, list[MemoryOperation]] | None:
    """The Answer Agent prompt after applying the completion's ops (plus the
    parsed ops), or None when the completion is invalid (bad JSON, unknown
    op, unknown id)."""
    try:
        operations = parse_operations(completion)
        spliced = splice_context(
            episode["context"],
            operations,
            valid_ids=episode_valid_ids(episode),
            default_timestamp=episode["turn"]["date_time"],
            cap=cap,
        )
    except OperationError:
        return None
    prompt = ANSWER_PROMPT.format(
        memories=format_memories(_context_entries(spliced)), question=episode["question"]
    )
    return prompt, operations


def _penalized(score: float, operations: list[MemoryOperation], add_penalty: float) -> float:
    return score - add_penalty * sum(op.op == "ADD" for op in operations)


def manager_reward(
    completion: str,
    episode: dict,
    answer_llm: LLMFn,
    *,
    metric: str = "f1",
    cap: int | None = None,
    add_penalty: float = 0.0,
) -> float:
    """Outcome reward for one Memory Manager completion.

    parse ops -> splice the episode's precomputed context (capped, so ADDs
    compete for slots) -> frozen Answer Agent answers the linked question ->
    token-level metric vs gold, minus ``add_penalty`` per ADD (bank-growth
    cost, issue #17). Invalid JSON, unknown ops, or unknown ids score 0.
    """
    result = _spliced_answer_prompt(completion, episode, cap=cap)
    if result is None:
        return 0.0
    prompt, operations = result
    prediction = parse_answer(answer_llm(prompt))
    return _penalized(
        REWARD_METRICS[metric](prediction, episode["answer"]), operations, add_penalty
    )


def make_manager_trl_reward(
    answer_batch, metric: str = "f1", *, cap: int | None = None, add_penalty: float = 0.0
):
    """TRL reward function: completions + the dataset's ``episode`` column.

    ``answer_batch`` maps a list of Answer Agent prompts to a list of raw
    answers in one call, so the 8 reward generations per group run as one
    batched forward pass instead of sequentially (~2x step time otherwise).
    Invalid completions never reach the answerer and score 0. ``cap`` and
    ``add_penalty`` implement the issue-#17 reward semantics.
    """

    def reward(completions: list, episode: list[dict], **kwargs) -> list[float]:
        rewards = [0.0] * len(completions)
        prompts, slots = [], []
        for i, (completion, ep) in enumerate(zip(completions, episode, strict=True)):
            result = _spliced_answer_prompt(_completion_text(completion), ep, cap=cap)
            if result is not None:
                prompts.append(result[0])
                slots.append((i, result[1]))
        for (i, operations), raw in zip(
            slots, answer_batch(prompts) if prompts else [], strict=True
        ):
            prediction = parse_answer(raw)
            score = REWARD_METRICS[metric](prediction, episode[i]["answer"])
            rewards[i] = _penalized(score, operations, add_penalty)
        return rewards

    reward.__name__ = f"manager_reward_{metric}"
    return reward


def load_episodes(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def apply_operations(
    bank, operations: list[MemoryOperation], *, default_timestamp: str | None = None
) -> tuple[int, int]:
    """Apply parsed ops to a real MemoryBank, skipping invalid ones.

    Used by the *postprocessor* rebuild (M1 bank + manager edits): ADDs
    default to the turn's timestamp; ops that fail (unknown id, e.g. after
    an earlier DELETE) are skipped, not fatal. Returns (applied, skipped).
    """
    from dataclasses import replace

    applied = skipped = 0
    for op in operations:
        if op.op == "ADD" and op.timestamp is None:
            op = replace(op, timestamp=default_timestamp)
        try:
            bank.apply(op)
            applied += 1
        except OperationError:
            skipped += 1
    return applied, skipped


def construct_bank(m1_bank, turn_operations) -> tuple:
    """True-manager bank construction (issue #17): start from an *empty* bank;
    the manager is the source of truth for whether a fact is stored at all.

    ``turn_operations`` yields (timestamp, operations) in conversation turn
    order. NOOP stores nothing (unlike the postprocessor rebuild, where the
    M1 fact is already in the bank). UPDATE/DELETE ids refer to the M1
    entries shown as "related" in the prompt; they resolve to the constructed
    entry holding that same text, and count as unresolved when the manager
    never stored that fact.

    Returns (bank, stats Counter).
    """
    from collections import Counter

    from memory_r1.memory_bank import MemoryBank

    bank = MemoryBank()
    m1_text = {entry.id: entry.text for entry in m1_bank.entries}
    constructed_id: dict[str, str] = {}  # text -> id in the constructed bank
    stats: Counter = Counter()
    for timestamp, operations in turn_operations:
        for op in operations:
            stats[op.op] += 1
            if op.op == "NOOP":
                continue
            if op.op == "ADD":
                if not op.text:
                    stats["invalid"] += 1
                    continue
                entry = bank.add(op.text, timestamp=op.timestamp or timestamp)
                constructed_id.setdefault(op.text, entry.id)
                continue
            target_text = m1_text.get(op.id or "")
            target_id = constructed_id.get(target_text) if target_text else None
            if target_id is None or bank.get(target_id) is None:
                stats["unresolved"] += 1
                continue
            if op.op == "UPDATE":
                if not op.text:
                    stats["invalid"] += 1
                    continue
                old = bank.get(target_id)
                bank.update(target_id, op.text, op.timestamp)
                constructed_id.pop(old.text, None)
                constructed_id.setdefault(op.text, target_id)
            elif op.op == "DELETE":
                old = bank.get(target_id)
                bank.delete(target_id)
                constructed_id.pop(old.text, None)
    return bank, stats
