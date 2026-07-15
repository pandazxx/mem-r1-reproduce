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
) -> list[dict]:
    """Apply memory operations to a retrieved context (list of id/text/timestamp
    dicts), approximating what post-op retrieval would see (docs/memory-manager-rl.md).

    UPDATE/DELETE must target a known bank id (``valid_ids``); targeting a valid
    id that was not retrieved into the context is allowed (UPDATE appends, DELETE
    is a no-op). Raises OperationError on unknown ids or missing fields.
    """
    spliced = [dict(m) for m in context]
    for op in operations:
        if op.op == "NOOP":
            continue
        if op.op == "ADD":
            if not op.text:
                raise OperationError("ADD requires text")
            spliced.append(
                {"id": None, "text": op.text, "timestamp": op.timestamp or default_timestamp}
            )
            continue
        # UPDATE / DELETE target an existing entry
        if op.id is None or op.id not in valid_ids:
            raise OperationError(f"{op.op}: unknown memory id {op.id!r}")
        if op.op == "UPDATE":
            if not op.text:
                raise OperationError("UPDATE requires text")
            for entry in spliced:
                if entry["id"] == op.id:
                    entry["text"] = op.text
                    entry["timestamp"] = op.timestamp or entry["timestamp"]
                    break
            else:
                spliced.append(
                    {"id": op.id, "text": op.text, "timestamp": op.timestamp or default_timestamp}
                )
        elif op.op == "DELETE":
            spliced = [entry for entry in spliced if entry["id"] != op.id]
    return spliced


def episode_valid_ids(episode: dict) -> set[str]:
    ids = {m["id"] for m in episode["related"]}
    ids.update(m["id"] for m in episode["context"] if m["id"] is not None)
    return ids


def _context_entries(context: list[dict]) -> list[MemoryEntry]:
    return [
        MemoryEntry(id=m["id"] or "", text=m["text"], timestamp=m["timestamp"]) for m in context
    ]


def manager_reward(
    completion: str, episode: dict, answer_llm: LLMFn, *, metric: str = "f1"
) -> float:
    """Outcome reward for one Memory Manager completion.

    parse ops -> splice the episode's precomputed context -> frozen Answer
    Agent answers the linked question -> token-level metric vs gold.
    Invalid JSON, unknown ops, or unknown ids score 0.
    """
    try:
        operations = parse_operations(completion)
        spliced = splice_context(
            episode["context"],
            operations,
            valid_ids=episode_valid_ids(episode),
            default_timestamp=episode["turn"]["date_time"],
        )
    except OperationError:
        return 0.0
    prompt = ANSWER_PROMPT.format(
        memories=format_memories(_context_entries(spliced)), question=episode["question"]
    )
    prediction = parse_answer(answer_llm(prompt))
    return REWARD_METRICS[metric](prediction, episode["answer"])


def make_manager_trl_reward(answer_llm: LLMFn, metric: str = "f1"):
    """TRL reward function: completions + the dataset's ``episode`` column."""

    def reward(completions: list, episode: list[dict], **kwargs) -> list[float]:
        return [
            manager_reward(_completion_text(completion), ep, answer_llm, metric=metric)
            for completion, ep in zip(completions, episode, strict=True)
        ]

    reward.__name__ = f"manager_reward_{metric}"
    return reward


def load_episodes(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]
