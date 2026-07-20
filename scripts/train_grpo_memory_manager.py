"""GRPO-train the Memory Manager (M4). GPU box only: uv sync --extra train.

Reads precomputed episodes (no API access needed) and trains a LoRA adapter
with TRL's GRPOTrainer. The reward runs the frozen Answer Agent on the
op-spliced context using the policy's own base weights with the adapter
disabled — no second model in GPU memory. See docs/memory-manager-rl.md.
"""

import argparse
import os
import sys
from pathlib import Path

# reduce CUDA fragmentation; a 24 GB card runs near the limit
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def make_frozen_answer_batch(holder: dict, max_new_tokens: int):
    """Batch-answer with the policy's base weights (LoRA disabled), greedy.

    ``holder`` is filled in after the trainer instantiates the model —
    the reward function is constructed first, so it late-binds. All prompts
    of a reward call (one per completion in the group) run as a single
    left-padded generate, roughly halving step time vs sequential calls.
    """
    import torch

    def answer_batch(prompts: list[str]) -> list[str]:
        model, tokenizer = holder["model"], holder["tokenizer"]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
            )
            for p in prompts
        ]
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # right padding corrupts decoder-only generation
        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            tokenizer.padding_side = previous_side
        inputs = inputs.to(model.device)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), model.disable_adapter():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        finally:
            if was_training:
                model.train()
        generated = output[:, inputs["input_ids"].shape[1] :]
        return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated]

    return answer_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo-manager-qwen3b.yaml")
    args = parser.parse_args()

    import yaml
    from datasets import Dataset
    from peft import LoraConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    from memory_r1.manager import build_manager_prompt, load_episodes, make_manager_trl_reward

    config = yaml.safe_load((ROOT / args.config).read_text())
    episodes = load_episodes(ROOT / config["episodes"])
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": build_manager_prompt(ep)}],
                "episode": ep,
            }
            for ep in episodes
        ]
    )
    print(f"model: {config['model']}, {len(dataset)} training episodes")

    holder: dict = {}
    answer_batch = make_frozen_answer_batch(holder, config.get("answer_max_new_tokens", 256))
    trainer = GRPOTrainer(
        model=config["model"],
        reward_funcs=make_manager_trl_reward(
            answer_batch,
            config["reward_metric"],
            cap=config.get("context_cap"),
            add_penalty=config.get("add_penalty", 0.0),
        ),
        args=GRPOConfig(**config["grpo"]),
        train_dataset=dataset,
        peft_config=LoraConfig(**config["lora"]),
    )
    holder["model"] = trainer.model
    holder["tokenizer"] = trainer.processing_class

    output_dir = ROOT / config["grpo"]["output_dir"]
    last_checkpoint = get_last_checkpoint(output_dir) if output_dir.is_dir() else None
    if last_checkpoint:
        print(f"resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    print(f"saved adapter to {config['grpo']['output_dir']}")


if __name__ == "__main__":
    main()
