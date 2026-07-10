"""GRPO-train the Answer Agent (M3). GPU box only: uv sync --extra train.

Reads precomputed retrieval contexts (no API access needed) and trains a
LoRA adapter with TRL's GRPOTrainer. See docs/grpo-answer-agent.md.
"""

import argparse
import os
import sys
from pathlib import Path

# reduce CUDA fragmentation; a 24 GB card runs near the limit
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo-answer-qwen3b.yaml")
    args = parser.parse_args()

    import yaml
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from memory_r1.grpo import build_prompt, load_contexts, make_trl_reward

    config = yaml.safe_load((ROOT / args.config).read_text())
    contexts = load_contexts(ROOT / config["train_contexts"])
    # message format so TRL applies the chat template — matches how the M2
    # baseline consumed the same prompt via the chat API (and lets EOS end
    # completions instead of every generation running to the token cap)
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": build_prompt(c["memories"], c["question"])}],
                "answer": c["answer"],
            }
            for c in contexts
        ]
    )
    print(f"model: {config['model']}, {len(dataset)} training prompts")

    trainer = GRPOTrainer(
        model=config["model"],
        reward_funcs=make_trl_reward(config["reward_metric"]),
        args=GRPOConfig(**config["grpo"]),
        train_dataset=dataset,
        peft_config=LoraConfig(**config["lora"]),
    )
    trainer.train()
    trainer.save_model()
    print(f"saved adapter to {config['grpo']['output_dir']}")


if __name__ == "__main__":
    main()
