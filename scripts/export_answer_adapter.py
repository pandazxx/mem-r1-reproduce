"""Export the GRPO-trained Answer Agent adapter for local use.

Merges the LoRA adapter into the base model (so tools like mlx-lm and
llama.cpp can load plain safetensors) and optionally pushes both the
adapter and the merged model to the HuggingFace Hub.

Runs wherever the adapter lives (the GPU pod; CPU is fine). Needs the
`train` extras and, for --push-repo, an HF_TOKEN env var with write access.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/grpo-answer-qwen3b")
    parser.add_argument("--out", default=None, help="merged model dir (default: <adapter>-merged)")
    parser.add_argument(
        "--push-repo", default=None, help="HF repo id (e.g. user/mem-r1-answer-qwen3b)"
    )
    args = parser.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(args.adapter)
    out_dir = Path(args.out) if args.out else adapter_dir.with_name(adapter_dir.name + "-merged")
    base = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
    print(f"merging {adapter_dir} into {base} -> {out_dir}")

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    print(f"saved merged model to {out_dir}")

    if args.push_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_repo, exist_ok=True)
        api.upload_folder(repo_id=args.push_repo, folder_path=out_dir)
        api.upload_folder(
            repo_id=args.push_repo,
            folder_path=adapter_dir,
            path_in_repo="lora",
            ignore_patterns=["checkpoint-*"],  # optimizer states, ~2 GB each
        )
        print(f"pushed merged model + lora/ to https://huggingface.co/{args.push_repo}")


if __name__ == "__main__":
    main()
