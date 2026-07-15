"""Local transformers inference for the (GRPO-trained) Answer Agent.

Needs the `train` extras (uv sync --extra train). Runs on CUDA, Apple
Silicon (MPS), or CPU. Prompts go through the model's chat template with
greedy decoding, mirroring both GRPO training (chat template) and the M2
API harness (temperature 0), so eval numbers stay comparable.
"""

from __future__ import annotations

from memory_r1.bootstrap import LLMFn


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_local_llm(model: str, adapter: str | None = None, max_new_tokens: int = 256) -> LLMFn:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _pick_device()
    # bf16 is unreliable on MPS; fp16 there, fp32 as the CPU fallback
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
    tokenizer = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel

        lm = PeftModel.from_pretrained(lm, adapter)
    lm.eval()

    def llm(prompt: str) -> str:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        with torch.no_grad():
            output = lm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()

    return llm


def make_local_batch_llm(model: str, adapter: str | None = None, max_new_tokens: int = 256):
    """Batched greedy generation; ``use_adapter=False`` runs the frozen base.

    Returns ``generate(prompts, use_adapter=True) -> list[str]``. Used by the
    M4 manager eval / bank rebuild, where the adapter is the policy and the
    base model is the frozen Answer Agent.
    """
    from contextlib import nullcontext

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _pick_device()
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
    tokenizer = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel

        lm = PeftModel.from_pretrained(lm, adapter)
    lm.eval()

    def generate(prompts: list[str], *, use_adapter: bool = True) -> list[str]:
        if not prompts:
            return []
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
        inputs = inputs.to(device)
        context = lm.disable_adapter() if (adapter and not use_adapter) else nullcontext()
        with torch.no_grad(), context:
            output = lm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated]

    return generate
