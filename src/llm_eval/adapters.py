from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class ModelAdapter:
    def generate(self, prompt: str, seed: int, temperature: float, top_p: float, max_new_tokens: int, system_prompt: str | None = None) -> str:
        raise NotImplementedError


class HuggingFaceAdapter(ModelAdapter):
    def __init__(self, model_id: str, revision: str | None = None, device: str = "auto", local_dir: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Hugging Face support requires requirements.txt") from exc
        self._torch = torch
        model_path = local_dir
        model_root = os.environ.get("PJ_LLM_MODEL_ROOT")
        configured_path = Path(model_path) if model_path else None
        if model_root and configured_path and not configured_path.is_absolute() and configured_path.parts and configured_path.parts[0] == "models":
            model_path = str(Path(model_root) / Path(*configured_path.parts[1:]))
        model_path = model_path if model_path and os.path.isdir(model_path) else model_id
        load_kwargs: dict[str, Any] = {"revision": revision} if revision and model_path == model_id else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
        kwargs: dict[str, Any] = dict(load_kwargs)
        use_cuda = torch.cuda.is_available() and (
            device in {"auto", "cuda"} or (isinstance(device, str) and device.startswith("cuda:"))
        )
        if use_cuda:
            kwargs["torch_dtype"] = torch.float16
            if device in {"auto", "cuda"}:
                kwargs["device_map"] = "auto"
            else:
                kwargs["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if device not in {"auto", "cuda"} and not (isinstance(device, str) and device.startswith("cuda:")):
            self.model.to(device)

    def generate(self, prompt: str, seed: int, temperature: float, top_p: float, max_new_tokens: int, system_prompt: str | None = None) -> str:
        self._torch.manual_seed(seed)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            encoded = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
            # transformers>=5 的 apply_chat_template 返回 BatchEncoding，需取出 input_ids 张量。
            inputs = encoded if self._torch.is_tensor(encoded) else encoded["input_ids"]
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        inputs = inputs.to(self.model.device)
        do_sample = temperature > 0
        with self._torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def score_continuation(self, prompt: str, continuation: str) -> float:
        """返回 continuation 在 prompt 条件下的对数概率之和（prompt + continuation 直接拼接）。"""
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        full_ids = self.tokenizer(prompt + continuation, return_tensors="pt").input_ids.to(self.model.device)
        p_len = prompt_ids.shape[-1]
        with self._torch.no_grad():
            logits = self.model(full_ids).logits[0]
        log_probs = logits.log_softmax(-1)
        cont_ids = full_ids[0, p_len:]
        total = 0.0
        for i in range(cont_ids.shape[0]):
            total += log_probs[p_len - 1 + i, cont_ids[i]].item()
        return total

    def score_answer(self, prompt: str, answer: str) -> float:
        """官方 TruthfulQA MC 评分：返回 '\nA: ' + answer 在 prompt 条件下的对数概率之和。"""
        return self.score_continuation(prompt, "\nA: " + answer)


def build_adapter(spec: Any) -> ModelAdapter:
    if spec.adapter in {"hf", "huggingface"}:
        return HuggingFaceAdapter(spec.model_id, spec.revision, spec.device, spec.local_dir)
    raise ValueError(f"Unknown adapter: {spec.adapter}")

