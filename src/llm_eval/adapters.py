from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        # Llama/Mistral 系列没有 pad token，批量打分时用 eos 填充（attention_mask 会屏蔽）。
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: dict[str, Any] = dict(load_kwargs)
        use_cuda = torch.cuda.is_available() and (
            device in {"auto", "cuda"} or (isinstance(device, str) and device.startswith("cuda:"))
        )
        if use_cuda:
            kwargs["torch_dtype"] = torch.float16
            if device in {"auto", "cuda"}:
                # 固定在一张卡上，不跨卡拆分。device_map="auto" 会按 0→1→2 顺序填卡，
                # 而绑定权重的模型（如 Qwen2.5-3B）lm_head 必须和 embed_tokens 同在
                # cuda:0；一旦 cuda:0 显存被别的任务占满，logits/log_softmax 必然 OOM。
                # 这里选可见设备中显存最空闲的一张，整个模型都放上去。
                target = max(
                    range(torch.cuda.device_count()),
                    key=lambda idx: torch.cuda.mem_get_info(idx)[0],
                )
                free_gib = torch.cuda.mem_get_info(target)[0] / 2**30
                logger.info("loading %s on cuda:%d (%.1f GiB free)", model_path, target, free_gib)
                kwargs["device_map"] = {"": target}
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

    def score_continuations(self, prompt: str, continuations: list[str], batch_size: int = 32) -> list[float]:
        """批量返回每个 continuation 在 prompt 条件下的对数概率之和。

        一次 forward 计算多个候选，BBQ 的 12 个选项 / TruthfulQA 的全部候选可以并排
        打分，避免逐条 forward 把重跑时间放大一个数量级。序列按右侧 padding，用
        attention_mask 只累加真实 continuation token 的 logprob。
        """
        if not continuations:
            return []
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        p_len = prompt_ids.shape[-1]
        self.tokenizer.padding_side = "right"
        scores: list[float] = []
        for start in range(0, len(continuations), batch_size):
            chunk = continuations[start : start + batch_size]
            batch = self.tokenizer(
                [prompt + continuation for continuation in chunk],
                return_tensors="pt",
                padding=True,
            )
            full_ids = batch.input_ids.to(self.model.device)
            attention_mask = batch.attention_mask.to(self.model.device)
            with self._torch.no_grad():
                logits = self.model(full_ids, attention_mask=attention_mask).logits
            log_probs = logits.log_softmax(-1)
            for row in range(full_ids.shape[0]):
                cont_ids = full_ids[row, p_len:]
                cont_mask = attention_mask[row, p_len:].bool()
                if cont_ids.numel() == 0:
                    scores.append(0.0)
                    continue
                positions = log_probs[row, p_len - 1 : p_len - 1 + cont_ids.numel(), :]
                token_scores = positions.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
                scores.append(float(token_scores[cont_mask].sum().item()))
        return scores

    def score_continuation(self, prompt: str, continuation: str) -> float:
        """返回 continuation 在 prompt 条件下的对数概率之和（prompt + continuation 直接拼接）。"""
        return self.score_continuations(prompt, [continuation])[0]

    def score_answer(self, prompt: str, answer: str) -> float:
        """官方 TruthfulQA MC 评分：返回 '\nA: ' + answer 在 prompt 条件下的对数概率之和。"""
        return self.score_continuation(prompt, "\nA: " + answer)


def build_adapter(spec: Any) -> ModelAdapter:
    if spec.adapter in {"hf", "huggingface"}:
        return HuggingFaceAdapter(spec.model_id, spec.revision, spec.device, spec.local_dir)
    raise ValueError(f"Unknown adapter: {spec.adapter}")

