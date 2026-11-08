from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


# 未在 experiment 里显式给出 seed 时的兜底值；真实种子以 configs/*.yaml 为准。
DEFAULT_SEED = 0


@dataclass
class ModelSpec:
    model_id: str
    adapter: str = "hf"
    revision: str | None = None
    device: str = "auto"
    local_dir: str | None = None


@dataclass
class JudgeSpec:
    """间接提示词注入的裁判模型（BIPIA 的 ModelEval，默认用 DeepSeek）。"""
    api_base: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 32
    max_retries: int = 5
    retry_backoff: float = 2.0


@dataclass
class EvaluationConfig:
    dataset_sources: list[dict[str, Any]]
    models: list[ModelSpec]
    seed: int = DEFAULT_SEED
    num_seeds: int = 3
    seeds: list[int] | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    judge: JudgeSpec | None = None


def _read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path) -> EvaluationConfig:
    raw = _read_yaml(path)
    experiment = raw.get("experiment", {})
    models = [ModelSpec(**item) for item in raw.get("models", [])]
    if not models:
        raise ValueError("Configuration must contain at least one model")
    dataset_sources = list(raw.get("dataset_sources", []))
    if not dataset_sources:
        raise ValueError("Configuration must contain dataset_sources")
    judge_raw = raw.get("judge") or {}
    judge = JudgeSpec(**judge_raw) if judge_raw else None
    seeds_raw = experiment.get("seeds")
    seeds = [int(s) for s in seeds_raw] if seeds_raw else None
    return EvaluationConfig(
        dataset_sources=dataset_sources,
        models=models,
        seed=int(experiment.get("seed", DEFAULT_SEED)),
        num_seeds=int(experiment.get("num_seeds", 3)),
        seeds=seeds,
        temperature=float(experiment.get("temperature", 0.0)),
        top_p=float(experiment.get("top_p", 1.0)),
        max_new_tokens=int(experiment.get("max_new_tokens", 128)),
        judge=judge,
    )

