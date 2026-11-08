from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any

from .config import EvaluationConfig
from .evaluator import Result
from .metrics import mean_of
from .scoring import aggregate_summaries, summarize


def write_report_bundle(
    results: list[Result],
    config: EvaluationConfig,
    output_dir: Path,
    seeds: list[int] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(_json_safe(result.to_dict()), ensure_ascii=False) + "\n")

    # 专门记录裁判失败的 BIPIA 样本（重试耗尽仍无法判定），便于事后重跑/分析。
    failures = [r for r in results if r.metadata.get("bipia_judge_error")]
    if failures:
        with (output_dir / "judge_failures.jsonl").open("w", encoding="utf-8") as handle:
            for r in failures:
                handle.write(json.dumps({
                    "model_id": r.model_id,
                    "seed": r.seed,
                    "case_id": r.case_id,
                    "task": r.metadata.get("bipia_task", ""),
                    "attack_name": r.metadata.get("bipia_attack_name", ""),
                    "position": r.metadata.get("bipia_position", ""),
                    "response": r.response,
                    "error": r.metadata.get("bipia_judge_error", ""),
                    "attempts": r.metadata.get("bipia_judge_attempts", 0),
                }, ensure_ascii=False) + "\n")

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_id", "dimension", "n", "mean", "ci_low", "ci_high"])
        writer.writeheader()
        by_key: dict[tuple[str, str], list[float]] = {}
        for result in results:
            if isinstance(result.score, float) and math.isnan(result.score):
                continue
            by_key.setdefault((result.model_id, result.dimension), []).append(result.score)
        from .metrics import mean_ci
        for (model_id, dimension), values in sorted(by_key.items()):
            average, low, high = mean_ci(values)
            writer.writerow({"model_id": model_id, "dimension": dimension, "n": len(values), "mean": round(average * 100, 3), "ci_low": round(low * 100, 3), "ci_high": round(high * 100, 3)})

    # 每个随机种子独立汇总，再跨种子聚合成 mean ± std 的最终评分表。
    seed_values = list(seeds) if seeds else sorted({result.seed for result in results})
    if not seed_values:
        seed_values = [config.seed]
    per_seed = [summarize([r for r in results if r.seed == seed]) for seed in seed_values]
    summary = aggregate_summaries(per_seed)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_plot(summary, output_dir / "score_overview.png")
    _write_html(summary, output_dir / "report.html", n_seeds=len(seed_values))


def _fmt(value: Any, fmt: str = ".2f") -> str:
    """把聚合后的数值渲染为 ``mean ± std``（无 std 信息时只显示数值）。"""
    if value is None:
        return "N/A"
    if isinstance(value, dict) and "mean" in value:
        return f"{value['mean']:{fmt}} ± {value['std']:{fmt}}"
    return format(value, fmt)


def _json_safe(obj: Any) -> Any:
    """递归把 NaN 转为 None，使 results.jsonl 保持合法 JSON。"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _score_labels() -> dict[str, str]:
    return {
        "truthfulqa": "TruthfulQA",
        "halueval": "HaluEval",
        "bbq": "偏见歧视",
        "indirect_prompt_injection": "间接提示词注入",
    }


def _write_plot(summary: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    # 图表用英文标签：系统可能没有中文字体，中文会渲染成方框（glyph missing 警告）。
    plot_labels = {
        "truthfulqa": "TruthfulQA",
        "halueval": "HaluEval",
        "bbq": "Bias (BBQ)",
        "indirect_prompt_injection": "Prompt Injection",
    }
    models = [item["model_id"] for item in summary]
    dimensions = sorted({dimension for item in summary for dimension in item["scores"]})
    width = 0.8 / max(1, len(models))
    x = list(range(len(dimensions)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, item in enumerate(summary):
        values = [mean_of(item["scores"].get(dimension, 0.0)) for dimension in dimensions]
        positions = [value + index * width for value in x]
        ax.bar(positions, values, width=width, label=item["model_id"])
    tick_labels = [plot_labels.get(dimension, dimension) for dimension in dimensions]
    ax.set_xticks([value + width * max(0, len(models) - 1) / 2 for value in x], tick_labels)
    ax.tick_params(axis="x", labelrotation=20)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("PJ-LLM Trustworthiness Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_html(summary: list[dict[str, Any]], path: Path, n_seeds: int = 1) -> None:
    score_labels = _score_labels()
    rows = []
    for item in summary:
        bbq = item.get("bbq", {})
        bipia = item.get("bipia", {})
        judge_failed = mean_of(bipia.get("judge_failed", 0))
        per_task = "".join(
            f"<li>{task}: ASR {_fmt(info.get('asr', 0.0), '.3f')} (n={info.get('n', 0)})</li>"
            for task, info in bipia.get("per_task", {}).items()
        )
        if judge_failed:
            per_task = f"<li>裁判失败: {judge_failed:.0f} 样本</li>" + per_task
        dimensions = "".join(
            f"<li>{score_labels.get(key, key)}: {_fmt(value, '.2f')}</li>" for key, value in item["scores"].items()
        )
        rows.append(
            f"<tr><td>{html.escape(item['model_id'])}</td><td>{_fmt(item['overall_score'], '.2f')}</td>"
            f"<td>{_fmt(item['mc1_accuracy'], '.2f')}</td><td>{_fmt(item['mc2_accuracy'], '.2f')}</td>"
            f"<td>{_fmt(item['halueval_accuracy'], '.2f')}</td>"
            f"<td>{_fmt(bbq.get('accuracy', 0.0), '.3f')}</td><td>{_fmt(bbq.get('accuracy_ambiguous', 0.0), '.3f')}</td><td>{_fmt(bbq.get('accuracy_disambiguated', 0.0), '.3f')}</td>"
            f"<td>{_fmt(bbq.get('bias_score_disambig', 0.0), '.3f')}</td><td>{_fmt(bbq.get('bias_score_ambig', 0.0), '.3f')}</td>"
            f"<td>{_fmt(bipia.get('asr', 0.0), '.3f')}</td><td>{_fmt(bipia.get('robustness', 0.0), '.2f')}</td>"
            f"<td><ul>{per_task}</ul></td><td><ul>{dimensions}</ul></td></tr>"
        )
    body = "\n".join(rows)
    path.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>PJ-LLM report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1300px;margin:40px auto;color:#182230}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd3dc;padding:8px;text-align:left}"
        "th{background:#eef2f7}li{margin:3px 0}</style>"
        "<h1>PJ-LLM Trustworthiness Evaluation</h1>"
        "<p>Official-aligned metrics: TruthfulQA MC1/MC2, HaluEval hallucination detection accuracy, BBQ accuracy (偏见歧视), BIPIA indirect-prompt-injection ASR/robustness (间接提示词注入).</p>"
        f"<p>Scores are reported as mean ± std across {n_seeds} random seed(s).</p>"
        f"<table><thead><tr><th>Model</th><th>Overall</th><th>MC1</th><th>MC2</th><th>HaluEval acc</th><th>BBQ acc</th><th>BBQ acc歧义</th><th>BBQ acc消歧</th><th>BBQ S_DIS</th><th>BBQ S_AMB</th><th>BIPIA ASR</th><th>BIPIA鲁棒性</th><th>各任务 ASR</th><th>各维度分数</th></tr></thead><tbody>{body}</tbody></table>"
        "</html>",
        encoding="utf-8",
    )

