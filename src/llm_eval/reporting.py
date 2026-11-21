from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any

from .config import EvaluationConfig
from .evaluator import Result
from .metrics import mean_ci, mean_of
from .scoring import aggregate_summaries, summarize


DIMENSION_KEYS = {
    "TruthfulQA": "truthfulqa",
    "HaluEval": "halueval",
    "BBQ": "bbq",
    "BIPIA": "indirect_prompt_injection",
}


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
    _write_artifacts(results, config, output_dir, seeds)


def write_reports_from_jsonl(
    results_path: str | Path,
    config: EvaluationConfig,
    output_dir: str | Path,
    seeds: list[int] | None = None,
) -> None:
    """从已有的 results.jsonl 重新生成 metrics/summary/report，不重跑模型。

    prompt 和（除 HaluEval 外的）response 不参与聚合，读入时丢弃以控制内存。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    with Path(results_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            benchmark = str(payload.get("metadata", {}).get("benchmark", ""))
            payload["prompt"] = ""
            if benchmark != "HaluEval":
                payload["response"] = ""
            score = payload.get("score")
            if score is None:
                payload["score"] = float("nan")
            results.append(Result(**payload))
    if not results:
        raise ValueError(f"No results found in {results_path}")
    _write_artifacts(results, config, output_dir, seeds)


def _write_artifacts(
    results: list[Result],
    config: EvaluationConfig,
    output_dir: Path,
    seeds: list[int] | None,
) -> None:
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

    case_stats = _case_stats(results)

    # metrics.csv：先按唯一 case 聚合（同一 case 的多个 seed 取均值），再算 95% CI。
    # 这样 n 是真实题数，不会因为复制 seed 把 CI 缩小 sqrt(seed 数) 倍。
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_id", "dimension", "n", "mean", "ci_low", "ci_high"])
        writer.writeheader()
        case_scores: dict[tuple[str, str], dict[str, list[float]]] = {}
        for result in results:
            score = result.score
            if isinstance(score, float) and math.isnan(score):
                continue
            case_scores.setdefault((result.model_id, result.dimension), {}).setdefault(
                result.case_id, []
            ).append(score)
        by_key = {
            key: [sum(scores) / len(scores) for scores in cases.values()]
            for key, cases in case_scores.items()
        }
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
    _write_html(summary, output_dir / "report.html", n_seeds=len(seed_values), case_stats=case_stats)


def _case_stats(results: list[Result]) -> dict[tuple[str, str], tuple[int, float, float, float]]:
    """按 (模型, benchmark) 统计唯一 case 的均值与 95% CI。

    同一 case 在多个 seed 下的分数先取平均，n 是唯一 case 数而不是「case × seed」。
    """
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for result in results:
        score = result.score
        if isinstance(score, float) and math.isnan(score):
            continue
        if score is None:
            continue
        benchmark = str(result.metadata.get("benchmark", ""))
        key = DIMENSION_KEYS.get(benchmark, result.dimension)
        grouped.setdefault((result.model_id, key), {}).setdefault(result.case_id, []).append(float(score))
    stats: dict[tuple[str, str], tuple[int, float, float, float]] = {}
    for model_key, cases in grouped.items():
        values = [sum(scores) / len(scores) for scores in cases.values()]
        average, low, high = mean_ci(values)
        stats[model_key] = (len(values), average * 100, low * 100, high * 100)
    return stats


def _fmt(value: Any, fmt: str = ".2f") -> str:
    """把聚合后的数值渲染为 ``mean ± std``（无 std 信息时只显示数值）。"""
    if value is None:
        return "N/A"
    if isinstance(value, dict) and "mean" in value:
        return f"{value['mean']:{fmt}} ± {value['std']:{fmt}}"
    return format(value, fmt)


def _num(value: Any) -> float:
    """取聚合值里的 mean；兼容单 seed 时被包成 {'mean','std'} 的数值。"""
    return float(mean_of(value))


def _fmt_mean(value: Any, fmt: str = ".2f") -> str:
    """表格只展示均值；不确定性统一放在各维度分数的 95% CI 列里。"""
    if value is None:
        return "N/A"
    return format(_num(value), fmt)


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
        values = [_num(item["scores"].get(dimension, 0.0)) for dimension in dimensions]
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


def _dimension_with_ci(key: str, value: Any, stats: dict[tuple[str, str], tuple[int, float, float, float]], model_id: str) -> str:
    mean_text = _fmt_mean(value, ".2f")
    seed_std = float(value.get("std", 0.0)) if isinstance(value, dict) else 0.0
    std_note = f", seed std {seed_std:.2f}" if seed_std > 1e-9 else ""
    info = stats.get((model_id, key))
    if info is None:
        return mean_text
    n, _, low, high = info
    return f"{mean_text} [95% CI {low:.2f}, {high:.2f}] (n={n:,}{std_note})"


def _confusion_text(item: dict[str, Any]) -> str:
    conf = item.get("halueval_confusion") or {}

    def field(name: str) -> float:
        value = conf.get(name, 0.0)
        return _num(value) if isinstance(value, dict) else float(value)

    if not field("n"):
        return "N/A"
    return (
        f"TP {field('tp'):.0f} / FP {field('fp'):.0f} / TN {field('tn'):.0f} / FN {field('fn'):.0f}"
        f"；预测 Yes 率 {field('yes_rate') * 100:.1f}%，无法解析 {field('invalid'):.0f}"
        f"；多数类基线 {field('majority_baseline') * 100:.1f}%"
    )


def _write_html(
    summary: list[dict[str, Any]],
    path: Path,
    n_seeds: int = 1,
    case_stats: dict[tuple[str, str], tuple[int, float, float, float]] | None = None,
) -> None:
    score_labels = _score_labels()
    case_stats = case_stats or {}
    rows = []
    for item in summary:
        bbq = item.get("bbq", {})
        bipia = item.get("bipia", {})
        judge_failed = mean_of(bipia.get("judge_failed", 0))
        per_task = "".join(
            f"<li>{task}: ASR {_fmt_mean(info.get('asr', 0.0), '.3f')} (n={info.get('n', 0)})</li>"
            for task, info in bipia.get("per_task", {}).items()
        )
        if judge_failed:
            per_task = f"<li>裁判失败: {judge_failed:.0f} 样本</li>" + per_task
        dimensions = "".join(
            f"<li>{score_labels.get(key, key)}: {_dimension_with_ci(key, value, case_stats, item['model_id'])}</li>"
            for key, value in item["scores"].items()
        )
        rows.append(
            f"<tr><td>{html.escape(item['model_id'])}</td><td>{_fmt_mean(item['overall_score'], '.2f')}</td>"
            f"<td>{_fmt_mean(item['mc1_accuracy'], '.2f')}</td><td>{_fmt_mean(item['mc2_accuracy'], '.2f')}</td>"
            f"<td>{_fmt_mean(item['halueval_accuracy'], '.2f')}</td>"
            f"<td>{_fmt_mean(bbq.get('accuracy', 0.0), '.3f')}</td><td>{_fmt_mean(bbq.get('accuracy_ambiguous', 0.0), '.3f')}</td><td>{_fmt_mean(bbq.get('accuracy_disambiguated', 0.0), '.3f')}</td>"
            f"<td>{_fmt_mean(bbq.get('bias_score_disambig', 0.0), '.3f')}</td><td>{_fmt_mean(bbq.get('bias_score_ambig', 0.0), '.3f')}</td>"
            f"<td>{_fmt_mean(bipia.get('asr', 0.0), '.3f')}</td><td>{_fmt_mean(bipia.get('robustness', 0.0), '.2f')}</td>"
            f"<td><ul>{per_task}</ul></td><td><ul>{dimensions}</ul></td>"
            f"<td>{_confusion_text(item)}</td></tr>"
        )
    body = "\n".join(rows)
    seed_note = (
        f"当前报告只含 {n_seeds} 个随机种子（seed 42）。"
        if n_seeds == 1
        else f"跨 {n_seeds} 个随机种子聚合；std 仅反映 seed 间波动。"
    )
    path.write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>PJ-LLM report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1500px;margin:40px auto;color:#182230}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd3dc;padding:8px;text-align:left}"
        "th{background:#eef2f7}li{margin:3px 0}.note{color:#4a5568;font-size:14px}</style>"
        "<h1>PJ-LLM Trustworthiness Evaluation</h1>"
        "<p><b>评测口径：</b>"
        "TruthfulQA MC1/MC2 按官方 lm-eval 口径（MC1 = P(best_answer) &gt; max P(错误候选)，对应只含一个正确项的 mc1_targets；MC2 为正确答案的总概率质量）；"
        "HaluEval 为官方 QA 幻觉检测准确率；"
        "BBQ 按官方 lm-eval <b>12 选项</b> logprob 协议（2 个具名答案 + 10 个 unknown 变体，歧义题的 target 为全部 unknown 变体）；"
        "BIPIA 按官方评测器（ModelEval 裁判使用 DeepSeek）。"
        "</p>"
        f"<p class='note'>{seed_note}"
        " 括号内 95% CI 按<b>唯一 case</b> 计算，n 为真实题数（不再把同一题的多个 seed 副本当作独立样本）。"
        " TruthfulQA/BBQ/HaluEval 在 temperature=0 下跨 seed 逐条相同，std 为 0 属预期；"
        " BIPIA 每个 seed 按 sample_ratio 抽取不同子集，跨 seed 的 n 是唯一 case 的并集，"
        "各级 ASR 的 n 仍按单个 seed 展示。"
        " BBQ 的「偏见答案」判定沿用 nyu-mll 论文 R 脚本口径，因此 S_DIS/S_AMB 不与 lm-eval 数值直接比较。"
        "</p>"
        "<p class='note'><b>解读限制：</b>Overall 为四个维度分数的等权平均；"
        "BIPIA 鲁棒性 = 1 − ASR，越弱/越不遵循指令的模型可能因此得高分，"
        "不应把该维度单独当作“更安全”的结论。"
        " BBQ 在 zero-shot 12 选 1 logprob 协议下模型很少选择 unknown，歧义准确率应视为协议相关指标，而非行为层面的偏见结论。</p>"
        f"<table><thead><tr><th>Model</th><th>Overall</th><th>MC1</th><th>MC2</th><th>HaluEval acc</th><th>BBQ acc</th><th>BBQ acc歧义</th><th>BBQ acc消歧</th><th>BBQ S_DIS</th><th>BBQ S_AMB</th><th>BIPIA ASR</th><th>BIPIA鲁棒性</th><th>各任务 ASR</th><th>各维度分数 (95% CI)</th><th>HaluEval 混淆矩阵</th></tr></thead><tbody>{body}</tbody></table>"
        "</html>",
        encoding="utf-8",
    )
