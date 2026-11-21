from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.llm_eval.config import load_config
from src.llm_eval.datasets import filter_cases_by_benchmark, load_cases
from src.llm_eval.evaluator import Evaluator
from src.llm_eval.reporting import _json_safe, write_report_bundle, write_reports_from_jsonl
from src.llm_eval.seed import set_global_seed


def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seeds(value: str | None) -> list[int] | None:
    items = _parse_list(value)
    return [int(item) for item in items] if items else None


def run(
    config_path: str,
    output_dir: str,
    benchmarks: list[str] | None = None,
    seeds_override: list[int] | None = None,
    limit: int | None = None,
    models_filter: list[str] | None = None,
) -> None:
    config = load_config(config_path)
    if models_filter:
        wanted = {model.strip() for model in models_filter if model.strip()}
        config.models = [spec for spec in config.models if spec.model_id in wanted]
        if not config.models:
            raise ValueError(f"No configured models matched: {sorted(wanted)}")
    # 用多个随机种子重复评测，报告分数以 mean ± std 表示。
    # 显式给出 experiment.seeds 时按给定列表运行，否则从 seed 起连续取 num_seeds 个。
    seeds = seeds_override or (
        config.seeds if config.seeds else [config.seed + offset for offset in range(max(1, config.num_seeds))]
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # 每完成一个 case 就追加写入，长跑被终止时已完成的样本不会全部丢失。
    partial_path = output / "results.partial.jsonl"
    all_results: list[object] = []
    with partial_path.open("w", encoding="utf-8") as handle:
        def sink(result) -> None:
            handle.write(json.dumps(_json_safe(result.to_dict()), ensure_ascii=False) + "\n")
            handle.flush()

        for seed in seeds:
            # 固定全局随机源（random/numpy/torch/sklearn），与生成用同一个种子；
            # 每个种子独立加载样本（BIPIA 的中间注入位置随种子变化）。
            set_global_seed(seed)
            cases = filter_cases_by_benchmark(load_cases(config.dataset_sources, seed=seed), benchmarks)
            if limit:
                cases = cases[:limit]
            evaluator = Evaluator(config)
            all_results.extend(evaluator.run(cases, seed=seed, on_result=sink))
    write_report_bundle(all_results, config, output, seeds=seeds)
    partial_path.unlink(missing_ok=True)
    print(f"Completed {len(all_results)} evaluations across {len(seeds)} seeds")
    print(f"Report: {output / 'report.html'}")


def merge_results(
    base_path: str,
    rerun_path: str,
    output_dir: str,
    config_path: str,
    keep_seeds: list[int] | None = None,
    replace_benchmarks: list[str] | None = None,
) -> None:
    """把局部重跑结果合并进既有全量结果。

    只保留 ``keep_seeds`` 里的种子，并让重跑覆盖指定 benchmark（默认 BBQ）
    在 base 里的旧行。重跑文件必须覆盖这些种子的全部样本，否则拒绝合并。
    """
    config = load_config(config_path)
    seeds = keep_seeds or config.seeds or [config.seed]
    keep = set(seeds)
    replaced = {name.lower() for name in (replace_benchmarks or ["BBQ"])}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged_path = output / "results.jsonl"
    kept = dropped = 0
    rerun_counts: Counter[tuple[str, str, int]] = Counter()

    with merged_path.open("w", encoding="utf-8") as out:
        def _keep(line: str) -> None:
            out.write(line if line.endswith("\n") else line + "\n")

        with Path(base_path).open("r", encoding="utf-8") as base:
            for line in base:
                if not line.strip():
                    continue
                row = json.loads(line)
                benchmark = str(row.get("metadata", {}).get("benchmark", "")).lower()
                if row.get("seed") not in keep or benchmark in replaced:
                    dropped += 1
                    continue
                _keep(line)
                kept += 1
        with Path(rerun_path).open("r", encoding="utf-8") as rerun:
            for line in rerun:
                if not line.strip():
                    continue
                row = json.loads(line)
                benchmark = str(row.get("metadata", {}).get("benchmark", "")).lower()
                if row.get("seed") not in keep or benchmark not in replaced:
                    continue
                _keep(line)
                kept += 1
                rerun_counts[(
                    row.get("model_id", ""),
                    str(row.get("metadata", {}).get("benchmark", "")),
                    int(row.get("seed", -1)),
                )] += 1

    # 完整性检查：每个模型、每个种子、每个被替换的 benchmark 都必须拿到全部预期样本。
    problems: list[str] = []
    for seed in seeds:
        expected: Counter[str] = Counter()
        for case in load_cases(config.dataset_sources, seed=seed):
            benchmark = str(case.metadata.get("benchmark", ""))
            if benchmark.lower() in replaced:
                expected[benchmark] += 1
        for model in config.models:
            for benchmark, count in expected.items():
                actual = rerun_counts.get((model.model_id, benchmark, seed), 0)
                if actual != count:
                    problems.append(f"{model.model_id} {benchmark} seed={seed}: {actual}/{count}")
    if problems:
        raise RuntimeError("rerun is incomplete, refusing to merge: " + "; ".join(problems))

    write_reports_from_jsonl(merged_path, config, output, seeds=seeds)
    print(f"Merged {kept} rows across seeds {seeds} (dropped {dropped} base rows) -> {merged_path}")
    print(f"Report: {output / 'report.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PJ-LLM unified trustworthy evaluation")
    sub = parser.add_subparsers(dest="command")

    evaluate = sub.add_parser("evaluate", help="run a YAML experiment")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--output", default="outputs/run")
    evaluate.add_argument("--benchmarks", default=None, help="逗号分隔，如 truthfulqa,bbq；默认全部")
    evaluate.add_argument("--seeds", default=None, help="逗号分隔的 seed 列表；默认用配置里的 seeds")
    evaluate.add_argument("--limit", type=int, default=None, help="只跑前 N 个 case（冒烟测试用）")
    evaluate.add_argument("--models", default=None, help="逗号分隔的 model_id；默认用配置里的全部模型")

    reports = sub.add_parser("reports", help="从已有 results.jsonl 重新生成指标与报告")
    reports.add_argument("--config", required=True)
    reports.add_argument("--results", required=True)
    reports.add_argument("--output", required=True)
    reports.add_argument("--seeds", default=None)

    merge = sub.add_parser("merge", help="把局部重跑结果合并进全量结果并重新出报告")
    merge.add_argument("--config", required=True)
    merge.add_argument("--base", required=True, help="原全量 results.jsonl")
    merge.add_argument("--rerun", required=True, help="局部重跑的 results.jsonl")
    merge.add_argument("--output", required=True)
    merge.add_argument("--keep-seeds", default=None, help="逗号分隔的种子列表；默认用配置里的 seeds")
    merge.add_argument("--keep-seed", type=int, default=None, help="单种子快捷方式（兼容旧用法）")
    merge.add_argument("--replace-benchmarks", default="BBQ")

    # Also accept: python main.py --config ... --output ...
    parser.add_argument("--config", dest="root_config")
    parser.add_argument("--output", dest="root_output", default="outputs/run")
    args = parser.parse_args()

    if args.command == "evaluate":
        run(
            args.config,
            args.output,
            benchmarks=_parse_list(args.benchmarks),
            seeds_override=_parse_seeds(args.seeds),
            limit=args.limit,
            models_filter=_parse_list(args.models),
        )
    elif args.command == "reports":
        config = load_config(args.config)
        seeds = _parse_seeds(args.seeds) or [config.seed]
        write_reports_from_jsonl(args.results, config, args.output, seeds=seeds)
    elif args.command == "merge":
        merge_results(
            args.base,
            args.rerun,
            args.output,
            args.config,
            keep_seeds=_parse_seeds(args.keep_seeds) or ([args.keep_seed] if args.keep_seed is not None else None),
            replace_benchmarks=_parse_list(args.replace_benchmarks),
        )
    elif args.root_config:
        run(args.root_config, args.root_output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
