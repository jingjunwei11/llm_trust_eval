from __future__ import annotations

import argparse
from pathlib import Path

from src.llm_eval.config import load_config
from src.llm_eval.datasets import load_cases
from src.llm_eval.evaluator import Evaluator
from src.llm_eval.reporting import write_report_bundle
from src.llm_eval.seed import set_global_seed


def run(config_path: str, output_dir: str) -> None:
    config = load_config(config_path)
    # 用多个随机种子重复评测，报告分数以 mean ± std 表示。
    # 显式给出 experiment.seeds 时按给定列表运行，否则从 seed 起连续取 num_seeds 个。
    seeds = config.seeds if config.seeds else [config.seed + offset for offset in range(max(1, config.num_seeds))]
    all_results: list[object] = []
    for seed in seeds:
        # 固定全局随机源（random/numpy/torch/sklearn），与生成用同一个种子；
        # 每个种子独立加载样本（BIPIA 的中间注入位置随种子变化）。
        set_global_seed(seed)
        cases = load_cases(config.dataset_sources, seed=seed)
        evaluator = Evaluator(config)
        all_results.extend(evaluator.run(cases, seed=seed))
    write_report_bundle(all_results, config, Path(output_dir), seeds=seeds)
    print(f"Completed {len(all_results)} evaluations across {len(seeds)} seeds")
    print(f"Report: {Path(output_dir) / 'report.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PJ-LLM unified trustworthy evaluation")
    sub = parser.add_subparsers(dest="command")

    evaluate = sub.add_parser("evaluate", help="run a YAML experiment")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--output", default="outputs/run")

    # Also accept: python main.py --config ... --output ...
    parser.add_argument("--config", dest="root_config")
    parser.add_argument("--output", dest="root_output", default="outputs/run")
    args = parser.parse_args()

    if args.command == "evaluate":
        run(args.config, args.output)
    elif args.root_config:
        run(args.root_config, args.root_output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

