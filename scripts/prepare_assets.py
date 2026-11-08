from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_eval.config import load_config
from src.llm_eval.datasets import _load_hf_rows


def _rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _model_ready(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    # 必须存在实际权重文件（单文件或分片），索引文件本身不算权重。
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def prepare_models(config, root: Path, model_root: Path | None = None) -> None:
    from huggingface_hub import snapshot_download

    specs = list(config.models)
    for spec in specs:
        if not spec.local_dir:
            raise ValueError(f"Model {spec.model_id} must define local_dir")
        configured = Path(spec.local_dir)
        target = (model_root / Path(*configured.parts[1:])) if model_root and configured.parts and configured.parts[0] == "models" else _rooted(root, spec.local_dir)
        if _model_ready(target):
            print(f"[skip] model exists: {spec.model_id} -> {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"[download] model: {spec.model_id} -> {target}")
        snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        if not _model_ready(target):
            raise RuntimeError(f"Model download did not produce a usable checkpoint: {target}")


def prepare_datasets(config, root: Path, data_root: Path | None = None) -> None:
    for source in config.dataset_sources:
        if source.get("type") == "bipia":
            # BIPIA 数据的检查与下载由 run.sh 的 ensure_bipia_data 处理。
            continue
        prepared = source.get("prepared_path")
        if not prepared:
            raise ValueError(f"Dataset {source.get('id')} must define prepared_path")
        configured = Path(prepared)
        target = (data_root / Path(*configured.parts[1:])) if data_root and configured.parts and configured.parts[0] == "data" else _rooted(root, prepared)
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] dataset exists: {source.get('id')} -> {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] dataset: {source.get('id')} -> {target}")
        rows = _load_hf_rows({**source, "prepared_path": None})
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if not target.exists() or target.stat().st_size == 0:
            raise RuntimeError(f"Dataset preparation produced no rows: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download missing PJ-LLM models and benchmark data")
    parser.add_argument("--config", default="configs/formal.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-root")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = load_config(_rooted(root, args.config))
    model_root = Path(args.model_root).resolve() if args.model_root else None
    data_root = Path(args.data_root).resolve() if args.data_root else None
    prepare_models(config, root, model_root)
    prepare_datasets(config, root, data_root)
    print("[ready] all configured models and datasets are available")


if __name__ == "__main__":
    main()

