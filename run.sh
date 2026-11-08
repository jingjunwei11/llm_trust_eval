#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 设备 / 环境相关参数
# 在不同机器上运行前，按需修改下面的配置。
# ---------------------------------------------------------------------------

# conda 初始化脚本的候选位置。
CONDA_INIT_PATHS=(
  "/data/jingjunwei/.conda/etc/profile.d/conda.sh"
)

# 使用的 GPU 编号；多卡机器上可改成空闲的卡（例如 3），可用 nvidia-smi 查看各卡显存。
# 默认用两张卡（0,1），配合 device_map="auto" 把裁判 14B 和被测 3B 分到不同卡。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Hugging Face Hub 端点；直连被阻断的网络可改用镜像。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Xet 传输后端依赖未被镜像代理的 cas-server，默认关闭改用普通 HTTP 下载。
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# CUDA 版 PyTorch 的 wheel 源（依次尝试）。
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124 https://download.pytorch.org/whl/cu121 https://download.pytorch.org/whl/cu118}"

# ---------------------------------------------------------------------------
# 项目路径与运行参数
# ---------------------------------------------------------------------------

ENV_NAME="eie-eva"
CONFIG_PATH="${PJ_LLM_CONFIG:-configs/formal.yaml}"


CONFIG_SEED="$(sed -nE 's/^[[:space:]]*seed:[[:space:]]*([0-9]+).*/\1/p' "$CONFIG_PATH" | head -n1)"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${CONFIG_SEED:-0}}"

MODEL_ROOT="${PJ_LLM_MODEL_ROOT:-$SCRIPT_DIR/models}"
DATA_ROOT="${PJ_LLM_DATA_ROOT:-$SCRIPT_DIR/data}"
LOG_ROOT="${PJ_LLM_LOG_ROOT:-$SCRIPT_DIR/logs}"
RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="${PJ_LLM_OUTPUT_ROOT:-$SCRIPT_DIR/outputs/formal-$RUN_TIMESTAMP}"

# 完整终端输出（stdout + stderr）同步写入运行日志。
mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/run-$RUN_TIMESTAMP.log"
exec > >(tee -a "$RUN_LOG") 2>&1

if ! command -v conda >/dev/null 2>&1; then
  for conda_sh in "${CONDA_INIT_PATHS[@]}"; do
    if [[ -f "$conda_sh" ]]; then

      source "$conda_sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "未检测到 conda，请先安装 Miniconda 或 Anaconda。" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"

source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[conda] using existing environment: $ENV_NAME"
else
  echo "[conda] creating environment: $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.11 pip
fi

if ! conda run -n "$ENV_NAME" python -c 'import yaml, matplotlib, torch, transformers, accelerate, datasets, huggingface_hub, tqdm, thefuzz, langdetect, emoji, requests' >/dev/null 2>&1; then
  echo "[conda] installing project dependencies"
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r requirements.txt
else
  echo "[conda] project dependencies are available"
fi

mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT" \
  "$DATA_ROOT/hf-cache/datasets" "$DATA_ROOT/hf-cache/transformers"
export HF_HOME="$DATA_ROOT/hf-cache"
export HF_DATASETS_CACHE="$DATA_ROOT/hf-cache/datasets"
export TRANSFORMERS_CACHE="$DATA_ROOT/hf-cache/transformers"
export PYTHONUNBUFFERED=1
export PJ_LLM_MODEL_ROOT="$MODEL_ROOT"
export PJ_LLM_DATA_ROOT="$DATA_ROOT"

install_cuda_torch() {
  for index in $PYTORCH_CUDA_INDEX_URL; do
    echo "[gpu] installing CUDA-enabled PyTorch from $index"
    if conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade torch --index-url "$index"; then
      if conda run -n "$ENV_NAME" python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
        echo "[gpu] CUDA-enabled PyTorch is ready"
        return 0
      fi
    fi
  done
  return 1
}

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "[gpu] NVIDIA GPU detected; model loading will use CUDA automatically"
  if conda run -n "$ENV_NAME" python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "[gpu] PyTorch CUDA is available"
  else
    if ! install_cuda_torch; then
      echo "[gpu] unable to install a CUDA-enabled PyTorch build" >&2
      exit 1
    fi
  fi
else
  echo "[gpu] NVIDIA GPU not detected; the run will use CPU" >&2
fi

# ---------------------------------------------------------------------------
# BIPIA 间接提示词注入数据：检查 data/benchmarks 下三个任务数据是否存在，缺失则下载。
# ---------------------------------------------------------------------------
BIPIA_DATA_FILES=(
  bipia_table.jsonl
  bipia_email.jsonl
  bipia_code.jsonl
  bipia_text_attack_test.json
  bipia_code_attack_test.json
)

download_bipia_repo() {
  local dest="$1"
  local tmp tar_url ok=0

  # 优先 git clone；github.com 直连被阻断时退回 codeload 的 tarball。
  if command -v git >/dev/null 2>&1 && git clone --depth 1 "https://github.com/microsoft/BIPIA.git" "$dest" >/dev/null 2>&1; then
    echo "[bipia] cloned BIPIA repository"
    return 0
  fi

  tar_url="https://codeload.github.com/microsoft/BIPIA/tar.gz/refs/heads/main"
  tmp="$(mktemp -d)"
  if curl -fsSL "$tar_url" -o "$tmp/bipia.tar.gz" 2>/dev/null; then
    ok=1
  elif command -v wget >/dev/null 2>&1 && wget -q "$tar_url" -O "$tmp/bipia.tar.gz" 2>/dev/null; then
    ok=1
  fi
  if [[ "$ok" -ne 1 ]]; then
    rm -rf "$tmp"
    echo "[bipia] failed to download BIPIA repository (git clone + codeload tarball both failed)" >&2
    return 1
  fi

  rm -rf "$dest"
  tar -xzf "$tmp/bipia.tar.gz" -C "$tmp"
  mv "$tmp"/BIPIA-* "$dest"
  rm -rf "$tmp"
  echo "[bipia] downloaded BIPIA repository tarball"
  return 0
}

ensure_bipia_data() {
  local bench_dir="$DATA_ROOT/benchmarks"
  local repo="$SCRIPT_DIR/BIPIA"
  local missing=0 f

  for f in "${BIPIA_DATA_FILES[@]}"; do
    if [[ ! -s "$bench_dir/$f" ]]; then
      missing=1
      break
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    echo "[bipia] data present: $bench_dir"
    return 0
  fi

  if [[ ! -d "$repo/benchmark" ]]; then
    echo "[bipia] BIPIA data missing, downloading repository"
    download_bipia_repo "$repo" || return 1
  fi

  mkdir -p "$bench_dir"
  cp "$repo/benchmark/table/test.jsonl"      "$bench_dir/bipia_table.jsonl"
  cp "$repo/benchmark/email/test.jsonl"      "$bench_dir/bipia_email.jsonl"
  cp "$repo/benchmark/code/test.jsonl"       "$bench_dir/bipia_code.jsonl"
  cp "$repo/benchmark/text_attack_test.json" "$bench_dir/bipia_text_attack_test.json"
  cp "$repo/benchmark/code_attack_test.json" "$bench_dir/bipia_code_attack_test.json"
  echo "[bipia] data copied to $bench_dir"
}

echo "[bipia] checking indirect prompt injection data"
ensure_bipia_data

echo "[assets] checking models and benchmark data"
conda run --no-capture-output -n "$ENV_NAME" python scripts/prepare_assets.py \
  --config "$CONFIG_PATH" \
  --root "$SCRIPT_DIR" \
  --model-root "$MODEL_ROOT" \
  --data-root "$DATA_ROOT"

echo "[eval] starting formal evaluation"
conda run --no-capture-output -n "$ENV_NAME" python main.py evaluate \
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_ROOT"

echo "[done] reports: $OUTPUT_ROOT"
echo "[done] log: $LOG_ROOT"
