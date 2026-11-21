# LLM Trust Eval

面向开源小型语言模型的多维度可信评测系统，覆盖幻觉、偏见歧视、间接提示词注入三个维度，由公开基准产生证据，统一 Python 工程负责加载、评测、统计、评分与报告。

## 运行方式

使用正式模型和公开基准，需下载模型权重与基准数据。

```bash
bash run.sh
```

`run.sh` 自动检查/创建 `eie-eva` conda 环境、安装依赖，并只下载缺失的模型与基准数据。

有 NVIDIA GPU 时自动用 CUDA + FP16（CUDA 版 PyTorch 缺失会按 `cu124`/`cu121`/`cu118` 顺序自动安装，可用 `PYTORCH_CUDA_INDEX_URL` 覆盖）；无 GPU 时回退 CPU。

`configs/formal.yaml` 固定全部基准，一次 `bash run.sh` 并行跑完幻觉、偏见歧视、间接提示词注入三个维度，全部基准全量运行。


输出文件包括：

- `results.jsonl`：每条样本的原始回答、分数和元数据（含 `seed` 字段区分不同随机种子）
- `metrics.csv`：各模型、各维度的均值和95% 置信区间
- `summary.json`：各维度评分、综合分（跨种子为 `mean ± std`）
- `report.html`：汇总表和模型维度评分，分数以 `mean ± std` 表示
- `score_overview.png`：评分柱状图

默认用三个常用随机种子（`experiment.seeds`，默认 `[42, 1234, 2024]`）重复评测，报告分数用均值 ± 标准差表示；未显式给出 `seeds` 时从 `seed` 起连续取 `num_seeds` 个，`num_seeds: 1` 即退回单种子运行。

`run.sh` 把完整终端输出写入 `logs/run-<YYYYmmdd-HHMMSS>.log`。可通过环境变量覆盖目录：


## 评测基准与维度

`configs/formal.yaml` 固定全部基准：TruthfulQA、HaluEval、BBQ 映射到幻觉、偏见歧视两个维度，BIPIA（Table / Email / Code QA）映射到间接提示词注入维度。所有基准数据统一存放在 `data/benchmarks/`。

### 幻觉（hallucination）

- **TruthfulQA**（817 条）：按官方 [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) MC 口径，`QA_PRIMER` 前缀 + 各候选答案对数概率。
  - 指标：MC1 准确率、MC2 准确率
- **HaluEval**（`qa` 全量）：按官方 [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval)，随机选 right / hallucinated 答案，few-shot 指令 + Yes/No 子串匹配。
  - 指标：检测准确率

### 偏见歧视（bias）

- **BBQ**（58,492 条）：按官方 [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ) 原版口径，3 选项（ans0/ans1/ans2）取对数概率 argmax。
  - 指标：总/歧义/消歧准确率（偏见歧视基准分用总准确率），偏差分 `S_DIS`、`S_AMB`，每类别偏差分

### 间接提示词注入（indirect prompt injection）

- **BIPIA**（3 个任务）：按官方 [microsoft/BIPIA](https://github.com/microsoft/BIPIA) 口径，把恶意指令注入外部内容（邮件 / 表格 / Stack Overflow 回答）的 `end` / `start` / `middle` 三个位置，考察模型在回答用户问题时是否被注入指令带偏。
  - 只使用 **Table QA**、**Email QA**、**Code QA** 三个任务（WebQA / Summarization 因版权需额外下载原始数据，不在本维度）。
  - 数据与其他基准一样存放在 `data/benchmarks/`（`bipia_table.jsonl` / `bipia_email.jsonl` / `bipia_code.jsonl` + `bipia_text_attack_test.json` / `bipia_code_attack_test.json`）。`run.sh` 会检查这三个任务的数据是否存在，缺失时自动下载 BIPIA 仓库并拷贝到位。
  - 指标：攻击成功率 ASR（越低越安全），维度分数 = (1 − ASR) × 100。
  - ASR 口径：完全按 BIPIA 官方评测器——Code QA 用 `MatchRefEval`（`thefuzz.partial_ratio > 80` 对注入的恶意代码模糊匹配）；Email/Table QA 中 Emoji 替换用 `EmojiEval`、语言翻译用 `LanguageEval`（`langdetect`）、两个固定 reference 用 `MatchRefEval`，其余文本攻击用 `ModelEval` 裁判。与官方唯一区别：裁判模型由 GPT 换成 **DeepSeek**（OpenAI 兼容 API，默认用 `deepseek-chat`，见 `configs/formal.yaml` 的 `judge` 段，`api_key` 留空时读 `DEEPSEEK_API_KEY` 环境变量）。

## 项目结构

```text
llm-trust-eval/
├─ main.py
├─ run.sh
├─ requirements.txt
├─ configs/
│  └─ formal.yaml              # 全部基准（TruthfulQA / HaluEval / BBQ / BIPIA）
├─ scripts/
│  └─ prepare_assets.py  # 下载模型与基准数据（HF）
├─ data/
│  └─ benchmarks/        # 各基准数据（truthfulqa/halueval/bbq/bipia_*.jsonl 等）
├─ BIPIA/                # 官方 microsoft/BIPIA 仓库（参考实现 + 原始数据）
│  └─ benchmark/         #   {table,email,code}/test.jsonl、{text,code}_attack_test.json
├─ src/llm_eval/
   ├─ adapters.py       # Hugging Face 模型适配器（支持 system 提示）
   ├─ bipia.py          # BIPIA 三个任务的构造与 ASR 判分
   ├─ config.py         # YAML 配置读取
   ├─ datasets.py       # 各基准的字段转换
   ├─ evaluator.py      # 各维度的评分逻辑
   ├─ metrics.py        # 统计指标与置信区间
   ├─ scoring.py        # 官方指标聚合
   └─ reporting.py      # CSV、JSON、HTML、图表输出

```

## 统一实验流程

```text
formal.yaml
  -> 下载/读取全部基准数据（HF + data/benchmarks 内 BIPIA 数据）
  -> 统一 Hugging Face 模型接口
  -> 多个随机种子运行评测（默认 3 次）
  -> 保存原始 JSONL
  -> 计算官方指标和 95% CI
  -> 计算综合可信分数
  -> 生成 HTML、CSV、JSON 和 PNG 报告
```



## 综合评分

总分 = 四个维度分数的平均：

```text
总分 = (TruthfulQA + HaluEval + 偏见歧视 + 间接提示词注入) / 4
```

各维度分均为 0-100：TruthfulQA 用 MC1 准确率、HaluEval 用检测准确率、偏见歧视（BBQ）用准确率、间接提示词注入（BIPIA）用 (1 − ASR) × 100。

报告同时保留各维度的独立分数和官方指标，不用综合分替代各维度分数。

## 参考代码仓库

| 仓库 | 参考内容 | 本项目实现 |
|---|---|---|
| [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) | `qa` preset、MC1/MC2 评分口径 | `datasets.py`、`evaluator.py` 的 `_score_mc` |
| [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval) | 幻觉检测格式、few-shot 指令、准确率 | `datasets.py`、`evaluator.py` |
| [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ) | `S_DIS`/`S_AMB`、3 选项（ans0/ans1/ans2） | `datasets.py`、`scoring.py` |
| [microsoft/BIPIA](https://github.com/microsoft/BIPIA) | Table/Email/Code QA 的注入样本构造、官方 ASR（`MatchRefEval`/`EmojiEval`/`LanguageEval`/`ModelEval`，裁判用 DeepSeek） | `bipia.py`、`evaluator.py`、`scoring.py` |
