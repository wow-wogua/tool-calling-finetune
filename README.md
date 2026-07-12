# Tool Calling Finetune — 工具调用小模型微调

> 用 LoRA 微调 Qwen3，使其学会根据用户查询选择正确的工具和参数。微调后的模型可接入 [viral-video-agent](https://github.com/wow-wogua/viral-video-agent) 的 LLM 网关，作为 Researcher 工具选择的本地 A/B 路径。

## 项目背景

[viral-video-agent](https://github.com/wow-wogua/viral-video-agent) 是一个多智能体爆款视频分析系统，其中 Researcher Agent 通过 LLM 输出 JSON 来选择工具（search_videos / rag_search / get_transcript / get_trend_data）。

项目2当前主链路使用 MiMo API（免费但存在网络和排队延迟），且 LLM 输出需要防御性解析。Researcher 的任务相对窄、工具集合固定、输出结构化，因此适合作为“小模型工具选择微调”的验证对象。这个项目验证三件事：
1. **分布内格式学习**：内置用例上工具名和参数格式有所改善
2. **本地接入能力**：模型可导出为 OpenAI 兼容服务并接入 LLM 网关
3. **泛化边界**：hard eval / holdout 没有提升，说明当前数据不足以证明生产替代价值

需要注意：本项目没有完成可信的成本或速度 A/B 基准。MiMo 当前免费，因此不能声称“成本降低 5-8 倍”；本地推理是否更快也取决于硬件、量化和并发。当前只把微调模型作为 Researcher 的可选 A/B 路径，而不是生产替代。

## 技术栈

| 组件 | 技术 |
|------|------|
| 基座模型 | Qwen3-4B |
| 微调方法 | LoRA (Low-Rank Adaptation) |
| 训练框架 | LLaMA Factory |
| 数据生成 | 本地模板化生成 + 人工构造 |
| 评测范式 | BFCL 风格自建工具调用评测（非官方榜单） |
| 部署 | FastAPI（OpenAI 兼容 API） |

当前 `lora_target=all` 的 adapter 实际包含约 33.0M 参数，约占 4B 基座的 0.83%，文件约 126 MiB。它不是“0.1% / 30MB”，低秩适配也不代表不会过拟合。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或手动安装
pip install transformers peft torch datasets accelerate fastapi uvicorn
pip install llamafactory  # LLaMA Factory
```

### 2. 生成训练数据

```bash
# 生成第一版程序化模板数据（约 240 条去重样例）
python scripts/generate_data.py

# 在第一版基础上补充 rag_search/none 边界样例，生成 generated_v2.json
python scripts/fix_rag_data.py
```

当前仓库已经包含生成后的数据文件，可直接从第 3 步开始。`generated_v2.json` 是合并文件，`prepare_dataset.py` 会和 `handcrafted.json` 一起加载并按 instruction 去重。

如果要复现实验后的 hard eval 补强，不覆盖旧训练集，使用 v2 / v3 数据入口：

```bash
python scripts/prepare_dataset_v2.py
python scripts/prepare_dpo_data_v2.py
python scripts/prepare_dataset_v3.py
python scripts/prepare_dpo_data_v3.py
```

v2 / v3 会额外加载边界样例与 DPO hard negative，输出独立的 `train_v*/eval_v*` 与 `dpo_train_v*/dpo_eval_v*`。这些样例和 `data/eval/hard_cases.json`、`data/eval/hard_holdout_v3.json` 没有精确重合，避免把评测原题塞回训练集。

### 3. 准备数据集

```bash
python scripts/prepare_dataset.py
```

### 4. LoRA 微调

```bash
# 本地训练（需要 GPU）
llamafactory-cli train configs/qwen3_lora.yaml

# hard eval 补强版，不覆盖旧输出目录
llamafactory-cli train configs/qwen3_lora_v2.yaml
llamafactory-cli train configs/qwen3_dpo_v2.yaml
llamafactory-cli train configs/qwen3_lora_v3.yaml
llamafactory-cli train configs/qwen3_dpo_v3.yaml

# 或用 Colab（免费 GPU）
# 打开 Colab → 新建 notebook → 运行上面的命令
```

如果 `llamafactory-cli` 在 Windows 上静默退出，可使用本仓库的 Python fallback：

```bash
python scripts/train.py configs/qwen3_lora_v2.yaml
python scripts/train_dpo.py configs/qwen3_dpo_v2.yaml
python scripts/train.py configs/qwen3_lora_v3.yaml
python scripts/train_dpo.py configs/qwen3_dpo_v3.yaml
```

### 5. 评测

```bash
# 评测基座模型（内置 50 条 BFCL 风格用例）
python scripts/evaluate.py --model Qwen/Qwen3-4B

# 评测当前仓库已有的 merged 微调模型
python scripts/evaluate.py --model outputs/qwen3_dpo_tool_calling_merged

# 如果只有 LoRA adapter，--model 传基座模型，--adapter 传 adapter 目录
python scripts/evaluate.py --model Qwen/Qwen3-4B --adapter outputs/qwen3_lora_tool_calling

# 独立 hard eval（44 条，不和 raw instruction 精确重合）
python scripts/evaluate.py --cases data/eval/hard_cases.json --model Qwen/Qwen3-4B --output results/hard_base.json
python scripts/evaluate.py --cases data/eval/hard_cases.json --model outputs/qwen3_dpo_tool_calling_merged --output results/hard_dpo_merged.json
python scripts/evaluate.py --cases data/eval/hard_cases.json --model outputs/qwen3_dpo_tool_calling_merged_v2 --output results/hard_dpo_merged_v2.json
python scripts/evaluate.py --cases data/eval/hard_cases.json --model outputs/qwen3_dpo_tool_calling_merged_v3 --output results/hard_dpo_merged_v3.json

# v3 独立 holdout（20 条）
python scripts/evaluate.py --cases data/eval/hard_holdout_v3.json --model Qwen/Qwen3-4B --output results/hard_holdout_base.json
python scripts/evaluate.py --cases data/eval/hard_holdout_v3.json --model outputs/qwen3_dpo_tool_calling_merged_v3 --output results/hard_holdout_dpo_merged_v3.json

# 默认运行5组：Base三种Prompt + direct adapter两种Prompt
python scripts/fair_experiment.py

# 如需复核历史4-bit merged产物，显式加入；merged不再作为默认结论依据
python scripts/fair_experiment.py --include-merged

# 只复跑direct adapter，避免重复基座推理
python scripts/fair_experiment.py --adapter-only

# 只更新参数归一化评分，不重新做模型推理
python scripts/rescore_experiment.py results/fair_experiment_YYYYMMDD_HHMMSS.json
```

`researcher_prompt.py` 是评测与本地服务共用的完整 Prompt 入口。当前评分会归一化平台别名、平台列表顺序、可省略默认参数和等价检索词，并单列 JSON 有效率、延迟与“臆造视频 ID/URL”安全指标。QLoRA 的主对照直接加载 `base + adapter`；历史4-bit merged产物只作诊断，避免量化merge路径掩盖adapter效果。

### 6. 导出模型

```bash
python scripts/export_model.py \
    --base-model Qwen/Qwen3-4B \
    --adapter outputs/qwen3_dpo_tool_calling \
    --output outputs/qwen3_dpo_tool_calling_merged

# v2 补强版
python scripts/export_model.py \
    --base-model Qwen/Qwen3-4B \
    --adapter outputs/qwen3_dpo_tool_calling_v2 \
    --output outputs/qwen3_dpo_tool_calling_merged_v2

# v3 补强版
python scripts/export_model.py \
    --base-model Qwen/Qwen3-4B \
    --adapter outputs/qwen3_dpo_tool_calling_v3 \
    --output outputs/qwen3_dpo_tool_calling_merged_v3
```

### 7. 接入 viral-video-agent

```python
# 在 viral-video-agent 中注册微调后的模型
model_registry.switch_to_finetuned(
    "researcher",
    model="qwen3-tool-calling",
    base_url="http://localhost:8002/v1",  # FastAPI 部署地址（serve_model.py）
    api_key="not-needed"
)
```

## 数据格式

### 数据来源

| 来源 | 当前文件 | 说明 |
|------|----------|------|
| 手写高质量样例 | `data/raw/handcrafted.json` | 30 条，覆盖典型工具调用格式 |
| 程序化模板样例 | `data/raw/generated.json` | 约 240 条，字符串模板 + 随机组合 |
| 边界补充合并集 | `data/raw/generated_v2.json` | 323 条 raw 记录，包含第一版模板数据、手写样例和 rag_search/none 边界补充 |
| 训练输入 | `scripts/prepare_dataset.py` 输出 | 加载 `handcrafted.json` + `generated_v2.json` 后按 instruction 去重，当前为 310 条有效样例，切分为 train 246 / eval 64 |
| hard eval 补强 SFT 样例 | `data/raw/boundary_v2.json` | 60 条相似但不重复的边界表达，覆盖 `rag_search`/`none`/`search_videos` 易混场景 |
| hard eval 补强 DPO 样例 | `data/raw/dpo_boundary_v2.json` | 48 条 chosen/rejected 偏好对，重点纠正 `rag_search` vs `none` 与 `search_videos` vs `rag_search` |
| v2 训练输入 | `scripts/prepare_dataset_v2.py` 输出 | 加载旧 raw + `boundary_v2.json` 后按 instruction 去重，当前为 370 条有效样例，切分为 train 294 / eval 76 |
| v3 边界 SFT 样例 | `data/raw/boundary_v3.json` | 40 条更聚焦的边界表达，补充 hard eval 暴露的自然语言说法 |
| v3 边界 DPO 样例 | `data/raw/dpo_boundary_v3.json` | 24 条 chosen/rejected 偏好对，继续强化 `rag_search` vs `none` |
| v3 训练输入 | `scripts/prepare_dataset_v3.py` 输出 | 加载旧 raw + v2/v3 边界样例后按 instruction 去重，当前为 train 327 / eval 83；DPO 为 train 21 / eval 3 |
| 独立 hard eval | `data/eval/hard_cases.json` | 44 条人工构造评测用例，和 raw instruction 无精确重合 |
| 独立 holdout | `data/eval/hard_holdout_v3.json` | 20 条新构造评测用例，不参与训练，用于复核 v3 是否只对 hard eval 过拟合 |

这里没有使用 MiMo API 或其他大模型批量生成训练数据。这样做的优点是标签可控、工具分布可控；缺点是真实用户表达仍然不足，所以补了独立 hard eval / holdout 用来检验模板分布外表现。v2 / v3 数据是 hard eval 暴露问题后的补强实验，最终结果显示：补小样本边界数据没有带来 hard eval / holdout 提升，后续应优先做 Prompt 路由规则、兜底规则或真实/半真实轨迹数据，而不是继续小样本追分。

每条训练数据包含用户查询和期望的工具调用输出：

```json
{
    "instruction": "分析B站美妆区的爆款视频",
    "input": "可用工具:\n- search_videos(keyword, platforms, limit): ...\n- rag_search(query, top_k): ...",
    "output": "{\"tool\": \"search_videos\", \"params\": {\"keyword\": \"美妆\", \"platforms\": [\"bilibili\"], \"limit\": 10}}"
}
```

### 覆盖场景

| 场景 | 工具 | 数据量 |
|------|------|--------|
| 搜索视频 | search_videos | ~80 条 |
| 知识库检索 | rag_search | ~50 条 |
| 视频转写 | get_transcript | ~40 条 |
| 趋势数据 | get_trend_data | ~40 条 |
| 无需工具 | none | ~50 条 |

## 评测结果

| 模型 | 工具准确率 | 完全准确率 | 说明 |
|------|-----------|-----------|------|
| Qwen3-4B (基座) | 88.0% | 54.0% | 50 条 BFCL 风格自建用例，`results/eval_20260630_185909.json` |
| Qwen3-4B + SFT | 92.0% | 78.0% | `results/eval_20260630_184816.json` |
| Qwen3-4B + SFT+DPO | **94.0%** | **80.0%** | `results/eval_20260630_185543.json` |
| MiMo v2.5-pro (API) | 90.0% (27/30) | - | 项目2历史 30 条 BFCL 风格用例；与本项目 50 条不可直接横比 |

上表是内置 50 条 BFCL 风格用例的历史结果。

### 独立 hard eval

| 模型 | 工具准确率 | 完全准确率 | 说明 |
|------|-----------|-----------|------|
| Qwen3-4B (基座) | 84.1% (37/44) | 52.3% (23/44) | `results/hard_base.json` |
| Qwen3-4B + SFT+DPO merged | 84.1% (37/44) | 52.3% (23/44) | `results/hard_dpo_merged.json` |
| Qwen3-4B + SFT+DPO merged v2 | 84.1% (37/44) | 52.3% (23/44) | `results/hard_dpo_merged_v2.json` |
| Qwen3-4B + SFT+DPO merged v3 | 84.1% (37/44) | 52.3% (23/44) | `results/hard_dpo_merged_v3.json` |

hard eval 和 raw instruction 无精确重合。结果显示：微调模型在内置 50 条用例上提升明显，但在 hard eval 上没有超过基座，主要错误集中在 `rag_search` vs `none` 的语义边界（方法论/规则/框架类问题被判为无需工具）。

### v3 独立 holdout

| 模型 | 工具准确率 | 完全准确率 | 说明 |
|------|-----------|-----------|------|
| Qwen3-4B (基座) | 65.0% (13/20) | 35.0% (7/20) | `results/hard_holdout_base.json` |
| Qwen3-4B + SFT+DPO merged v3 | 65.0% (13/20) | 35.0% (7/20) | `results/hard_holdout_dpo_merged_v3.json` |

### 同 Prompt 公平实验：历史 merged 对照（2026-07-11）

以下 5 组使用同一 hard44 / holdout20、4-bit 加载、`temperature=0`、`do_sample=False`、`max_new_tokens=128`。参数准确率只统计有人工参数标注的用例；Safe 指标在完全正确之外，额外惩罚输入中不存在却被模型臆造的具体视频 ID/URL。

| 模型 + Prompt | 数据集 | 工具 | 参数 | 完全 | Safe | JSON | 平均延迟 |
|---------------|--------|------|------|------|------|------|----------|
| Base + base | hard44 | 95.5% | 83.3% | 86.4% | 79.5% | 97.7% | 1764 ms |
| Base + base | holdout20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1846 ms |
| Base + strengthened | hard44 | 95.5% | 79.2% | 84.1% | 75.0% | 95.5% | 1541 ms |
| Base + strengthened | holdout20 | 80.0% | 60.0% | 70.0% | 70.0% | 100.0% | 1373 ms |
| Base + rules | hard44 | 100.0% | 75.0% | 86.4% | 77.3% | 100.0% | 1688 ms |
| Base + rules | holdout20 | 80.0% | 60.0% | 70.0% | 70.0% | 100.0% | 1626 ms |
| SFT+DPO v3 merged(历史) + base | hard44 | 95.5% | 83.3% | 86.4% | 79.5% | 97.7% | 1791 ms |
| SFT+DPO v3 merged(历史) + base | holdout20 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1878 ms |
| SFT+DPO v3 merged(历史) + rules | hard44 | 100.0% | 75.0% | 86.4% | 77.3% | 100.0% | 1896 ms |
| SFT+DPO v3 merged(历史) + rules | holdout20 | 80.0% | 60.0% | 70.0% | 70.0% | 100.0% | 1894 ms |

结果文件：`results/fair_experiment_20260711_152552_rescored.json`。

这轮 merged 结果与 Base 完全相同。后续 direct adapter 复核证明 adapter 实际会改变输出，因此该 merged 路径不能用于判断“微调完全无效”；已有4-bit merged产物只保留为历史诊断。

### Direct adapter 复核（当前主口径，2026-07-12）

| 模型 + Prompt | 数据集 | 工具 | 参数 | 完全 | Safe | JSON | 平均延迟 |
|---------------|--------|------|------|------|------|------|----------|
| SFT+DPO v3 adapter + base | hard44 | 88.6% | 79.2% | 77.3% | 68.2% | 90.9% | 2654 ms |
| SFT+DPO v3 adapter + base | holdout20 | 95.0% | 93.3% | 90.0% | 90.0% | 100.0% | 2554 ms |
| SFT+DPO v3 adapter + rules | hard44 | 97.7% | 87.5% | 90.9% | 75.0% | 100.0% | 2819 ms |
| SFT+DPO v3 adapter + rules | holdout20 | 90.0% | 66.7% | 75.0% | 75.0% | 100.0% | 2355 ms |

结果文件：`results/adapter_direct_experiment_20260712.json`。

Direct adapter 的结论更具体：默认 base Prompt 下，adapter 在 hard44 和 holdout20 都低于 Base；rules Prompt 下部分完全准确率提高，但 hard 安全准确率下降，而且整体仍没有超过最稳的 `Base + base`。因此当前不能宣称微调带来自然表达泛化提升或生产替代价值。现有 FastAPI 服务默认直接加载4-bit基座 + adapter；如果必须导出 merged 模型，应使用修正后的 `export_model.py` 先以 BF16 合并，再单独量化和复测。

### 实验结论

当前小样本 LoRA 证明了分布内格式学习和已覆盖场景有效：内置 50 条 BFCL 风格用例从 88% / 54% 提升到 94% / 80%。但历史 hard/holdout、direct adapter 公平实验都没有显示稳定超过最强基座配置，不能证明自然表达边界泛化提升。旧4-bit merged结果与Base完全相同属于额外的导出路径风险，不再作为主结论依据。

这不等于 Researcher 不适合微调。Researcher 仍然是合理的微调验证对象，因为任务窄、工具固定、输出结构化、评测清楚；只是当前“小规模模板数据 + LoRA”不足以解决 `rag_search` vs `none` 的自然语言边界。下一步如果要提升，应优先补 Prompt 路由规则、兜底规则，或沉淀真实/半真实任务轨迹，再决定是否扩大训练数据继续微调。

### 后续增强泛化能力

1. **Prompt 路由规则**：已完成 hard44/holdout20 对照；hard 提升、holdout 下降，因此保留为可切换实验版本，不设为默认。
2. **代码侧兜底规则**：对明显的方法论/规则类 query 加轻量关键词或意图分类兜底，避免模型直接返回 `none`。
3. **半真实任务轨迹**：由人工按真实使用方式写任务表达，覆盖短句、口语、含糊表达和上下文省略，而不是继续堆模板句。
4. **真实日志清洗**：未来项目2如果有 TraceTracker/用户日志，把成功工具调用轨迹清洗成训练样例。
5. **扩大后再微调**：只有当半真实/真实数据足够多，并且 holdout 能稳定区分改动收益时，再继续 SFT/DPO。

## 目录结构

```
tool-calling-finetune/
├── data/
│   ├── raw/
│   │   ├── handcrafted.json    # 手写高质量样例（30条）
│   │   ├── generated.json      # 第一版模板数据
│   │   ├── generated_v2.json   # 第二版模板数据（323条，prepare_dataset 优先使用）
│   │   ├── boundary_v2.json    # hard eval 补强 SFT 样例（60条）
│   │   ├── dpo_boundary_v2.json # hard eval 补强 DPO 偏好对（48条）
│   │   ├── boundary_v3.json    # v3 边界 SFT 样例（40条）
│   │   └── dpo_boundary_v3.json # v3 DPO 偏好对（24条）
│   ├── eval/
│   │   ├── hard_cases.json     # 44条独立 hard eval
│   │   └── hard_holdout_v3.json # 20条独立 holdout
│   ├── processed/
│   │   ├── train.json          # SFT 训练集（80%）
│   │   ├── eval.json           # 评估集（20%）
│   │   ├── train_v2.json       # v2 SFT 训练集
│   │   ├── eval_v2.json        # v2 SFT 评估集
│   │   ├── dpo_train_v2.json   # v2 DPO 训练集
│   │   ├── dpo_eval_v2.json    # v2 DPO 评估集
│   │   ├── train_v3.json       # v3 SFT 训练集
│   │   ├── eval_v3.json        # v3 SFT 评估集
│   │   ├── dpo_train_v3.json   # v3 DPO 训练集
│   │   └── dpo_eval_v3.json    # v3 DPO 评估集
│   └── dataset_info.json       # LLaMA Factory 配置
├── scripts/
│   ├── generate_data.py        # 数据生成脚本
│   ├── fix_rag_data.py         # rag_search/none 边界样例补充
│   ├── prepare_dataset.py      # 数据处理脚本
│   ├── prepare_dataset_v2.py   # v2 SFT 数据处理脚本
│   ├── prepare_dpo_data_v2.py  # v2 DPO 数据处理脚本
│   ├── prepare_dataset_v3.py   # v3 SFT 数据处理脚本
│   ├── prepare_dpo_data_v3.py  # v3 DPO 数据处理脚本
│   ├── researcher_prompt.py     # 训练外评测与服务共用 Prompt 入口
│   ├── evaluation_core.py       # 参数归一化、安全与延迟指标
│   ├── evaluate.py             # 评测脚本
│   ├── fair_experiment.py      # Base/SFT+DPO × Prompt 公平实验矩阵
│   ├── rescore_experiment.py   # 不重新推理的离线重评分
│   ├── serve_model.py          # OpenAI 兼容本地推理服务
│   └── export_model.py         # 模型导出脚本
├── configs/
│   ├── qwen3_lora.yaml         # LLaMA Factory 微调配置
│   ├── qwen3_dpo.yaml          # DPO 配置
│   ├── qwen3_lora_v2.yaml      # hard eval 补强版 SFT 配置
│   ├── qwen3_dpo_v2.yaml       # hard eval 补强版 DPO 配置
│   ├── qwen3_lora_v3.yaml      # v3 SFT 配置
│   └── qwen3_dpo_v3.yaml       # v3 DPO 配置
├── results/                    # 评测结果
├── README.md
├── .gitignore
└── .env.example
```

## 和 viral-video-agent 的联动

```
viral-video-agent (项目2)          tool-calling-finetune (项目3)
┌─────────────────────┐           ┌─────────────────────┐
│  Researcher Agent    │           │  数据生成             │
│  - LLM 驱动工具选择  │ ────────→ │  - 模板化批量生成     │
│  - 三层兜底解析      │           │  - 手写/边界样例      │
│  - MCP 协议调用      │           │                      │
├─────────────────────┤           ├─────────────────────┤
│  LLM 网关            │           │  LoRA 微调            │
│  - ModelRegistry     │ ←─────── │  - Qwen3-4B + LoRA   │
│  - 热切换 + A/B 测试  │           │  - LLaMA Factory     │
│  - 成本追踪          │           │                      │
├─────────────────────┤           ├─────────────────────┤
│  评测框架            │           │  评测                 │
│  - BFCL风格 (30条)   │ ←─────── │  - 自建50条工具用例   │
│  - tau-inspired (18) │           │  - 基座 vs 微调对比   │
└─────────────────────┘           └─────────────────────┘
```

## 部署与使用

### 启动微调模型 API
```cmd
cd D:\internship\tool-calling-finetune
set BASE_MODEL_PATH=C:\Users\0\.cache\modelscope\Qwen\Qwen3-4B
set FINETUNED_ADAPTER_PATH=outputs\qwen3_dpo_tool_calling_v3
python scripts\serve_model.py
```

服务默认使用 `base + direct adapter`。只有重新用 BF16 正确合并并完成复测后，才设置 `FINETUNED_MODEL_PATH` 改走 merged 模型。
API 地址: http://localhost:8002/v1

### 接入 viral-video-agent
```cmd
cd D:\internship\viral-video-agent
set USE_FINETUNED_MODEL=true
docker-compose up -d
```
只有 Researcher 使用微调模型，其他 Agent 仍用 MiMo API。服务端会识别项目2已经生成的完整 Researcher Prompt，避免重复包 Prompt；项目2与项目3对齐的是工具 schema 和 JSON 输出格式，任务/平台上下文由项目2额外注入。

### 关闭
- 微调模型 API: Ctrl+C
- viral-video-agent: `docker-compose down`

## License

MIT
