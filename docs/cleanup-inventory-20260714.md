# v4 清理清单（2026-07-14）

## 边界与基线

- 仓库：`tool-calling-finetune` 仓库根目录
- 清理前仓库总占用：11,528,335,466 bytes（10.737 GiB）
- 清理前 `outputs/`：11,525,651,829 bytes（10.734 GiB）
- 外部基座：本机已有的 Qwen3-4B 缓存（通过 `BASE_MODEL_PATH` 指定），8,060,926,179 bytes（7.507 GiB）
- 外部基座不属于清理范围，不得删除或重复下载。
- v4 完成前保留所有旧 Adapter；先用 v3 Direct Adapter 完成同条件基线。

## A 类：必须保留的历史证据

- `results/legacy/v1-v3/results/`：v1-v3 内置评测、hard44、holdout20、公平实验、Direct Adapter 复核。
- `results/legacy/v1-v3/cases/`：旧 `hard_cases.json` 与 `hard_holdout_v3.json`。
- `results/legacy/v1-v3/configs/`：v1-v3 SFT/DPO 训练配置记录。
- `results/legacy/v1-v3/legacy_manifest.json`：旧 raw/processed 数据的来源、数量、字节数和 SHA-256；原中间数据已删除。
- `README.md` 中 v1-v3 的实验演进、分布内结果、hard/holdout 边界和 4-bit merge 风险。
- Git 历史；旧源码即使第二次清理后删除，也必须能从 Git 恢复。

历史关键结论：

- 内置 50 条：Base 88% tool / 54% full；SFT 92% / 78%；SFT+DPO 94% / 80%。
- 旧 hard44：Base 与历史 merged v3 均为 84.1% tool / 52.3% full。
- 旧 holdout20：Base 与历史 merged v3 均为 65.0% tool / 35.0% full。
- Direct Adapter + base prompt：hard44 77.3% full、holdout20 90.0% full。
- Direct Adapter + rules prompt：hard44 90.9% full 但 safe 75.0%；holdout20 75.0% full。
- 因此 v1-v3 不能声称自然表达泛化提升或生产替代价值。

## B 类：v4 完成前暂时保留

- `outputs/qwen3_dpo_tool_calling_v3`：旧 v3 Direct Adapter 公平基线。
- `outputs/qwen3_lora_tool_calling_v2`、`outputs/qwen3_lora_tool_calling_v3`。
- `outputs/qwen3_dpo_tool_calling_v2`、`outputs/qwen3_dpo_tool_calling_v3`。
- `scripts/evaluation_core.py`、`train.py`、`train_dpo.py`、`serve_model.py`、`export_model.py`。
- 当前评测与服务入口，直到 v4 入口完成验证。

## C 类：确认无用后可以删除

### 第一次清理

- `outputs/Qwen3-4B-base`：仅有 Hugging Face cache 标记，196 bytes，无 `config.json` 或权重。
- `outputs/qwen3_dpo_tool_calling_merged`
- `outputs/qwen3_dpo_tool_calling_merged_v2`
- `outputs/qwen3_dpo_tool_calling_merged_v3`
- `scripts/__pycache__`

三份 merged 目录各 2,833,649,555 bytes（2.639 GiB），其 `config.json` 均保留 `load_in_4bit=true` 的 bitsandbytes 量化配置。对应评测 JSON 和 README 结论已在 A 类保留，因此可删除大体积权重，仅保留诊断证据。

执行结果：已删除上述三份 merged、空基座残留和 `scripts/__pycache__`，共释放 8,501,013,985 bytes（7.917 GiB）。外部 ModelScope 基座仍完整。

### 第二次清理（v4.1 完成并验证后）

- 删除旧 v2/v3 SFT 与 DPO Adapter、v4 初始 Adapter、v4 smoke Adapter，以及最终 v4.1 中重复的 `checkpoint-47`。
- 删除已写入 legacy manifest 的 `data/raw/`、`data/processed/`。
- 删除被统一 v4 入口替代的 `prepare_dataset*`、`prepare_dpo_data*`、旧评测/重评分/Prompt、`fix_rag_data.py`、`convert_to_gguf.py` 等脚本。
- 删除服务日志、Base smoke 和 Adapter reload 临时结果；保留 OpenAI 服务 smoke、完整评测、A/B、训练环境与重评分证据。
- 保留 `train.py`、`train_dpo.py`、`evaluate_v4.py`、`serve_model.py`、`export_model.py` 等当前通用入口；`train_dpo.py` 要求显式配置，避免把 v1-v3 历史 DPO 当成 v4 默认流程。

执行前再次运行了 `rg`，确认待删脚本只有旧链路内部引用，v4 主线无调用者；历史结果、用例、配置和 manifest 已归档。

第二次清理统计：

- 清理前仓库：5,096,011,052 bytes（4.746030 GiB）。
- 清理前 `outputs/`：5,089,051,501 bytes（4.739549 GiB）。
- 删除目标合计：4,947,007,428 bytes（4.607260 GiB）。
- 清理后仓库：149,006,315 bytes（0.138773 GiB）。
- 清理后 `outputs/`：143,705,895 bytes（0.133837 GiB）。
- 最终 `outputs/` 只保留 `qwen3_lora_tool_calling_v4_1` 顶层 Direct Adapter，不含 checkpoint。
- 外部 ModelScope 基座仍为 8,060,926,179 bytes（7.507323 GiB），未修改。

两次清理累计按删除目标计释放 13,448,021,413 bytes（约 12.524 GiB）。
