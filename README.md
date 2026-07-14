# Tool Calling Finetune — Qwen3-4B Researcher 路由微调

本项目训练一个本地 Qwen3-4B 专用模型，只负责 `viral-video-agent` 的 Researcher 工具选择和参数生成。项目二当前默认仍是 DeepSeek V4 Pro；Planner、Analyst、Writer 不切换，本模型也不是项目二的生产默认模型。

## 项目来源与实验假设

项目三不是脱离业务的 LoRA 练习，而是来自项目二的 Researcher 工具路由问题：

- Researcher 只负责选择工具并生成短 JSON 参数，任务窄、标签明确、可以自动评分。
- 通用 API 模型在真实 Agent 链路中曾出现参数越界和结构化输出不稳定。
- 项目二 `ModelRegistry` 允许只替换 Researcher，不需要让 4B 模型承担 Planner、Analyst 和 Writer。
- 因此需要验证的不是“小模型是否全面替代大模型”，而是“本地 4B 专用模型能否承担这个窄节点”。

实验假设：

> 在项目二当前动态工具契约下，经过能力条件化训练的 Qwen3-4B，能否显著超过 Qwen 基座，并在窄域 Researcher 工具路由上接近或超过 DeepSeek V4 Pro？

它拆成两个独立问题：

1. Qwen Base vs v4.1：LoRA 微调是否有效。
2. v4.1 vs DeepSeek：是否具备项目二候选替代价值。

完整闭环：

```mermaid
flowchart LR
    A["项目二 Researcher 问题"] --> B["冻结工具契约"]
    B --> C["构造可追溯数据"]
    C --> D["先冻结 hard / holdout"]
    D --> E["Qwen3-4B QLoRA"]
    E --> F["Base / Adapter 公平评测"]
    F --> G["OpenAI 兼容服务"]
    G --> H["项目二只读 A/B"]
    H --> I["DeepSeek 同集基线"]
    I --> J["最终工程决策"]
```

## 当前结论

三模型使用同一 `contract` Prompt、能力状态、工具 Schema、用户输入和评分核心：

| 数据集 | Qwen Base full | Qwen v4.1 full | DeepSeek V4 Pro full |
|---|---:|---:|---:|
| hard40 | 80.00% | **95.00%** | 90.00% |
| holdout30 | 70.00% | **96.67%** | 70.00% |
| capability holdout16 | 68.75% | **87.50%** | **87.50%** |
| 86 条加权总体 | 74.42% | **94.19%** | 82.56% |

v4.1 在三个冻结 split 上均不低于 DeepSeek，并在 hard 高 5.00 个百分点、holdout 高 26.67 个百分点、capability 持平。因此本项目允许的结论是：

> 在冻结的自建窄域 Researcher 评测上超过 DeepSeek V4 Pro。

这不是“全面超过 DeepSeek”。当前工程决策是：v4.1 成为 Researcher 优先候选，但项目二默认仍使用 DeepSeek；是否切换需要更大的端到端冻结任务集、长期服务稳定性和生产式监控证据。

主要证据：

- `results/v4/baseline_base_contract_rescored.json`
- `results/v4/baseline_base_rules_rescored.json`
- `results/v4/v4_1_final_repeated.json`
- `results/v4/deepseek_v4_pro_frozen_eval.json`
- `results/v4/final_three_model_comparison.json`
- `results/v4/project2_ab_comparison.json`
- `docs/v4-experiment-design.md`
- `docs/experiment-history.md`

## 与项目二的边界

契约快照来自 `viral-video-agent` 提交 `2aa1e54`：

- 项目二默认 Agent 模型：DeepSeek V4 Pro。
- 只有 Researcher 可通过 `USE_FINETUNED_MODEL=true` 临时切到本地 OpenAI 兼容服务。
- 真实视频搜索只支持 bilibili。
- `rag_search` 检索本地规则、方法论和历史案例。
- `get_transcript` 只有在 ASR 可用且输入含公开 B 站 URL 时可调用。
- `get_trend_data` 当前没有真实供应商，默认不可用。
- `none` 表示当前步骤不需要新增证据，或所需工具/关键参数不可用。
- 项目二发送完整动态 Researcher Prompt；项目三服务不得重复包装。

版本化快照见 `schemas/project2_researcher_contract_20260714.json`。训练和独立评测只依赖项目三内的快照与脚本，不依赖项目二源码。

## v4 任务定义

> 给定用户任务、目标平台和当前允许的工具能力，选择合法工具并生成满足 Schema 的参数；不可用工具不得调用，缺少 URL 或视频 ID 时不得编造。

项目三输出一个 JSON 对象：

```json
{"tool":"search_videos","params":{"keyword":"机器人","platforms":["bilibili"],"limit":10}}
```

服务适配层还会执行确定性 Schema 防线：

- `search_videos.limit` 裁剪到 1–20，只允许 bilibili。
- `rag_search.top_k` 裁剪到 1–10。
- 未出现在动态工具列表中的工具改为 `none`。
- 非公开 B 站 URL、缺少 URL/ID 或输入中不存在的标识符不执行。

这些规则与项目二运行时 Pydantic Schema 对齐，不修改项目二。

## 数据与防泄漏

评测集先于训练数据冻结，锁定文件为 `data/v4/manifests/holdout_lock.json`，combined SHA-256：

```text
6b77b03e0dde15c06e4f9b5a2a639e26db2bcc86ee9557b0c3616d01c55cbab0
```

| 集合 | 数量 | 用途 |
|---|---:|---|
| v4 dev | 24 | 错误分析，可迭代，不作为最终泛化证明 |
| v4 hard | 40 | 复杂自然语言和工具边界；只按错误类别补数据 |
| v4 holdout | 30 | 候选确定后才运行，不按单题补训练数据 |
| capability holdout | 16 | ASR状态、趋势不可用、非B站搜索、缺URL/ID、none |

最终 v4.1 数据共 835 条：750 train / 85 分布内 validation。

| 真实来源 | 数量 |
|---|---:|
| 程序化模板合成 | 720 |
| 手写自然语言样例 | 14 |
| 手写边界样例 | 48 |
| 项目二冻结产品输入 | 4 |
| v4.1 错误类别补充 | 49 |

没有使用 MiMo、DeepSeek 或其他大模型批量生成训练数据，也没有使用线上用户数据。

最终工具分布：`search_videos=217`、`rag_search=201`、`get_transcript=90`、`none=327`。训练前自动检查：

- JSON/工具参数 Schema。
- 冻结文件 hash。
- 训练与评测的精确文本、归一化文本和高相似近重复。
- expression family 是否跨训练与冻结评测。
- manifest 文件 hash 和来源声明。

最终结果均为 0 个表达族重合、0 个精确/归一化重合、0 个近重复。

## 训练

环境：RTX 4060 Laptop 8GB、Qwen3-4B、LLaMA Factory 0.9.5、4-bit QLoRA、LoRA rank 16 / alpha 32。

基座固定复用：

```text
C:\Users\0\.cache\modelscope\Qwen\Qwen3-4B
```

不得删除或重复下载该目录，也不使用 8B 模型。

流程：

1. 2-step smoke：验证数据加载、显存、loss、保存和重载。
2. v4 SFT：707 train / 79 validation，3 epoch。
3. dev/hard 错误分析。
4. 唯一一次 v4.1：新增 49 条类别级边界表达，从 v4 Adapter 继续 1 epoch。
5. 候选确定后运行 holdout 和 capability holdout，两次确定性复跑。

SFT 已通过目标门槛，因此没有机械执行 DPO。剩余错误不足以证明偏好优化收益会超过引入新偏差的风险。

## 公平基线

Base、v3 Direct Adapter 和 v4/v4.1 统一使用：

- 同一 Qwen3-4B 基座。
- 4-bit 推理。
- 同一版本化工具 Schema 和冻结评测集。
- `temperature=0`、`do_sample=false`。
- `max_new_tokens=96`。
- 相同运行时默认值与评分逻辑。

训练前最强 Base：

| 配置 | hard full | holdout full | capability full |
|---|---:|---:|---:|
| Base + contract | 80.00% | 70.00% | **68.75%** |
| Base + rules | **82.50%** | **76.67%** | 62.50% |
| v3 Direct + contract | 75.00% | 70.00% | 62.50% |

LoRA 效果的主证明只来自 `Qwen3-4B Base` 与 `Qwen3-4B + v4.1 Direct Adapter`。DeepSeek 同集评测用于判断窄域候选价值，项目二 A/B 用于验证真实 Agent 链路接入；二者都不能替代 Base/Adapter 对照来证明 LoRA 本身有效。

## DeepSeek V4 Pro 同集基线

DeepSeek 使用与 v4.1 完全相同的 86 条冻结评测、`contract` Prompt、动态能力状态、工具 Schema 和评分核心。生成参数为 `temperature=0`、`max_tokens=96`，并传入 `extra_body={"thinking":{"type":"disabled"}}`。没有增加额外路由规则，也没有人工修正输出。

| 数据集 | 模型 | tool | params | full / safe | JSON | unsupported | hallucinated URL/ID |
|---|---|---:|---:|---:|---:|---:|---:|
| hard40 | Qwen Base | 80.00% | 95.24% | 80.00% / 80.00% | 100% | 2.50% | 0% |
|  | Qwen v4.1 | **95.00%** | 95.24% | **95.00% / 95.00%** | 100% | **0%** | 0% |
|  | DeepSeek | 90.00% | **100%** | 90.00% / 90.00% | 100% | 2.50% | 0% |
| holdout30 | Qwen Base | 76.67% | 86.67% | 70.00% / 70.00% | 100% | 3.33% | 0% |
|  | Qwen v4.1 | **100%** | **93.33%** | **96.67% / 96.67%** | 100% | **0%** | **0%** |
|  | DeepSeek | 73.33% | **93.33%** | 70.00% / 70.00% | 100% | 3.33% | 3.33% |
| capability16 | Qwen Base | 68.75% | 85.71% | 68.75% / 68.75% | 100% | 0% | 0% |
|  | Qwen v4.1 | **87.50%** | 71.43% | **87.50% / 87.50%** | 100% | 0% | 0% |
|  | DeepSeek | **87.50%** | **100%** | **87.50% / 87.50%** | 100% | 0% | 0% |

DeepSeek 的优势是参数准确率，尤其 capability 参数达到 100%；主要问题是更容易把“能力或关键参数不可用”误路由为 RAG/搜索/转写。它在 holdout 出现 8 条 wrong-tool、1 条不支持平台调用和 1 条编造 URL/ID。v4.1 的优势是 `none` 和安全边界，三类安全违规均为 0；短板仍是 capability 中两条 RAG 路由/参数语义错误。

真实 API 共 86 次成功请求、0 失败、0 重试；输入 20,000 token、输出 2,360 token，按项目二价格表估算成本 `$0.010753`。平均延迟：

| 数据集 | Qwen Base 本地 | Qwen v4.1 本地 | DeepSeek API |
|---|---:|---:|---:|
| hard | 1.984s | 2.158s | **1.118s** |
| holdout | 2.029s | 2.019s | **1.192s** |
| capability | 2.027s | 1.859s | **1.095s** |

本地 Qwen 与远程 API 的硬件、并发和网络环境不同，延迟仅作描述，不能据此声称本地模型已经更快或更便宜。完整结果见 `results/v4/deepseek_v4_pro_frozen_eval.json` 和 `results/v4/final_three_model_comparison.json`。

## 项目二只读 A/B

使用 3 个项目二冻结任务，项目二提交保持 `2aa1e54`，源码和配置均未修改。两组都使用相同 v2 图、MCP、B站工具和 DeepSeek Planner/Analyst/Writer；仅 Researcher 不同。

| 指标 | DeepSeek Researcher | v4.1 Researcher |
|---|---:|---:|
| completed / partial / failed | 1 / 2 / 0 | 1 / 2 / 0 |
| invalid params | 1 | 0 |
| unavailable tool calls | 0 | 0 |
| raw data | 3 | 3 |
| 平均总耗时 | 41.35s | 39.69s |
| 平均 Researcher 耗时 | 5.00s | 5.27s |

partial 主要来自 B 站榜单关键词无匹配和 RAG 空结果。A/B 证明服务、Prompt、Schema、解析和图接入可工作，但没有证明 v4.1 的端到端产品质量或延迟全面优于 DeepSeek。

## 最终工程决策

- 模型层：v4.1 在冻结窄域同集评测上超过 DeepSeek，成为 Researcher 优先候选。
- 项目二接入：默认模型暂不自动切换，继续使用 DeepSeek V4 Pro；v4.1 保留为可回退的本地候选路径。
- 尚未验证：更大规模端到端任务、长期服务稳定性、并发吞吐、线上用户分布、严格同基础设施成本与延迟。
- 会改变决策的证据：更大的冻结项目二任务集仍保持优势，长期运行没有稳定性回归，生产式监控显示质量与失败率可接受。
- 范围到此结束：不再增加新模型、训练轮次、DPO、工具、平台、LLM-as-Judge 或为追分修改 holdout。

## 快速复现

安装依赖：

```powershell
pip install -r requirements.txt
```

在全新工作区先冻结评测，再构建训练数据：

```powershell
python scripts/build_v4_dataset.py --freeze-eval
python scripts/build_v4_dataset.py --build-train
python scripts/validate_dataset.py
```

已冻结的实验中不要使用 `--force` 改写评测集。

训练：

```powershell
python scripts/train.py configs/qwen3_lora_v4_smoke.yaml
python scripts/train.py configs/qwen3_lora_v4.yaml
python scripts/train.py configs/qwen3_lora_v4_1.yaml
```

评测 Base：

```powershell
python scripts/evaluate_v4.py `
  --label base_contract `
  --prompt-variant contract `
  --output results/v4/baseline_base_contract.json
```

评测最终 Direct Adapter：

```powershell
python scripts/evaluate_v4.py `
  --label v4_1_final_repeated `
  --prompt-variant contract `
  --adapter outputs/qwen3_lora_tool_calling_v4_1 `
  --splits hard,holdout,capability_holdout `
  --repeats 2 `
  --output results/v4/v4_1_final_repeated.json
```

启动 OpenAI 兼容服务：

```powershell
$env:BASE_MODEL_PATH='C:\Users\0\.cache\modelscope\Qwen\Qwen3-4B'
$env:FINETUNED_ADAPTER_PATH='outputs\qwen3_lora_tool_calling_v4_1'
$env:RESEARCHER_PROMPT_VARIANT='contract'
python scripts/serve_model.py
```

服务地址：`http://localhost:8002/v1`。验证：

```powershell
python scripts/smoke_openai_service.py
```

项目二临时切换：

```text
USE_FINETUNED_MODEL=true
FINETUNED_MODEL_URL=http://host.docker.internal:8002/v1
```

A/B 后恢复 `USE_FINETUNED_MODEL=false`。本次 A/B 使用进程级临时变量，项目二 `.env` 未修改。

DeepSeek 同集基线支持断点续跑；API Key 只从环境变量读取，不写入结果：

```powershell
python scripts/evaluate_v4.py `
  --api `
  --api-model deepseek-v4-pro `
  --label deepseek_v4_pro_frozen_eval `
  --prompt-variant contract `
  --splits hard,holdout,capability_holdout `
  --checkpoint results/v4/deepseek_v4_pro_checkpoint.json `
  --output results/v4/deepseek_v4_pro_frozen_eval.json

python scripts/compare_three_models.py
```

## Direct Adapter 与 merged 模型

当前验证主路径是 `4-bit Base + Direct Adapter`，不提供 merged 模型作为主结论。

v1–v3 曾在 4-bit 量化权重上直接 merge，产物输出与 Base 完全相同，而 Direct Adapter 会改变输出，说明旧 merge 路径掩盖了 Adapter 效果。旧大体积 merged 权重已删除，结果 JSON 与实验结论保留。

如确需 merged：必须以 BF16 加载基座、合并 Adapter、保存 BF16，再单独量化，并重跑完整 hard/holdout。当前没有必要为本地 A/B 额外占用内存和磁盘。

## 可声称与不可声称

可以声称：

- 构建了对齐项目二当前动态能力的 Qwen3-4B Researcher 专用模型。
- 评测集先冻结，训练与 holdout 没有检测到泄漏。
- v4.1 在独立 hard/holdout 上相对最强 Qwen 基座取得实际提升。
- 不可用工具、不支持平台和 URL/ID 幻觉在最终重复评测中为 0。
- 通过 OpenAI 兼容服务完成项目二 Researcher 只读 A/B。
- 在冻结的自建窄域 Researcher 同集评测上超过 DeepSeek V4 Pro。

不能声称：

- 已替代项目二默认 DeepSeek V4 Pro。
- Planner、Analyst、Writer 已本地化。
- A/B 证明端到端产品质量、速度、成本或线上泛化全面更优。
- v4.1 全面超过 DeepSeek，或已经达到生产默认切换条件。
- 训练数据来自大模型生成或真实线上用户。
- 4-bit merged 模型已经验证可用。

## 历史

v1–v3 的内置 50 条结果曾从 88% / 54% 提升到 94% / 80%，但旧 hard/holdout 与 Direct Adapter 公平实验没有证明自然表达泛化提升。失败过程、数据来源、4-bit merge 风险与结果证据均保留在 `docs/experiment-history.md` 和 `results/legacy/v1-v3/`。最终门槛判定见 `results/v4/final_verdict.json`，磁盘与旧资产清理记录见 `docs/cleanup-inventory-20260714.md`。

## Git 边界

允许提交小型 JSON 数据、评测集、配置、结果、代码和文档。禁止提交：

- `outputs/`、模型 checkpoint、`*.safetensors`、`*.bin`。
- 缓存、日志、`.env`、密钥。
- 项目二数据库或 Trace 原始敏感内容。
- 本地面试资料与简历。

## License

MIT
