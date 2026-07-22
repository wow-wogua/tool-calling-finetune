# v4 / v4.1 实验设计与最终结论

## 1. 目标

训练 Qwen3-4B，使其成为项目二当前 Researcher 工具路由任务的本地专用模型。只替换 Researcher；项目二默认仍是 DeepSeek V4 Pro，其他 Agent 不切换。

项目来源于项目二的真实工程问题，而不是为了练习 LoRA：Researcher 只输出工具名和 JSON 参数，任务窄、标签明确、可自动评测；通用 API 模型曾出现参数越界和结构化输出不稳定；项目二 `ModelRegistry` 又允许只替换这一节点。因此提出实验假设：

> 在项目二当前动态工具契约下，经过能力条件化训练的 Qwen3-4B，能否显著超过 Qwen 基座，并在窄域 Researcher 工具路由上接近或超过 DeepSeek V4 Pro？

两个问题必须分开回答：

1. Qwen Base vs v4.1：证明 LoRA 微调是否有效。
2. v4.1 vs DeepSeek：判断是否具备项目二候选替代价值。

工程闭环：项目二 Researcher 问题 → 冻结契约 → 可追溯数据 → 先冻结 hard/holdout → QLoRA → Base/Adapter 公平评测 → OpenAI 兼容服务 → 项目二只读 A/B → DeepSeek 同集基线 → 工程决策。

## 2. 契约

来源：`viral-video-agent@2aa1e54`。

快照：`schemas/project2_researcher_contract_20260714.json`。

任务输入：

- 当前 Researcher 子任务。
- 用户目标平台。
- 当前动态可用工具。

输出：`{"tool":"...","params":{...}}`。

边界：

- 实时搜索只支持 bilibili。
- RAG 可检索多平台知识资料，但不等于多平台实时搜索。
- ASR 可用且有公开 B 站 URL 才能调用转写。
- 趋势工具默认不可用。
- 缺少关键标识符或无需新增证据时输出 `none`。

## 3. 评测先冻结

| 文件 | 数量 | 职责 |
|---|---:|---|
| `data/eval/v4_dev.json` | 24 | 错误分析 |
| `data/eval/v4_hard.json` | 40 | 复杂自然语言与边界 |
| `data/eval/v4_holdout.json` | 30 | 最终候选泛化验证 |
| `data/eval/v4_capability_holdout.json` | 16 | 能力状态与安全边界 |

combined SHA-256：`6b77b03e0dde15c06e4f9b5a2a639e26db2bcc86ee9557b0c3616d01c55cbab0`。

切分规则：

- expression family 不跨训练和冻结评测。
- 精确、归一化和 0.96 以上近重复禁止训练。
- holdout 候选确定后才运行。
- holdout 单题不得用于再补数据。

## 4. 数据

最终 835 条：750 train / 85 validation。

| 来源 | 数量 | 参与 |
|---|---:|---|
| 程序化模板 | 720 | train/validation |
| 手写自然语言 | 14 | train/validation |
| 手写边界 | 48 | train/validation |
| 项目二冻结产品输入 | 4 | train/validation |
| v4.1 类别补充 | 49 | train/validation |

不使用 MiMo/DeepSeek/其他 LLM 生成，不使用线上用户数据。

工具分布：search217、rag201、transcript90、none327。

## 5. 训练

硬件：RTX 4060 Laptop 8GB。

基座：本机已有的 Qwen3-4B 缓存，通过 `BASE_MODEL_PATH` 指定。

v4：

- 4-bit QLoRA。
- rank16、alpha32、dropout0.05、target all。
- 3 epoch、learning rate 1e-4。
- 135 optimizer steps。
- 最终 validation loss 约 0.00165。

v4.1：

- 从 v4 Direct Adapter 继续训练。
- 1 epoch、learning rate 2e-5。
- 47 optimizer steps。
- 唯一一次错误类别改进。

## 6. 指标

- tool_accuracy
- params_accuracy
- full_accuracy
- safe_accuracy
- json_valid_rate
- unavailable_tool_violation_rate
- unsupported_platform_violation_rate
- hallucinated_id_or_url_rate
- capability_state_accuracy
- latency mean / p50 / p95

评分模拟项目二运行时默认值：search 默认 keyword/limit/platforms，RAG 缺 platform 时由单一目标平台注入。

## 7. 通过门槛

- hard 与 holdout 都超过最强 Qwen Base。
- full 有稳定提升。
- safe 不下降。
- 不可用工具、不支持平台和标识符幻觉接近 0。
- JSON 接近 100%。
- 两次确定性运行基本一致。

## 8. 结果

最强 Base：

| 数据集 | full | safe |
|---|---:|---:|
| hard | 82.50% | 82.50% |
| holdout | 76.67% | 76.67% |
| capability holdout | 68.75% | 68.75% |

v4.1：

| 数据集 | tool | params | full | safe | JSON | unavailable | unsupported | hallucinated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard | 95.0% | 95.24% | 95.0% | 95.0% | 100% | 0% | 0% | 0% |
| holdout | 100% | 93.33% | 96.67% | 96.67% | 100% | 0% | 0% | 0% |
| capability | 87.5% | 71.43% | 87.5% | 87.5% | 100% | 0% | 0% | 0% |

两次复跑原始输出完全一致。

## 9. 错误分析

hard 剩余 2 条 wrong_tool：

- 一条“真实案例 vs 理论”复合表达仍选择 RAG。
- 一条无 ID 趋势表达仍被当成搜索关键词。

holdout 剩余 1 条 wrong_params：工具正确，RAG 检索 query/top_k 参数与人工期望不完全等价。

capability holdout 剩余 2 条 wrong-tool，未产生不可用工具、不支持平台或标识符幻觉。

不继续追单题。

## 10. DPO 决策

不执行。SFT 已满足通过门槛，剩余错误不足以证明 DPO 的预期收益。v1–v3 的 DPO 历史也没有证明独立 hard/holdout 泛化，因此避免机械两阶段训练。

## 11. 服务与项目二 A/B

服务：FastAPI OpenAI 兼容接口，4-bit Base + Direct Adapter。项目二完整 Prompt 不重复包装；响应经项目二 Schema 适配。

3 个冻结任务 A/B：

| 指标 | DeepSeek | v4.1 |
|---|---:|---:|
| completed / partial / failed | 1 / 2 / 0 | 1 / 2 / 0 |
| invalid params | 1 | 0 |
| unavailable calls | 0 | 0 |
| raw data | 3 | 3 |
| mean total | 41.35s | 39.69s |
| mean Researcher | 5.00s | 5.27s |

外部工具空结果主导 partial，不能据此比较模型整体质量。

## 12. DeepSeek V4 Pro 同集基线

### 12.1 公平条件

- hard40、holdout30、capability16，共 86 条；冻结 hash 未变化。
- DeepSeek 和 v4.1 接收相同 `contract` Prompt、能力状态、工具 Schema 和用户输入。
- 同一 `strict_json`、参数归一化和 `score_case` 评分核心。
- DeepSeek 使用 `temperature=0`、`max_tokens=96`、`extra_body={"thinking":{"type":"disabled"}}`。
- 不增加额外路由规则，不人工修正输出，不根据结果改数据或重新训练。
- 每条成功响应即时写 checkpoint；86 次成功请求，0 失败、0 重试。

### 12.2 三模型结果

主表使用相同 `contract` Prompt 的 Qwen Base，而不是逐 split 挑选 Prompt 的“最强 Base”；后者仍保留用于更保守的 LoRA 门槛判断。

| 数据集 | 指标 | Qwen Base | Qwen v4.1 | DeepSeek V4 Pro |
|---|---|---:|---:|---:|
| hard40 | tool | 80.00% | **95.00%** | 90.00% |
|  | params | 95.24% | 95.24% | **100%** |
|  | full / safe | 80.00% | **95.00%** | 90.00% |
| holdout30 | tool | 76.67% | **100%** | 73.33% |
|  | params | 86.67% | **93.33%** | **93.33%** |
|  | full / safe | 70.00% | **96.67%** | 70.00% |
| capability16 | tool | 68.75% | **87.50%** | **87.50%** |
|  | params | 85.71% | 71.43% | **100%** |
|  | full / safe | 68.75% | **87.50%** | **87.50%** |
| 86条总体 | tool | 76.74% | **95.35%** | 83.72% |
|  | params | 90.70% | 90.70% | **97.67%** |
|  | full / safe | 74.42% | **94.19%** | 82.56% |

三模型 JSON 均为 100%，不可用工具违规均为 0。v4.1 的不支持平台和 URL/ID 幻觉均为 0；DeepSeek 总体分别为 2.33% 和 1.16%。

### 12.3 错误差异

- DeepSeek：hard 4 条 wrong-tool；holdout 8 条 wrong-tool、1 条 wrong-params，其中包含 1 条不支持平台调用和 1 条编造 B 站 URL；capability 2 条 wrong-tool。主要失败模式是没有充分选择 `none`。
- v4.1：hard 2 条唯一 wrong-tool、holdout 1 条 wrong-params、capability 2 条 wrong-tool；没有三类安全违规。主要短板是少量真实样本/RAG语义判断和 capability RAG 参数。
- DeepSeek 参数准确率更高，v4.1 工具路由与安全边界更强。

### 12.4 Token、成本和延迟

- 输入 20,000 token，输出 2,360 token，总计 22,360 token。
- 按项目二 2026-07-13 价格表估算：`$0.010753`。
- DeepSeek API 总体 mean / p50 / p95：1.140s / 1.101s / 1.500s。
- v4.1 本地 RTX 4060 总体 mean / p50 / p95：2.054s / 1.373s / 3.876s。

延迟来自远程 API 与本地单卡两套不同基础设施，只能描述，不能作为严格速度结论。本实验也没有测量并发、常驻显存、电费、服务运维和生产总拥有成本。

## 13. 最终工程决策

离线模型层结论：v4.1 在三个冻结 split 上均不低于 DeepSeek，hard 高 5.00 个百分点、holdout 高 26.67 个百分点、capability 持平。因此允许表述：

> 在冻结的自建窄域 Researcher 评测上超过 DeepSeek V4 Pro。

项目二接入决策：v4.1 成为优先候选，但项目二默认暂不自动切换，继续使用 DeepSeek V4 Pro，并保留本地 OpenAI 兼容回退路径。

产品层证据：3 条端到端 A/B 只证明接入可工作，两组 completed/partial 分布相同，不能证明整体产品质量胜出。

线上层尚未验证：真实用户分布、更大的端到端任务集、长期稳定性、并发吞吐、生产式失败率和严格同基础设施成本/延迟。

会改变决策的证据：更大的冻结项目二端到端任务集仍保持优势，长时间服务没有稳定性回归，生产式监控显示质量与失败率可接受。

范围在此结束：不新增模型、训练轮次、DPO、工具、平台、LLM-as-Judge，也不为追分修改 holdout。

当前主路径：`Qwen3-4B 4-bit Base + outputs/qwen3_lora_tool_calling_v4_1 Direct Adapter`。
