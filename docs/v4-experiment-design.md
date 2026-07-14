# v4 / v4.1 实验设计与最终结论

## 1. 目标

训练 Qwen3-4B，使其成为项目二当前 Researcher 工具路由任务的本地专用模型。只替换 Researcher；项目二默认仍是 DeepSeek V4 Pro，其他 Agent 不切换。

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

基座：`C:\Users\0\.cache\modelscope\Qwen\Qwen3-4B`。

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

capability holdout 剩余 2 条参数问题，未产生不可用工具、不支持平台或标识符幻觉。

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

## 12. 最终结论

通过：v4.1 在独立 hard/holdout 上相对 Qwen3-4B 基座取得稳定提升，安全指标满足要求，具备 Researcher 候选替代价值。

未通过或未证明：项目二整体产品质量、成本、吞吐、线上用户泛化、默认生产替换、merged 模型部署。

当前主路径：`Qwen3-4B 4-bit Base + outputs/qwen3_lora_tool_calling_v4_1 Direct Adapter`。
