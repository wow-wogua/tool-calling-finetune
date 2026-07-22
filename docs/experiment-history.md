# 实验演进：v1–v4.1

本文件保留失败过程和结论变化。旧权重可以删除，历史结果和判断不能删除。

## v1：先验证结构化工具调用能否学会

真实数据来源：

- 30 条手写样例。
- `generate_data.py` 的程序化模板合成。
- `fix_rag_data.py` 的 `rag_search` / `none` 边界补充。
- 去重后 310 条有效 instruction。

内置 50 条 BFCL 风格自建用例：

| 模型 | tool | full |
|---|---:|---:|
| Qwen3-4B Base | 88% | 54% |
| SFT | 92% | 78% |
| SFT + DPO | 94% | 80% |

这一阶段证明了格式学习和已覆盖场景有效，但内置评测与模板数据接近，不能外推自然语言泛化。

## v2：补 hard 边界

新增：

- 60 条 SFT 边界表达。
- 48 条 DPO chosen/rejected。
- 44 条独立 hard eval。

目标是解决 `search_videos` / `rag_search` / `none` 易混问题。历史 merged 结果在 hard44 上与 Base 相同：84.1% tool、52.3% full。

结论：针对错误类型补数据并不自动等于分布外泛化提升。

## v3：增加 holdout 和 Direct Adapter 复核

新增：

- 40 条 SFT 边界表达。
- 24 条 DPO 偏好对。
- 20 条新 holdout。

旧 merged v3 在 holdout20 上仍与 Base 相同：65% tool、35% full。随后用同一基座、Prompt、数据集和生成参数直接加载 Adapter：

| 配置 | hard full / safe | holdout full / safe |
|---|---:|---:|
| v3 Direct + base prompt | 77.3% / 68.2% | 90.0% / 90.0% |
| v3 Direct + rules prompt | 90.9% / 75.0% | 75.0% / 75.0% |

Direct Adapter 会改变输出，旧 merged 却与 Base 完全相同。检查发现旧 merged 的 `config.json` 保留 `load_in_4bit=true`，说明当时在量化权重上的 merge 路径不可信。

结论：

- 不能用旧 merged 判断 Adapter 无效。
- 也不能因某个 Prompt 下单项提升就声称稳定泛化。
- 当前主结论必须来自 Direct Adapter。

## v4：按项目二当前契约重新定义任务

重新设计原因：

- 旧训练目标仍包含静态多平台实时搜索和默认可用趋势工具，与项目二当前能力不一致。
- 旧 hard/holdout 没有充分条件化 ASR 可用状态。
- 训练输入没有逐字对齐项目二 `researcher_dynamic` 完整 Prompt。
- 数据切分只做文本去重，不足以约束表达族泄漏。

v4 的任务变为：给定任务、平台和当前能力列表，选择合法工具并生成 Schema 参数；不可用工具不调用，缺 URL/ID 不编造。

评测先冻结：dev24、hard40、holdout30、capability holdout16。最终 v4 初始数据 786 条，其中 707 train / 79 validation。

训练配置：Qwen3-4B、4-bit QLoRA、LoRA rank16 / alpha32、3 epoch。训练前完成 2-step smoke 和 Adapter 重载。

统一重评分后：

| 数据集 | v4 full / safe | 主要剩余错误 |
|---|---:|---|
| dev | 100% / 100% | 无 |
| hard | 92.5% / 92.5% | 非B站转写、无ID趋势、真实案例/方法论各一类 |

v4 已超过 Base，但不支持平台违规仍为 2.5%，因此使用唯一一次类别级改进。

## v4.1：一次错误类别改进

只新增 49 条新表达：

- 非 B 站链接不得调用转写。
- 没有视频 ID 时趋势请求选择 `none`。
- “需要真实案例，不是方法论”选择 `search_videos`。

没有复制 hard 原题，也没有在补数据前查看 v4 holdout。最终数据 835 条：750 train / 85 validation。从 v4 Adapter 以 2e-5 学习率继续 1 epoch。

最终两次确定性复跑：

| 数据集 | tool | params | full | safe | JSON | capability |
|---|---:|---:|---:|---:|---:|---:|
| hard40 | 95.0% | 95.24% | 95.0% | 95.0% | 100% | 90.91% |
| holdout30 | 100% | 93.33% | 96.67% | 96.67% | 100% | 100% |
| capability16 | 87.5% | 71.43% | 87.5% | 87.5% | 100% | 87.5% |

不可用工具、不支持平台和 URL/ID 幻觉均为 0；两次原始输出无差异。

## 为什么没有做 v4 DPO

v1–v3 已证明“有 chosen/rejected”不等于会改善独立边界。v4.1 SFT 已通过 hard/holdout 和安全门槛，剩余错误数量少且包含任务语义歧义。继续 DPO 的边际证据不足，可能损害已经稳定的 `none`、参数和能力状态表现，因此跳过。

## 项目二只读 A/B

项目二提交：`2aa1e54`。3 个冻结任务中，两组都是 1 completed / 2 partial / 0 failed，raw data 都为 3。v4.1 通过服务 Schema 适配将 invalid params 从 1 降到 0，但 Researcher 延迟没有稳定优势。

两组 partial 的共同原因是 B 站榜单关键词无匹配或 RAG 空结果。因此 A/B 证明接入兼容和候选价值，不证明产品效果全面优于 DeepSeek。

## 2026-07-20：真实 Arq Worker 可选接入

前述 3 条只读 A/B 保留为历史兼容性证据。后续在不修改项目三训练代码、推理代码、数据、Prompt、Schema、模型、Adapter、checkpoint 和评测结果的前提下，将现有 Qwen3-4B + v4.1 Direct Adapter 服务接入项目二真实 Arq Worker：

- `USE_FINETUNED_MODEL=false` 仍是默认值；显式启用时只覆盖 Researcher。
- App 与 Worker 共用路由初始化，Planner、Analyst、Writer 仍走 DeepSeek V4 Pro，MiMo ASR 不变。
- `qwen3-tool-calling` 只是 OpenAI 兼容服务模型 ID，不是另一个模型版本。
- 有效 canary 中 Planner 调用 DeepSeek 1 次，Researcher 本地请求 2 次且 HTTP `200/200`，JSON 解析 2 次，`invalid_params=0`。
- 最终 `partial / EVIDENCE_INSUFFICIENT` 是外部工具返回 `empty` 导致，不是 Worker 或模型路由失败。
- true→false 回退后 Researcher 覆盖清除，本地请求数不再增长。

该证据只覆盖一次有限 canary，没有证明生产稳定性、并发、成本、延迟或产品质量更优；没有替换项目二默认 DeepSeek，也没有重新训练、继续微调、合并 Adapter 或重新量化。

## DeepSeek V4 Pro 同集基线

为了回答 v4.1 与项目二默认 Researcher 模型处于什么水平，在不改训练、不改评测集的前提下，对 hard40、holdout30、capability16 全部 86 条调用 `deepseek-v4-pro`。

公平条件：同一个 `contract` Prompt、能力状态、工具 Schema、用户输入和评分核心；`temperature=0`、`max_tokens=96`、thinking disabled。逐条 checkpoint 保存，最终 86 次成功、0 失败、0 重试，输入 20,000 token、输出 2,360 token，估算成本 `$0.010753`。

| 数据集 | Qwen Base contract full | v4.1 full | DeepSeek full |
|---|---:|---:|---:|
| hard40 | 80.00% | **95.00%** | 90.00% |
| holdout30 | 70.00% | **96.67%** | 70.00% |
| capability16 | 68.75% | **87.50%** | **87.50%** |
| 加权总体 | 74.42% | **94.19%** | 82.56% |

DeepSeek 的参数准确率总体更高（97.67% vs v4.1 90.70%），但工具准确率更低（83.72% vs 95.35%）。它更容易在能力、平台或关键参数不满足时仍调用 RAG/搜索/转写：总体不支持平台违规 2.33%，URL/ID 幻觉 1.16%；v4.1 三类安全违规均为 0。

结论规则要求 v4.1 在所有冻结 split 都不低于 DeepSeek，且至少一个 split 更高，才归类为“超过”。本次 hard 高 5.00 个百分点、holdout 高 26.67 个百分点、capability 持平，因此允许说：

> 在冻结的自建窄域 Researcher 评测上超过 DeepSeek V4 Pro。

这不推翻项目二 3 条 A/B 的边界，也不等于全面超过 DeepSeek。工程上把 v4.1 提升为优先候选，但项目二默认不自动切换；需要更大的端到端冻结任务集、长期稳定性和生产式监控才能改变默认决策。

## 失败实验如何影响设计

- 内置高分但 hard 不提升：评测必须先冻结并分层。
- 针对单题补数据：改为只按错误类别补全新表达，并限制一次。
- 静态工具列表：改为动态能力条件化。
- 量化 merge 输出不可信：主路径改为 Direct Adapter。
- 参数越界：OpenAI 服务层按项目二 Schema 做确定性校验。
- A/B partial：区分模型路由错误与外部工具空结果。
- 强基线缺失：加入 DeepSeek 同集评测，把“LoRA 是否有效”和“是否值得替换 API 模型”拆成两个问题。

至此项目三闭环结束：问题来源、契约、数据、训练、独立评测、服务化、Agent 接入、强基线和工程决策均已形成证据链，不再通过新增模型或修改 holdout 追分。
