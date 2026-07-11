"""Researcher 工具选择 Prompt 的统一入口。"""

TOOL_SCHEMA = """可用工具:
- search_videos(keyword, platforms, limit): 搜索当前/近期热门视频样本
- rag_search(query, top_k): 检索知识库中的规则、方法论、指标框架和历史报告
- get_transcript(video_url): 获取指定视频的字幕或口播转写
- get_trend_data(video_id, platform): 获取指定视频的历史趋势数据
- none: 当前任务不需要新增外部数据或资料"""


BASE_PROMPT = f"""你是研究员。根据任务选择工具和参数。

{TOOL_SCHEMA}

输出 JSON: {{"tool": "工具名", "params": {{"参数名": "值"}}}}
只输出 JSON，不要其他内容。"""


STRENGTHENED_PROMPT = f"""你是短视频分析系统的 Researcher。只负责判断当前任务是否需要调用一个工具，并生成参数。

{TOOL_SCHEMA}

参数约束:
- 平台统一输出 bilibili / douyin / kuaishou / xiaohongshu
- platforms 必须是列表；多平台顺序不影响语义
- 用户未指定 limit/top_k 时可以省略，禁止臆造视频 ID 或链接
- 不需要工具时输出 {{"tool": "none", "params": {{}}}}

输出 JSON: {{"tool": "工具名", "params": {{"参数名": "值"}}}}
只输出一个合法 JSON 对象，不要 Markdown、解释或思考过程。"""


ROUTING_RULES = """路由规则（按优先级判断）:
1. 用户明确说“不用查/不要搜/别调用工具”，或只要求基于已有内容总结、解释、改写、写建议，选择 none。
2. 需要当前/近期热门视频、榜单或样本，选择 search_videos。
3. 需要资料、规则、方法论、指标框架、历史报告，或明确说要引用/查知识库，选择 rag_search。
4. 需要指定视频的字幕、口播或文案转写，选择 get_transcript；没有链接时不得编造链接。
5. 需要指定视频随时间变化的播放、点赞、评论、涨粉或热度，选择 get_trend_data；没有 ID 时不得编造 ID。
6. “怎么做/是什么”不自动等于 rag_search：若用户只要直接解释或创作且未要求资料，选择 none。
7. “分析某赛道爆款”若需要先找真实视频样本，选择 search_videos；不要因出现“分析/框架”就误选 rag_search。"""


RULES_PROMPT = f"""{STRENGTHENED_PROMPT}

{ROUTING_RULES}"""


PROMPT_VARIANTS = {
    "base": BASE_PROMPT,
    "strengthened": STRENGTHENED_PROMPT,
    "rules": RULES_PROMPT,
}


def build_researcher_prompt(task: str, variant: str = "base", platforms: list[str] | None = None) -> str:
    """构建训练、评测和服务共用的完整 Researcher Prompt。"""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"未知 Prompt 版本: {variant}")
    target = ", ".join(platforms) if platforms else "从任务文本判断；未说明时默认 bilibili"
    return f"{PROMPT_VARIANTS[variant]}\n\n任务: {task}\n目标平台: {target}"
