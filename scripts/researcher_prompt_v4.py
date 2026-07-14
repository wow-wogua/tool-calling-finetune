"""Project-2-compatible dynamic Researcher prompt for the v4 experiment."""

from __future__ import annotations


TOOL_DEFINITIONS = {
    "search_videos": {
        "signature": "search_videos(keyword, platforms, limit)",
        "description": "搜索B站当前/近期热门视频样本，单次最多20条；支持平台: bilibili",
    },
    "rag_search": {
        "signature": "rag_search(query, top_k, platform)",
        "description": "检索本地知识库中的平台规则、方法论和历史案例；支持平台: bilibili, douyin, kuaishou, xiaohongshu, generic",
    },
    "get_transcript": {
        "signature": "get_transcript(video_url)",
        "description": "转写公开B站视频；支持平台: bilibili",
    },
    "get_trend_data": {
        "signature": "get_trend_data(video_id, platform)",
        "description": "获取指定B站视频的历史趋势数据；支持平台: bilibili",
    },
}


DEFAULT_CAPABILITIES = {
    "search_videos": {"enabled": True, "availability": "real"},
    "rag_search": {"enabled": True, "availability": "local"},
    "get_transcript": {"enabled": True, "availability": "mimo"},
    "get_trend_data": {"enabled": False, "availability": "unavailable"},
}


ROUTING_RULES = """路由规则:
1. 需要B站当前、近期或真实视频样本时选择 search_videos；其他平台的实时搜索不受支持。
2. 需要知识库中的规则、方法论、指标框架、参考资料或历史案例时选择 rag_search。
3. 只有 get_transcript 出现在当前可用工具中且任务给出公开B站 URL 时，才能选择 get_transcript。
4. 只有 get_trend_data 出现在当前可用工具中且任务给出视频 ID 时，才能选择 get_trend_data。
5. 用户明确不要搜索、只基于已有 Evidence 总结、直接创作，或所需工具/参数不可用时选择 none。
6. 不得臆造 URL、视频 ID、平台能力或未提供的关键参数。"""


def normalize_capabilities(capabilities: dict | None) -> dict:
    normalized = {
        name: dict(state) for name, state in DEFAULT_CAPABILITIES.items()
    }
    for name, state in (capabilities or {}).items():
        if name in normalized and isinstance(state, dict):
            normalized[name].update(state)
    return normalized


def render_available_tools(capabilities: dict | None = None) -> str:
    states = normalize_capabilities(capabilities)
    lines = []
    for name, definition in TOOL_DEFINITIONS.items():
        state = states[name]
        if not state.get("enabled"):
            continue
        suffix = " [仅演示mock]" if state.get("availability") == "mock" else ""
        lines.append(
            f"- {definition['signature']}: {definition['description']}{suffix}"
        )
    lines.append("- none: 当前步骤不需要新增外部数据或资料")
    return "\n".join(lines)


def build_researcher_prompt(
    task: str,
    platforms: list[str] | None = None,
    capabilities: dict | None = None,
    variant: str = "contract",
) -> str:
    """Build the complete prompt sent by project 2 as one user message."""
    if variant not in {"contract", "rules"}:
        raise ValueError(f"unknown v4 prompt variant: {variant}")
    target_platforms = platforms or ["bilibili"]
    prompt = f"""你是证据采集执行器。根据当前任务选择一个当前可用工具并生成参数。

当前任务: {task}
用户目标平台: {target_platforms}

当前可用工具:
{render_available_tools(capabilities)}

约束:
- 只能选择上面列出的工具或 none
- 不得输出未支持的平台
- 不得臆造视频 ID 或链接
- 只输出一个合法 JSON 对象

输出格式：{{"tool": "工具名", "params": {{"参数名": "值"}}}}"""
    if variant == "rules":
        prompt = f"{prompt}\n\n{ROUTING_RULES}"
    return prompt


def is_complete_project2_prompt(text: str) -> bool:
    markers = ("当前任务:", "当前可用工具:", "输出格式")
    return all(marker in (text or "") for marker in markers)
