"""Freeze v4 evaluation sets first, then build reproducible SFT data.

No LLM is used. Sources are handwritten cases and deterministic templates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from researcher_prompt_v4 import DEFAULT_CAPABILITIES, build_researcher_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
EVAL_ROOT = DATA_ROOT / "eval"
V4_ROOT = DATA_ROOT / "v4"
SEED = 20260714


CAPABILITY_PROFILES = {
    "asr_on": {
        "search_videos": {"enabled": True, "availability": "real"},
        "rag_search": {"enabled": True, "availability": "local"},
        "get_transcript": {"enabled": True, "availability": "mimo"},
        "get_trend_data": {"enabled": False, "availability": "unavailable"},
    },
    "asr_off": {
        "search_videos": {"enabled": True, "availability": "real"},
        "rag_search": {"enabled": True, "availability": "local"},
        "get_transcript": {"enabled": False, "availability": "unconfigured"},
        "get_trend_data": {"enabled": False, "availability": "unavailable"},
    },
}


def _capabilities(profile: str) -> dict:
    if profile not in CAPABILITY_PROFILES:
        raise ValueError(f"unknown capability profile: {profile}")
    return {
        name: dict(state)
        for name, state in CAPABILITY_PROFILES[profile].items()
    }


def case(
    case_id: str,
    text: str,
    tool: str,
    params: dict,
    intent_family: str,
    expression_family: str,
    parameter_pattern: str,
    *,
    platforms: list[str] | None = None,
    profile: str = "asr_on",
    safety_tags: list[str] | None = None,
) -> dict:
    return {
        "id": case_id,
        "input": text,
        "platforms": platforms or ["bilibili"],
        "capability_profile": profile,
        "capabilities": _capabilities(profile),
        "expected_tool": tool,
        "expected_params": params,
        "intent_family": intent_family,
        "expression_family": expression_family,
        "parameter_pattern": parameter_pattern,
        "source": "handwritten_eval",
        "safety_tags": safety_tags or [],
    }


def eval_catalog() -> dict[str, list[dict]]:
    dev = [
        case("dev-search-01", "先给我找8条B站家居区最近火起来的视频", "search_videos", {"keyword": "家居", "platforms": ["bilibili"], "limit": 8}, "real_sample_search", "dev_search_request", "keyword_limit"),
        case("dev-search-02", "分析之前先拉一批B站数码测评真实样本", "search_videos", {"keyword": "数码测评", "platforms": ["bilibili"]}, "real_sample_search", "dev_search_prerequisite", "keyword"),
        case("dev-search-03", "B站知识区这两天有哪些热门内容可当样本", "search_videos", {"keyword": "知识", "platforms": ["bilibili"]}, "real_sample_search", "dev_search_question", "keyword"),
        case("dev-search-04", "拉20个B站汽车频道近期热视频", "search_videos", {"keyword": "汽车", "platforms": ["bilibili"], "limit": 20}, "real_sample_search", "dev_search_count", "keyword_limit"),
        case("dev-rag-01", "从知识库找一下完播率诊断方法", "rag_search", {"query": "完播率诊断方法", "platform": "bilibili"}, "knowledge_retrieval", "dev_rag_direct", "query_platform"),
        case("dev-rag-02", "我需要引用B站标题写作规则，不要去搜视频", "rag_search", {"query": "B站标题写作规则", "platform": "bilibili"}, "knowledge_retrieval", "dev_rag_no_video", "query_platform", safety_tags=["no_video_search"]),
        case("dev-rag-03", "查内部资料里的竞品复盘模板", "rag_search", {"query": "竞品复盘模板"}, "knowledge_retrieval", "dev_rag_internal", "query"),
        case("dev-rag-04", "抖音推荐机制有现成资料吗，只查知识库", "rag_search", {"query": "抖音推荐机制", "platform": "douyin"}, "knowledge_retrieval", "dev_rag_other_platform", "query_platform", platforms=["douyin"]),
        case("dev-none-01", "沿用现有Evidence，把结论压缩成三点", "none", {}, "no_new_evidence", "dev_none_existing", "empty", safety_tags=["existing_evidence"]),
        case("dev-none-02", "不要查任何东西，直接给五个标题方向", "none", {}, "explicit_no_tool", "dev_none_explicit", "empty", safety_tags=["explicit_no_tool"]),
        case("dev-none-03", "解释一下互动率是什么意思", "none", {}, "direct_explanation", "dev_none_explain", "empty"),
        case("dev-none-04", "把上面的报告改成更口语的版本", "none", {}, "rewrite_existing", "dev_none_rewrite", "empty", safety_tags=["existing_evidence"]),
        case("dev-asr-01", "转写这个B站视频 https://www.bilibili.com/video/BV1Ab411c7X1", "get_transcript", {"video_url": "https://www.bilibili.com/video/BV1Ab411c7X1"}, "transcript_with_url", "dev_asr_url", "video_url"),
        case("dev-asr-02", "把 https://b23.tv/BV1Cd4y1Q7p2 的口播整理出来", "get_transcript", {"video_url": "https://b23.tv/BV1Cd4y1Q7p2"}, "transcript_with_url", "dev_asr_short_url", "video_url"),
        case("dev-asr-missing-01", "帮我把那个视频转成文字", "none", {}, "transcript_missing_url", "dev_asr_missing", "empty", safety_tags=["missing_url"]),
        case("dev-asr-off-01", "转写 https://www.bilibili.com/video/BV1Ef4y1R7m3", "none", {}, "transcript_unavailable", "dev_asr_disabled", "empty", profile="asr_off", safety_tags=["unavailable_tool"]),
        case("dev-platform-01", "搜一下抖音近期母婴爆款样本", "none", {}, "unsupported_realtime_platform", "dev_platform_douyin", "empty", platforms=["douyin"], safety_tags=["unsupported_platform"]),
        case("dev-platform-02", "小红书最近流行哪些家装视频，给真实样本", "none", {}, "unsupported_realtime_platform", "dev_platform_xhs", "empty", platforms=["xiaohongshu"], safety_tags=["unsupported_platform"]),
        case("dev-trend-01", "看看BV1Gh411K7n4过去一周播放趋势", "none", {}, "trend_unavailable", "dev_trend_id", "empty", safety_tags=["unavailable_tool"]),
        case("dev-trend-02", "这条视频热度怎么变化的", "none", {}, "trend_missing_id", "dev_trend_missing", "empty", safety_tags=["missing_id", "unavailable_tool"]),
        case("dev-rag-05", "给我知识库里关于开头三秒钩子的参考资料", "rag_search", {"query": "开头三秒钩子"}, "knowledge_retrieval", "dev_rag_reference", "query"),
        case("dev-search-05", "想研究B站宠物赛道，先找近期真实视频", "search_videos", {"keyword": "宠物", "platforms": ["bilibili"]}, "real_sample_search", "dev_search_research", "keyword"),
        case("dev-none-05", "基于刚才已经拿到的样本做归纳，不要新增检索", "none", {}, "no_new_evidence", "dev_none_no_new", "empty", safety_tags=["existing_evidence", "explicit_no_tool"]),
        case("dev-search-06", "B站舞蹈区热门榜给我前12个", "search_videos", {"keyword": "舞蹈", "platforms": ["bilibili"], "limit": 12}, "real_sample_search", "dev_search_rank", "keyword_limit"),
    ]

    hard = [
        case("hard-01", "不是要运营方法论，我要能点开的B站露营热视频，先抓6条", "search_videos", {"keyword": "露营", "platforms": ["bilibili"], "limit": 6}, "real_sample_search", "hard_contrast_sample_vs_method", "keyword_limit"),
        case("hard-02", "别给我现搜样本，只从资料库找B站封面设计的判断框架", "rag_search", {"query": "B站封面设计判断框架", "platform": "bilibili"}, "knowledge_retrieval", "hard_contrast_method_vs_sample", "query_platform", safety_tags=["no_video_search"]),
        case("hard-03", "手头已有20条样本了，按现有证据说说共同点，不再补数据", "none", {}, "no_new_evidence", "hard_existing_evidence", "empty", safety_tags=["existing_evidence", "explicit_no_tool"]),
        case("hard-04", "我只是想要十个选题脑暴，不需要你查库也不用搜视频", "none", {}, "explicit_no_tool", "hard_double_negative", "empty", safety_tags=["explicit_no_tool"]),
        case("hard-05", "先搜B站最近火的AI办公教程，数量控制在7条", "search_videos", {"keyword": "AI办公教程", "platforms": ["bilibili"], "limit": 7}, "real_sample_search", "hard_search_colloquial", "keyword_limit"),
        case("hard-06", "做B站历史区复盘，第一步要真实案例，不是查理论", "search_videos", {"keyword": "历史", "platforms": ["bilibili"]}, "real_sample_search", "hard_search_not_theory", "keyword"),
        case("hard-07", "查一下知识库是否有用户留存分析框架，返回最多9条参考", "rag_search", {"query": "用户留存分析框架", "top_k": 9}, "knowledge_retrieval", "hard_rag_topk", "query_top_k"),
        case("hard-08", "想了解快手的推荐规则，资料库里有就行，不做实时搜索", "rag_search", {"query": "快手推荐规则", "platform": "kuaishou"}, "knowledge_retrieval", "hard_rag_other_platform", "query_platform", platforms=["kuaishou"]),
        case("hard-09", "请把这个公开视频的字幕拿下来：https://www.bilibili.com/video/BV1Jk4y1L7s5?spm_id_from=333", "get_transcript", {"video_url": "https://www.bilibili.com/video/BV1Jk4y1L7s5?spm_id_from=333"}, "transcript_with_url", "hard_asr_query_url", "video_url"),
        case("hard-10", "链接在这 https://b23.tv/BV1Lm4y1M7t6，提取口播即可", "get_transcript", {"video_url": "https://b23.tv/BV1Lm4y1M7t6"}, "transcript_with_url", "hard_asr_link_first", "video_url"),
        case("hard-11", "我忘记贴链接了，但你先把视频转写出来", "none", {}, "transcript_missing_url", "hard_asr_missing_pressure", "empty", safety_tags=["missing_url"]),
        case("hard-12", "ASR现在没开，不过还是帮我转写 https://www.bilibili.com/video/BV1Np4y1N7u7", "none", {}, "transcript_unavailable", "hard_asr_unavailable_explicit", "empty", profile="asr_off", safety_tags=["unavailable_tool"]),
        case("hard-13", "给我这条视频最近30天的涨粉曲线，BV1Qr4y1P7v8", "none", {}, "trend_unavailable", "hard_trend_with_id", "empty", safety_tags=["unavailable_tool"]),
        case("hard-14", "趋势供应商还没接，直接查一下BV1St4y1Q7w9的热度变化", "none", {}, "trend_unavailable", "hard_trend_supplier_off", "empty", safety_tags=["unavailable_tool"]),
        case("hard-15", "抖音美食最近哪些视频爆了？要真实条目，不要方法论", "none", {}, "unsupported_realtime_platform", "hard_unsupported_douyin", "empty", platforms=["douyin"], safety_tags=["unsupported_platform"]),
        case("hard-16", "同时找B站和小红书的近期穿搭样本", "none", {}, "unsupported_realtime_platform", "hard_mixed_platform", "empty", platforms=["bilibili", "xiaohongshu"], safety_tags=["unsupported_platform"]),
        case("hard-17", "B站当前最火的科普视频有哪些？", "search_videos", {"keyword": "科普", "platforms": ["bilibili"]}, "real_sample_search", "hard_search_question_short", "keyword"),
        case("hard-18", "给我找B站全站热门前5，关键词可以留空", "search_videos", {"keyword": "", "platforms": ["bilibili"], "limit": 5}, "real_sample_search", "hard_search_empty_keyword", "empty_keyword_limit"),
        case("hard-19", "需要引用一套短视频竞品分析流程，优先内部资料", "rag_search", {"query": "短视频竞品分析流程"}, "knowledge_retrieval", "hard_rag_reference_priority", "query"),
        case("hard-20", "不用联网，看看知识库里有没有AIDA和AISAS的对比", "rag_search", {"query": "AIDA AISAS 对比"}, "knowledge_retrieval", "hard_rag_no_network", "query"),
        case("hard-21", "把已经生成的分析改成面试时两分钟能讲完的版本", "none", {}, "rewrite_existing", "hard_rewrite_interview", "empty", safety_tags=["existing_evidence"]),
        case("hard-22", "直接说明什么叫冷启动，别检索", "none", {}, "direct_explanation", "hard_explain_no_search", "empty", safety_tags=["explicit_no_tool"]),
        case("hard-23", "先别分析，抓18条B站健身热视频给后面用", "search_videos", {"keyword": "健身", "platforms": ["bilibili"], "limit": 18}, "real_sample_search", "hard_search_deferred_analysis", "keyword_limit"),
        case("hard-24", "B站母婴赛道有没有最近的真实爆款可参考", "search_videos", {"keyword": "母婴", "platforms": ["bilibili"]}, "real_sample_search", "hard_search_reference", "keyword"),
        case("hard-25", "找资料解释B站流量池机制，别拿当前热视频凑数", "rag_search", {"query": "B站流量池机制", "platform": "bilibili"}, "knowledge_retrieval", "hard_rag_not_samples", "query_platform", safety_tags=["no_video_search"]),
        case("hard-26", "查知识库中的口播节奏设计方法，最多给4条", "rag_search", {"query": "口播节奏设计方法", "top_k": 4}, "knowledge_retrieval", "hard_rag_small_topk", "query_top_k"),
        case("hard-27", "这里只要基于前文给建议，任何新工具都不要调用", "none", {}, "explicit_no_tool", "hard_none_any_tool", "empty", safety_tags=["existing_evidence", "explicit_no_tool"]),
        case("hard-28", "写三个B站科技区标题，完全不需要外部依据", "none", {}, "direct_creation", "hard_none_creation", "empty"),
        case("hard-29", "提取 https://www.bilibili.com/video/BV1Uv4y1R7x1 里的台词", "get_transcript", {"video_url": "https://www.bilibili.com/video/BV1Uv4y1R7x1"}, "transcript_with_url", "hard_asr_extract", "video_url"),
        case("hard-30", "这个链接是YouTube的 https://youtube.com/watch?v=abc123，帮我转写", "none", {}, "transcript_unsupported_platform", "hard_asr_non_bilibili", "empty", safety_tags=["unsupported_platform"]),
        case("hard-31", "没有视频ID，你先给我查趋势", "none", {}, "trend_missing_id", "hard_trend_missing_id", "empty", safety_tags=["missing_id", "unavailable_tool"]),
        case("hard-32", "趋势工具不可用时就不要硬调，我只想知道下一步怎么办", "none", {}, "trend_unavailable", "hard_trend_instruction", "empty", safety_tags=["unavailable_tool"]),
        case("hard-33", "去小红书搜一批真实护肤视频，再回来分析", "none", {}, "unsupported_realtime_platform", "hard_unsupported_xhs", "empty", platforms=["xiaohongshu"], safety_tags=["unsupported_platform"]),
        case("hard-34", "查小红书种草方法论，资料库内容就够了", "rag_search", {"query": "小红书种草方法论", "platform": "xiaohongshu"}, "knowledge_retrieval", "hard_rag_xhs", "query_platform", platforms=["xiaohongshu"]),
        case("hard-35", "B站游戏区今天有什么热门，来10条", "search_videos", {"keyword": "游戏", "platforms": ["bilibili"], "limit": 10}, "real_sample_search", "hard_search_today", "keyword_limit"),
        case("hard-36", "我需要过去的案例复盘，不是当前热门榜", "rag_search", {"query": "历史案例复盘"}, "knowledge_retrieval", "hard_rag_historical", "query"),
        case("hard-37", "根据已有表格总结，不增加样本、不查规则", "none", {}, "no_new_evidence", "hard_none_two_sources", "empty", safety_tags=["existing_evidence", "explicit_no_tool"]),
        case("hard-38", "给现有报告补一个更有冲击力的开头", "none", {}, "rewrite_existing", "hard_none_edit", "empty", safety_tags=["existing_evidence"]),
        case("hard-39", "ASR可用，转写短链 https://b23.tv/BV1Wx4y1S7y2", "get_transcript", {"video_url": "https://b23.tv/BV1Wx4y1S7y2"}, "transcript_with_url", "hard_asr_state_on", "video_url"),
        case("hard-40", "ASR没配置，链接是 https://b23.tv/BV1Yz4y1T7z3，也不要调用", "none", {}, "transcript_unavailable", "hard_asr_state_off", "empty", profile="asr_off", safety_tags=["unavailable_tool"]),
    ]

    holdout = [
        case("holdout-01", "做露营账号竞品研究，素材得是真实的，先从B站取9条最近视频", "search_videos", {"keyword": "露营", "platforms": ["bilibili"], "limit": 9}, "real_sample_search", "holdout_material_first", "keyword_limit"),
        case("holdout-02", "B站摄影分区眼下什么内容热，给样本就行", "search_videos", {"keyword": "摄影", "platforms": ["bilibili"]}, "real_sample_search", "holdout_current_heat", "keyword"),
        case("holdout-03", "我要验证一个标题套路，先抽14个B站财经热视频", "search_videos", {"keyword": "财经", "platforms": ["bilibili"], "limit": 14}, "real_sample_search", "holdout_validate_hypothesis", "keyword_limit"),
        case("holdout-04", "从B站搜点近期手工区案例，后面再谈方法", "search_videos", {"keyword": "手工", "platforms": ["bilibili"]}, "real_sample_search", "holdout_examples_before_method", "keyword"),
        case("holdout-05", "我缺的是用户画像分析的参考框架，不是视频列表", "rag_search", {"query": "用户画像分析框架"}, "knowledge_retrieval", "holdout_need_framework", "query"),
        case("holdout-06", "翻一下资料库里的B站分区标签规则", "rag_search", {"query": "B站分区标签规则", "platform": "bilibili"}, "knowledge_retrieval", "holdout_browse_docs", "query_platform"),
        case("holdout-07", "只看内部沉淀，找一套视频数据异常排查清单", "rag_search", {"query": "视频数据异常排查清单"}, "knowledge_retrieval", "holdout_internal_only", "query"),
        case("holdout-08", "有没有适合小红书的内容矩阵方法论？不需要实时帖子", "rag_search", {"query": "小红书内容矩阵方法论", "platform": "xiaohongshu"}, "knowledge_retrieval", "holdout_other_platform_docs", "query_platform", platforms=["xiaohongshu"]),
        case("holdout-09", "现成数据足够了，照着它写一个简短结论", "none", {}, "no_new_evidence", "holdout_sufficient_data", "empty", safety_tags=["existing_evidence"]),
        case("holdout-10", "这一步纯改文案，外部工具全部跳过", "none", {}, "explicit_no_tool", "holdout_copy_edit", "empty", safety_tags=["explicit_no_tool"]),
        case("holdout-11", "直接给我一个账号定位的例子", "none", {}, "direct_creation", "holdout_direct_example", "empty"),
        case("holdout-12", "把刚才那五点合成一段话", "none", {}, "rewrite_existing", "holdout_merge_existing", "empty", safety_tags=["existing_evidence"]),
        case("holdout-13", "请识别这条B站视频的语音：https://www.bilibili.com/video/BV1Za4y1U7a4", "get_transcript", {"video_url": "https://www.bilibili.com/video/BV1Za4y1U7a4"}, "transcript_with_url", "holdout_asr_identify", "video_url"),
        case("holdout-14", "https://b23.tv/BV1Bc4y1V7b5 这条的字幕给我", "get_transcript", {"video_url": "https://b23.tv/BV1Bc4y1V7b5"}, "transcript_with_url", "holdout_asr_url_first", "video_url"),
        case("holdout-15", "链接稍后再给，你先生成这条视频的逐字稿", "none", {}, "transcript_missing_url", "holdout_asr_deferred_url", "empty", safety_tags=["missing_url"]),
        case("holdout-16", "当前没有转写服务，但我把链接给你 https://www.bilibili.com/video/BV1De4y1W7c6", "none", {}, "transcript_unavailable", "holdout_asr_no_service", "empty", profile="asr_off", safety_tags=["unavailable_tool"]),
        case("holdout-17", "给我抖音最近的宠物爆款原始样本", "none", {}, "unsupported_realtime_platform", "holdout_douyin_samples", "empty", platforms=["douyin"], safety_tags=["unsupported_platform"]),
        case("holdout-18", "快手汽车赛道这周热视频拉一批", "none", {}, "unsupported_realtime_platform", "holdout_kuaishou_samples", "empty", platforms=["kuaishou"], safety_tags=["unsupported_platform"]),
        case("holdout-19", "B站和抖音都搜一下最近的知识视频", "none", {}, "unsupported_realtime_platform", "holdout_mixed_search", "empty", platforms=["bilibili", "douyin"], safety_tags=["unsupported_platform"]),
        case("holdout-20", "BV1Fg4y1X7d7这条的播放曲线拿一下", "none", {}, "trend_unavailable", "holdout_trend_curve", "empty", safety_tags=["unavailable_tool"]),
        case("holdout-21", "没给BV号也没接趋势服务，先查热度变化", "none", {}, "trend_missing_id", "holdout_trend_double_missing", "empty", safety_tags=["missing_id", "unavailable_tool"]),
        case("holdout-22", "B站校园区近期样本选五个", "search_videos", {"keyword": "校园", "platforms": ["bilibili"], "limit": 5}, "real_sample_search", "holdout_select_count", "keyword_limit"),
        case("holdout-23", "找B站影视区当下热门内容做对照", "search_videos", {"keyword": "影视", "platforms": ["bilibili"]}, "real_sample_search", "holdout_make_comparison", "keyword"),
        case("holdout-24", "资料库里关于选题打分的办法，取3条最相关的", "rag_search", {"query": "选题打分方法", "top_k": 3}, "knowledge_retrieval", "holdout_rag_top3", "query_top_k"),
        case("holdout-25", "要一份抖音内容审核规则参考，不抓实时内容", "rag_search", {"query": "抖音内容审核规则", "platform": "douyin"}, "knowledge_retrieval", "holdout_rag_douyin_policy", "query_platform", platforms=["douyin"]),
        case("holdout-26", "已经有Evidence，不要为润色再调用任何东西", "none", {}, "explicit_no_tool", "holdout_polish_existing", "empty", safety_tags=["existing_evidence", "explicit_no_tool"]),
        case("holdout-27", "用一句话解释什么是互动成本", "none", {}, "direct_explanation", "holdout_one_sentence", "empty"),
        case("holdout-28", "这条公开视频 https://www.bilibili.com/video/BV1Hi4y1Y7e8 的口播稿提取一下", "get_transcript", {"video_url": "https://www.bilibili.com/video/BV1Hi4y1Y7e8"}, "transcript_with_url", "holdout_asr_public", "video_url"),
        case("holdout-29", "服务没开时不要转写 https://b23.tv/BV1Jk4y1Z7f9", "none", {}, "transcript_unavailable", "holdout_asr_respect_state", "empty", profile="asr_off", safety_tags=["unavailable_tool"]),
        case("holdout-30", "小红书近期读书类真实帖子样本给我", "none", {}, "unsupported_realtime_platform", "holdout_xhs_samples", "empty", platforms=["xiaohongshu"], safety_tags=["unsupported_platform"]),
    ]

    capability_holdout = [
        case("cap-01-on", "转写 https://www.bilibili.com/video/BV1Lm4y1A7g1", "get_transcript", {"video_url": "https://www.bilibili.com/video/BV1Lm4y1A7g1"}, "capability_asr", "cap_pair_01", "video_url", profile="asr_on", safety_tags=["capability_focus"]),
        case("cap-01-off", "转写 https://www.bilibili.com/video/BV1Lm4y1A7g1", "none", {}, "capability_asr", "cap_pair_01", "empty", profile="asr_off", safety_tags=["capability_focus", "unavailable_tool"]),
        case("cap-02-on", "提取 https://b23.tv/BV1No4y1B7h2 的字幕", "get_transcript", {"video_url": "https://b23.tv/BV1No4y1B7h2"}, "capability_asr", "cap_pair_02", "video_url", profile="asr_on", safety_tags=["capability_focus"]),
        case("cap-02-off", "提取 https://b23.tv/BV1No4y1B7h2 的字幕", "none", {}, "capability_asr", "cap_pair_02", "empty", profile="asr_off", safety_tags=["capability_focus", "unavailable_tool"]),
        case("cap-03", "想转写一个视频，但没有给链接", "none", {}, "capability_missing_url", "cap_missing_url", "empty", profile="asr_on", safety_tags=["capability_focus", "missing_url"]),
        case("cap-04", "搜索B站近期新能源车视频", "search_videos", {"keyword": "新能源车", "platforms": ["bilibili"]}, "capability_search", "cap_bilibili_search", "keyword", safety_tags=["capability_focus"]),
        case("cap-05", "搜索抖音近期新能源车视频", "none", {}, "capability_platform", "cap_douyin_search", "empty", platforms=["douyin"], safety_tags=["capability_focus", "unsupported_platform"]),
        case("cap-06", "搜索快手近期新能源车视频", "none", {}, "capability_platform", "cap_kuaishou_search", "empty", platforms=["kuaishou"], safety_tags=["capability_focus", "unsupported_platform"]),
        case("cap-07", "查抖音新能源车内容方法论", "rag_search", {"query": "抖音新能源车内容方法论", "platform": "douyin"}, "capability_rag", "cap_douyin_rag", "query_platform", platforms=["douyin"], safety_tags=["capability_focus"]),
        case("cap-08", "查B站新能源车内容方法论", "rag_search", {"query": "B站新能源车内容方法论", "platform": "bilibili"}, "capability_rag", "cap_bilibili_rag", "query_platform", safety_tags=["capability_focus"]),
        case("cap-09", "查BV1Qr4y1C7i3的播放趋势", "none", {}, "capability_trend", "cap_trend_id", "empty", safety_tags=["capability_focus", "unavailable_tool"]),
        case("cap-10", "没有BV号，查一下播放趋势", "none", {}, "capability_trend", "cap_trend_missing", "empty", safety_tags=["capability_focus", "unavailable_tool", "missing_id"]),
        case("cap-11", "基于已有结果做摘要", "none", {}, "capability_none", "cap_existing_summary", "empty", safety_tags=["capability_focus", "existing_evidence"]),
        case("cap-12", "不要搜索，直接写标题", "none", {}, "capability_none", "cap_explicit_none", "empty", safety_tags=["capability_focus", "explicit_no_tool"]),
        case("cap-13", "B站近期旅行视频找11个", "search_videos", {"keyword": "旅行", "platforms": ["bilibili"], "limit": 11}, "capability_search", "cap_search_limit", "keyword_limit", safety_tags=["capability_focus"]),
        case("cap-14", "知识库找旅行视频复盘框架", "rag_search", {"query": "旅行视频复盘框架"}, "capability_rag", "cap_rag_query", "query", safety_tags=["capability_focus"]),
    ]
    return {
        "v4_dev.json": dev,
        "v4_hard.json": hard,
        "v4_holdout.json": holdout,
        "v4_capability_holdout.json": capability_holdout,
    }


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(cases: list[dict]) -> dict:
    return {
        "tools": dict(sorted(Counter(item["expected_tool"] for item in cases).items())),
        "capability_profiles": dict(sorted(Counter(item["capability_profile"] for item in cases).items())),
        "intent_families": dict(sorted(Counter(item["intent_family"] for item in cases).items())),
        "sources": dict(sorted(Counter(item["source"] for item in cases).items())),
    }


def freeze_eval(force: bool = False) -> None:
    catalog = eval_catalog()
    for filename, cases in catalog.items():
        path = EVAL_ROOT / filename
        if path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite frozen eval file: {path}")
        write_json(path, cases)

    manifests = {}
    for filename, cases in catalog.items():
        path = EVAL_ROOT / filename
        manifests[filename] = {
            "count": len(cases),
            "sha256": sha256_file(path),
            "case_ids": [item["id"] for item in cases],
            "distribution": distribution(cases),
        }
    combined = hashlib.sha256(
        canonical_json({name: item["sha256"] for name, item in manifests.items()}).encode("utf-8")
    ).hexdigest()
    lock = {
        "schema_version": 1,
        "created_date": "2026-07-14",
        "seed": SEED,
        "creation_method": "handwritten evaluation catalog; no LLM generation",
        "files": manifests,
        "combined_sha256": combined,
        "rules": [
            "holdout files are frozen before training data generation",
            "holdout cases must not be copied or paraphrased into training data",
            "training and holdout expression_family values must be disjoint",
        ],
    }
    write_json(V4_ROOT / "manifests" / "holdout_lock.json", lock)
    print(json.dumps({"frozen": manifests, "combined_sha256": combined}, ensure_ascii=False, indent=2))


def verify_eval_lock() -> dict:
    lock_path = V4_ROOT / "manifests" / "holdout_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("holdout lock missing; run --freeze-eval first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for filename, expected in lock["files"].items():
        path = EVAL_ROOT / filename
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise ValueError(f"frozen eval hash changed: {filename}")
    return lock


def training_case(
    case_id: str,
    text: str,
    tool: str,
    params: dict,
    intent_family: str,
    expression_family: str,
    parameter_pattern: str,
    *,
    platforms: list[str] | None = None,
    profile: str = "asr_on",
    source: str = "programmatic_template",
) -> dict:
    item = case(
        case_id,
        text,
        tool,
        params,
        intent_family,
        expression_family,
        parameter_pattern,
        platforms=platforms,
        profile=profile,
    )
    item["source"] = source
    return item


def build_training_catalog() -> list[dict]:
    rows: list[dict] = []
    seen = set()

    def add(item: dict) -> None:
        identity = (item["input"], item["capability_profile"], canonical_json(item["platforms"]))
        if identity in seen:
            return
        seen.add(identity)
        rows.append(item)

    search_topics = [
        "美食教程", "国风音乐", "职场成长", "运动健身", "亲子教育", "科技新品",
        "游戏攻略", "城市旅行", "家常菜", "纪录片", "校园生活", "宠物日常",
        "汽车测评", "数码开箱", "读书分享", "手工制作", "摄影技巧", "舞蹈翻跳",
        "历史科普", "心理成长", "家居改造", "电影解说", "财经知识", "户外徒步",
    ]
    search_templates = [
        ("给后续分析准备素材，先在B站找{topic}近期热视频", None),
        ("从B站抓一批最近的{topic}真实样本", None),
        ("B站{topic}现在什么内容热，先取视频", None),
        ("我要复盘{topic}赛道，第一步拉B站当前案例", None),
        ("在B站搜索{topic}热门视频，返回{limit}条", "limit"),
        ("B站{topic}样本取前{limit}个", "limit"),
        ("先别总结，帮我找{limit}条B站{topic}近期内容", "limit"),
        ("研究{topic}之前，从B站拿{limit}个真实视频", "limit"),
    ]
    for topic_index, topic in enumerate(search_topics):
        for template_index, (template, mode) in enumerate(search_templates):
            limit = 3 + ((topic_index * 3 + template_index) % 18)
            text = template.format(topic=topic, limit=limit)
            params = {"keyword": topic, "platforms": ["bilibili"]}
            pattern = "keyword"
            if mode == "limit":
                params["limit"] = limit
                pattern = "keyword_limit"
            add(training_case(
                f"train-search-{topic_index:02d}-{template_index:02d}", text,
                "search_videos", params, "real_sample_search",
                f"train_search_t{template_index:02d}", pattern,
            ))

    rag_topics = [
        "标题写作框架", "封面设计原则", "开头钩子方法", "完播率诊断", "互动率指标",
        "账号定位方法", "用户画像模板", "竞品分析流程", "选题评估模型", "内容矩阵方法",
        "B站推荐规则", "抖音审核规范", "小红书种草方法", "快手运营规则", "短视频脚本结构",
        "AIDA框架", "AISAS模型", "内容复盘清单", "数据异常排查", "历史爆款案例",
        "口播节奏设计", "视频叙事方法", "评论区运营", "发布节奏策略",
    ]
    rag_templates = [
        ("从知识库检索{topic}", None),
        ("我需要一份{topic}的内部参考", None),
        ("查资料库里有没有{topic}", None),
        ("不要找实时视频，给我{topic}相关资料", None),
        ("知识库搜索{topic}，取{top_k}条", "top_k"),
        ("关于{topic}，最多返回{top_k}份参考", "top_k"),
        ("先查{topic}的方法论，不要拉样本", None),
        ("引用内部沉淀的{topic}", None),
    ]
    for topic_index, topic in enumerate(rag_topics):
        for template_index, (template, mode) in enumerate(rag_templates):
            top_k = 2 + ((topic_index + template_index) % 9)
            text = template.format(topic=topic, top_k=top_k)
            params = {"query": topic}
            pattern = "query"
            if mode == "top_k":
                params["top_k"] = top_k
                pattern = "query_top_k"
            add(training_case(
                f"train-rag-{topic_index:02d}-{template_index:02d}", text,
                "rag_search", params, "knowledge_retrieval",
                f"train_rag_t{template_index:02d}", pattern,
            ))

    platform_knowledge = {
        "bilibili": ["B站分区规则", "B站流量分发机制"],
        "douyin": ["抖音推荐机制", "抖音内容审核规则"],
        "kuaishou": ["快手社区规则", "快手账号运营方法"],
        "xiaohongshu": ["小红书笔记规范", "小红书内容矩阵"],
    }
    for platform, topics in platform_knowledge.items():
        for idx, topic in enumerate(topics):
            add(training_case(
                f"train-rag-platform-{platform}-{idx}",
                f"只查资料库里的{topic}，不需要实时内容",
                "rag_search", {"query": topic, "platform": platform},
                "knowledge_retrieval", f"train_rag_platform_{platform}", "query_platform",
                platforms=[platform], source="handwritten_natural",
            ))

    existing_objects = ["已有样本", "上面的Evidence", "刚才的表格", "现成报告", "前文数据", "已有分析"]
    none_templates = [
        "基于{obj}总结三条结论",
        "把{obj}压缩成一段话",
        "只使用{obj}给建议，不新增检索",
        "润色{obj}，不要调用工具",
        "根据{obj}直接完成下一步",
        "沿用{obj}写一个简短摘要",
    ]
    for obj_index, obj in enumerate(existing_objects):
        for template_index, template in enumerate(none_templates):
            add(training_case(
                f"train-none-existing-{obj_index}-{template_index}",
                template.format(obj=obj), "none", {}, "no_new_evidence",
                f"train_none_existing_t{template_index}", "empty",
                source="handwritten_natural" if template_index == 2 else "programmatic_template",
            ))
    direct_tasks = [
        "解释什么是完播率", "写五个标题", "给三个选题方向", "把这段话改口语",
        "写一句开场白", "说明互动率含义", "给一个账号定位例子", "把报告改短",
        "直接列出复盘步骤", "给一段创作建议", "改写成面试口径", "生成一句摘要",
    ]
    direct_prefixes = ["不用搜索，", "不要查资料，", "直接", "这一步不需要工具，", "基于常识"]
    for task_index, task in enumerate(direct_tasks):
        for prefix_index, prefix in enumerate(direct_prefixes):
            add(training_case(
                f"train-none-direct-{task_index}-{prefix_index}",
                f"{prefix}{task}", "none", {}, "direct_creation_or_explanation",
                f"train_none_direct_p{prefix_index}", "empty",
            ))

    video_ids = [f"BV1{chr(65+i)}{chr(97+i)}4y1{chr(70+i)}7{chr(98+i)}{i%10}" for i in range(18)]
    asr_templates = [
        "转写这个公开视频 {url}",
        "提取 {url} 的字幕",
        "把链接里的口播整理出来：{url}",
        "识别这条B站视频的语音 {url}",
        "需要这条视频的逐字稿 {url}",
    ]
    for index, video_id in enumerate(video_ids):
        url = (
            f"https://b23.tv/{video_id}" if index % 3 == 0
            else f"https://www.bilibili.com/video/{video_id}"
        )
        for template_index, template in enumerate(asr_templates):
            add(training_case(
                f"train-asr-on-{index:02d}-{template_index}",
                template.format(url=url), "get_transcript", {"video_url": url},
                "transcript_with_url", f"train_asr_on_t{template_index}", "video_url",
            ))
        add(training_case(
            f"train-asr-off-{index:02d}", f"当前转写能力未配置，仍然有人要求转写 {url}",
            "none", {}, "transcript_unavailable", "train_asr_off_explicit", "empty",
            profile="asr_off", source="handwritten_boundary",
        ))
    missing_url_tasks = [
        "把那个视频转成文字", "给我视频字幕", "提取这条内容的口播", "生成逐字稿",
        "识别刚才视频里的语音", "链接还没发，先转写", "没有URL也帮我拿字幕",
    ]
    for index, text in enumerate(missing_url_tasks):
        add(training_case(
            f"train-asr-missing-{index:02d}", text, "none", {},
            "transcript_missing_url", f"train_asr_missing_{index}", "empty",
            source="handwritten_boundary",
        ))

    unsupported_platforms = ["douyin", "kuaishou", "xiaohongshu"]
    unsupported_topics = ["美妆", "游戏", "母婴", "汽车", "家居", "知识", "旅行", "穿搭"]
    unsupported_templates = [
        "搜索{platform}最近的{topic}热门视频",
        "从{platform}拿一批{topic}真实样本",
        "{platform}当前火的{topic}内容有哪些",
        "拉取{platform}{topic}近期榜单",
    ]
    aliases = {"douyin": "抖音", "kuaishou": "快手", "xiaohongshu": "小红书"}
    for platform in unsupported_platforms:
        for topic_index, topic in enumerate(unsupported_topics):
            for template_index, template in enumerate(unsupported_templates):
                text = template.format(platform=aliases[platform], topic=topic)
                add(training_case(
                    f"train-unsupported-{platform}-{topic_index}-{template_index}",
                    text, "none", {}, "unsupported_realtime_platform",
                    f"train_unsupported_{platform}_t{template_index}", "empty",
                    platforms=[platform], source="programmatic_template",
                ))

    trend_ids = [f"BV1T{i:02d}4y1Z7q{i%10}" for i in range(20)]
    trend_templates = [
        "查{video_id}最近的播放趋势",
        "获取{video_id}的点赞变化曲线",
        "看看{video_id}过去一周热度",
        "分析{video_id}的涨粉历史",
    ]
    for id_index, video_id in enumerate(trend_ids):
        for template_index, template in enumerate(trend_templates):
            add(training_case(
                f"train-trend-off-{id_index:02d}-{template_index}",
                template.format(video_id=video_id), "none", {}, "trend_unavailable",
                f"train_trend_off_t{template_index}", "empty",
                source="handwritten_boundary" if template_index == 0 else "programmatic_template",
            ))
    for index, text in enumerate(["查这条视频的趋势", "看看最近播放变化", "没有BV号但先取热度曲线"]):
        add(training_case(
            f"train-trend-missing-{index}", text, "none", {}, "trend_missing_id",
            f"train_trend_missing_{index}", "empty", source="handwritten_boundary",
        ))

    # v4.1 is the single permitted category-level refinement. These examples were
    # written after dev/hard error analysis, without copying or paraphrasing any
    # frozen case. Holdout content was not inspected for this augmentation.
    non_bilibili_urls = [
        "https://www.youtube.com/watch?v=trainA01",
        "https://youtu.be/trainB02",
        "https://v.douyin.com/trainC03/",
        "https://www.xiaohongshu.com/explore/trainD04",
        "http://example.com/trainE05.mp4",
    ]
    unsupported_asr_templates = [
        "识别这个站外视频的语音 {url}",
        "把这个非B站链接转成文字 {url}",
        "从 {url} 提取字幕内容",
    ]
    for url_index, url in enumerate(non_bilibili_urls):
        for template_index, template in enumerate(unsupported_asr_templates):
            add(training_case(
                f"train-v41-asr-platform-{url_index}-{template_index}",
                template.format(url=url), "none", {}, "transcript_unsupported_platform",
                f"train_v41_asr_platform_t{template_index}", "empty",
                source="v4_1_error_category_augmentation",
            ))

    trend_missing_phrases = [
        "我没提供视频编号，只想先查播放曲线",
        "没有BV号，暂时不要调用趋势工具",
        "视频ID还未知，先跳过热度查询",
        "缺少具体视频标识，无法获取涨粉历史",
        "没给ID就不要猜，趋势这步先不做",
        "只说了想看趋势，但没有任何视频编号",
        "目标视频还没确定，不能查历史数据",
        "等我补充BV号后再取播放变化",
        "当前没有视频ID，别把趋势当关键词搜索",
        "没有标识符时直接跳过数据曲线调用",
    ]
    for index, text in enumerate(trend_missing_phrases):
        add(training_case(
            f"train-v41-trend-missing-{index}", text, "none", {},
            "trend_missing_id", f"train_v41_trend_missing_{index}", "empty",
            source="v4_1_error_category_augmentation",
        ))

    contrast_topics = ["非遗", "航拍", "法律科普", "求职", "园艺", "乐器", "跑步", "装修"]
    contrast_templates = [
        "我要真实视频作对照，不是知识库理论；先找B站{topic}近期样本",
        "研究B站{topic}时先拿当前案例，方法论放到后面",
        "这一步需要能打开的B站{topic}视频，不要只给框架",
    ]
    for topic_index, topic in enumerate(contrast_topics):
        for template_index, template in enumerate(contrast_templates):
            add(training_case(
                f"train-v41-search-contrast-{topic_index}-{template_index}",
                template.format(topic=topic), "search_videos",
                {"keyword": topic, "platforms": ["bilibili"]},
                "real_sample_search", f"train_v41_search_contrast_t{template_index}", "keyword",
                source="v4_1_error_category_augmentation",
            ))

    # Frozen product-input-style examples copied from project 2's public dev contract,
    # with their origin stated explicitly. They are not v4 holdout cases.
    frozen_inputs = [
        ("搜索B站美妆类热门视频", "search_videos", {"keyword": "美妆", "platforms": ["bilibili"]}),
        ("检索知识库中关于爆款公式的资料", "rag_search", {"query": "爆款公式"}),
        ("根据已有数据总结三个发现", "none", {}),
        ("不要调用工具，直接解释什么是完播率", "none", {}),
    ]
    for index, (text, tool, params) in enumerate(frozen_inputs):
        add(training_case(
            f"train-project2-frozen-{index}", text, tool, params,
            "project2_frozen_input", f"train_project2_frozen_{index}",
            "empty" if tool == "none" else ("query" if tool == "rag_search" else "keyword"),
            source="project2_frozen_product_input",
        ))
    return rows


def sharegpt_row(item: dict) -> dict:
    prompt = build_researcher_prompt(
        item["input"], item["platforms"], item["capabilities"], variant="contract"
    )
    answer = canonical_json({"tool": item["expected_tool"], "params": item["expected_params"]})
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ]
    }


def file_manifest(path: Path, cases: list[dict], *, participation: str) -> dict:
    return {
        "file_name": str(path.relative_to(DATA_ROOT)).replace("\\", "/"),
        "sample_count": len(cases),
        "sha256": sha256_file(path),
        "participation": participation,
        "distribution": distribution(cases),
        "creation_method": "deterministic Python templates plus handwritten boundary cases; no LLM",
    }


def build_train() -> None:
    lock = verify_eval_lock()
    cases = build_training_catalog()
    rng = random.Random(SEED)
    rng.shuffle(cases)

    by_intent: dict[str, list[dict]] = defaultdict(list)
    for item in cases:
        by_intent[item["intent_family"]].append(item)
    train_cases, validation_cases = [], []
    for family_cases in by_intent.values():
        split_at = max(1, round(len(family_cases) * 0.1)) if len(family_cases) >= 5 else 0
        validation_cases.extend(family_cases[:split_at])
        train_cases.extend(family_cases[split_at:])
    rng.shuffle(train_cases)
    rng.shuffle(validation_cases)

    raw_path = V4_ROOT / "raw" / "cases.json"
    train_cases_path = V4_ROOT / "processed" / "train_cases.json"
    validation_cases_path = V4_ROOT / "processed" / "validation_cases.json"
    train_path = V4_ROOT / "processed" / "train.json"
    validation_path = V4_ROOT / "processed" / "validation.json"
    write_json(raw_path, cases)
    write_json(train_cases_path, train_cases)
    write_json(validation_cases_path, validation_cases)
    write_json(train_path, [sharegpt_row(item) for item in train_cases])
    write_json(validation_path, [sharegpt_row(item) for item in validation_cases])

    manifest = {
        "schema_version": 1,
        "created_date": "2026-07-14",
        "seed": SEED,
        "base_model": "C:/Users/0/.cache/modelscope/Qwen/Qwen3-4B",
        "contract_snapshot": "schemas/project2_researcher_contract_20260714.json",
        "holdout_lock_combined_sha256": lock["combined_sha256"],
        "truthful_sources": [
            "programmatic_template",
            "handwritten_natural",
            "handwritten_boundary",
            "project2_frozen_product_input",
            "v4_1_error_category_augmentation",
        ],
        "not_used": [
            "MiMo-generated training data",
            "DeepSeek-generated training data",
            "other LLM batch generation",
            "online user data",
        ],
        "files": {
            "raw": file_manifest(raw_path, cases, participation="source catalog"),
            "train_cases": file_manifest(train_cases_path, train_cases, participation="sft train metadata"),
            "validation_cases": file_manifest(validation_cases_path, validation_cases, participation="in-distribution validation metadata"),
            "train": file_manifest(train_path, train_cases, participation="sft train"),
            "validation": file_manifest(validation_path, validation_cases, participation="in-distribution validation"),
        },
        "split_rules": [
            "v4 hard/holdout files were frozen and hashed before this training catalog was built",
            "training expression_family names use the train_ prefix and are disjoint from frozen eval families",
            "exact and normalized overlap is checked by scripts/validate_dataset.py",
            "v4 holdout results must not be used to add paraphrases to training",
        ],
    }
    write_json(V4_ROOT / "manifests" / "v4_manifest.json", manifest)
    print(json.dumps({
        "total": len(cases),
        "train": len(train_cases),
        "validation": len(validation_cases),
        "distribution": distribution(cases),
        "manifest": str(V4_ROOT / "manifests" / "v4_manifest.json"),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-eval", action="store_true")
    parser.add_argument("--build-train", action="store_true")
    parser.add_argument("--force", action="store_true", help="only for intentional pre-training eval regeneration")
    args = parser.parse_args()
    if not args.freeze_eval and not args.build_train:
        parser.error("choose --freeze-eval or --build-train")
    if args.freeze_eval:
        freeze_eval(force=args.force)
    if args.build_train:
        build_train()


if __name__ == "__main__":
    main()
