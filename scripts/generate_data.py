"""
批量生成工具调用训练数据（本地生成，不需要 API）。

用法:
    python scripts/generate_data.py

输出:
    data/raw/generated.json  (约 240 条去重后的模板化样例)
"""

import json
import random
import sys
from pathlib import Path

# Windows GBK 兼容
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent

random.seed(42)

# ── 模板库 ──

SEARCH_TEMPLATES = [
    # 标准格式
    ("分析{platform}{category}区的爆款视频", {"category": True, "platform": True}),
    ("搜索{platform}{category}区热门", {"category": True, "platform": True}),
    ("找{platform}{category}区最近火的视频", {"category": True, "platform": True}),
    ("看看{platform}{category}区有什么热门内容", {"category": True, "platform": True}),
    ("获取{platform}热门排行榜前{limit}名", {"limit": True, "platform": True}),
    ("帮我找{platform}{category}区的热门{type}", {"category": True, "platform": True, "type": True}),
    # 口语化
    ("帮我瞅瞅{platform}{category}区最近有啥火的", {"category": True, "platform": True}),
    ("{platform}{category}区最近啥最火", {"category": True, "platform": True}),
    ("看看{platform}最近有啥好看的", {"platform": True}),
    ("{platform}{category}区有啥爆款没", {"category": True, "platform": True}),
    # 简短
    ("{category}热门", {"category": True}),
    ("{category}区爆款", {"category": True}),
    ("{platform}{category}热门", {"category": True, "platform": True}),
    # 详细
    ("分析{platform}{category}区最近一周点赞最高的视频", {"category": True, "platform": True}),
    ("搜索{platform}{category}区本月播放量最高的视频", {"category": True, "platform": True}),
    ("找{platform}最近三天{category}区的爆款", {"category": True, "platform": True}),
    # 跨平台
    ("搜索{platform}和{platform2}{category}区的热门", {"category": True, "platform": True, "platform2": True}),
    ("对比{platform}和{platform2}{category}区的爆款", {"category": True, "platform": True, "platform2": True}),
    # 指定数量
    ("找{platform}{category}区热门视频TOP{limit}", {"category": True, "platform": True, "limit": True}),
    ("获取{platform}{category}区前{limit}名视频", {"category": True, "platform": True, "limit": True}),
]

CATEGORIES = ["美妆", "美食", "游戏", "科技", "生活", "音乐", "舞蹈", "知识", "影视", "动画", "汽车", "体育", "搞笑", "时尚", "萌宠", "家居", "数码", "职场", "情感", "娱乐"]
PLATFORMS = [("B站", "bilibili"), ("抖音", "douyin"), ("快手", "kuaishou")]
LIMITS = [5, 10, 15, 20, 30, 50]
TYPES = ["视频", "MV", "解说", "测评", "教程", "合集"]


def gen_search_videos():
    results = []
    for _ in range(80):
        template, fields = random.choice(SEARCH_TEMPLATES)
        category = random.choice(CATEGORIES)
        plat_text, plat_id = random.choice(PLATFORMS)

        query = template.replace("{category}", category).replace("{platform}", plat_text)
        params = {"keyword": category, "platforms": [plat_id]}

        if fields.get("limit"):
            limit = random.choice(LIMITS)
            query = query.replace("{limit}", str(limit))
            params["limit"] = limit

        if fields.get("type"):
            t = random.choice(TYPES)
            query = query.replace("{type}", t)

        if fields.get("platform2"):
            plat2_text, plat2_id = random.choice([p for p in PLATFORMS if p[1] != plat_id])
            query = query.replace("{platform2}", plat2_text)
            params["platforms"] = [plat_id, plat2_id]

        # 无关键词的情况
        if not fields.get("category"):
            params["keyword"] = ""

        results.append({
            "instruction": query,
            "output": {"tool": "search_videos", "params": params}
        })
    return results


RAG_TEMPLATES = [
    "有没有关于{topic}的方法论",
    "{topic}是什么原理",
    "怎么{action}，有没有框架可以用",
    "知识库里有没有关于{topic}的内容",
    "检索一下{topic}的文档",
    "有没有{topic}的报告可以参考",
    "帮我找一下{topic}的资料",
    "查一下{topic}",
    "{topic}有什么好的做法",
    "怎么理解{topic}",
    "有没有{topic}的案例",
    "检索知识库中关于{topic}的文档",
    "{topic}的最佳实践是什么",
    "有没有{topic}相关的参考",
    "帮我查一下{topic}的规则",
]

RAG_TOPICS = [
    "短视频运营", "爆款公式", "AIDA框架", "竞品分析", "B站推荐算法",
    "抖音流量分配", "完播率优化", "视频封面设计", "标题撰写技巧",
    "短视频脚本结构", "用户画像分析", "内容选题策略", "发布时间选择",
    "互动率提升", "涨粉策略", "短视频SEO", "直播运营", "短视频带货",
    "内容矩阵搭建", "账号定位", "钩子设计", "黄金3秒", "评论区运营",
    "粉丝留存", "短视频数据分析", "行业趋势", "平台规则", "违规避坑",
    "短视频变现", "品牌合作",
]
RAG_ACTIONS = [
    "分析竞品账号", "提高完播率", "设计视频封面", "写短视频脚本",
    "做内容选题", "分析用户画像", "提升互动率", "做短视频SEO",
    "搭建内容矩阵", "做账号定位", "设计钩子", "运营评论区",
]


def gen_rag_search():
    results = []
    for _ in range(50):
        template = random.choice(RAG_TEMPLATES)
        topic = random.choice(RAG_TOPICS)
        action = random.choice(RAG_ACTIONS)

        query = template.replace("{topic}", topic).replace("{action}", action)
        top_k = random.choice([3, 5, 5, 5, 5, 10])  # 5 最常见

        params = {"query": topic, "top_k": top_k}
        results.append({
            "instruction": query,
            "output": {"tool": "rag_search", "params": params}
        })
    return results


TRANSCRIPT_TEMPLATES = [
    "帮我转写这个视频的口播内容",
    "提取视频里的文案",
    "把视频里说的话整理成文字",
    "获取这个视频的字幕内容",
    "帮我转写一下这个视频",
    "这个视频的脚本是什么，帮我转写",
    "我想看看这个视频的话术",
    "视频里的口播文案帮我整理一下",
    "转写这个视频的台词",
    "帮我把视频内容转成文字",
    "获取视频的语音转文字结果",
    "这个视频讲了什么，帮我转写出来",
    "提取视频中的口播文案",
    "帮我记录一下这个视频的内容",
    "视频字幕提取",
    "转写视频 https://www.bilibili.com/video/BV1xx411c7mD",
    "帮我转写 https://www.bilibili.com/video/BV1xx411c7mD 的内容",
    "提取这个视频的文案 https://www.douyin.com/video/123456",
    "把这个视频转成文字 https://www.bilibili.com/video/BV1yy411c7mE",
    "视频内容转写",
    "帮我看看这个视频说了啥",
    "这个视频的文案帮我抄一下",
    "口播内容转文字",
    "帮我提取视频里的关键话术",
    "视频里的人说了什么",
    "帮我把视频语音转成文字稿",
    "获取视频的完整文案",
    "这个视频的旁白帮我整理一下",
    "转写视频对话内容",
    "帮我记录视频中的要点",
    "视频解说词转写",
    "提取视频的讲述内容",
    "帮我整理视频的文字版",
    "视频内容文字化",
    "这个视频有没有字幕，帮我获取",
    "帮我转写视频的语音内容",
    "视频里说了啥帮我写出来",
    "提取视频脚本文案",
    "帮我把视频转成文稿",
    "视频内容转文本",
]

VIDEO_URLS = [
    "VIDEO_URL",
    "VIDEO_URL",
    "VIDEO_URL",
    "https://www.bilibili.com/video/BV1xx411c7mD",
    "https://www.bilibili.com/video/BV1yy411c7mE",
]


def gen_get_transcript():
    results = []
    for i in range(40):
        query = TRANSCRIPT_TEMPLATES[i % len(TRANSCRIPT_TEMPLATES)]
        url = random.choice(VIDEO_URLS)

        results.append({
            "instruction": query,
            "output": {"tool": "get_transcript", "params": {"video_url": url}}
        })
    return results


TREND_TEMPLATES = [
    "这个视频的播放量趋势怎么样",
    "看看这个视频最近的数据变化",
    "查一下这个视频的涨粉趋势",
    "这个视频是什么时候开始爆的",
    "对比一下这个视频上周和这周的数据",
    "看看视频的数据走势",
    "这个视频的互动数据变化",
    "播放量最近涨了吗",
    "这个视频的趋势数据帮我看看",
    "视频最近的热度变化",
    "什么时候开始起量的",
    "看看这个视频的增长曲线",
    "视频数据最近怎么样",
    "这个视频的流量趋势",
    "帮我查一下视频的历史数据",
    "视频播放量变化趋势",
    "最近数据有什么波动",
    "这个视频是突然爆的还是慢慢涨的",
    "视频热度什么时候到顶的",
    "看看数据有没有下滑",
    "视频涨粉速度怎么样",
    "这个视频的数据表现如何",
    "帮我分析一下视频的趋势",
    "视频播放量什么时候破的百万",
    "最近一周数据变化大吗",
    "视频互动率趋势",
    "看看评论量的变化",
    "视频分享数据趋势",
    "这个视频的数据健康吗",
    "帮我看看视频的长期表现",
    "查一下BV1xx411c7mD的趋势",
    "看看BV1yy411c7mE最近的数据",
    "抖音视频数据趋势",
    "快手视频热度变化",
    "这个视频的数据有没有异常",
    "视频完播率趋势怎么样",
    "帮我对比视频上月和本月数据",
    "视频数据有没有周期性波动",
    "看看视频的自然流量趋势",
    "视频推荐流量变化",
]

VIDEO_IDS = ["VIDEO_ID", "VIDEO_ID", "VIDEO_ID", "BV1xx411c7mD", "BV1yy411c7mE"]
TREND_PLATFORMS = ["bilibili", "bilibili", "bilibili", "douyin", "kuaishou"]


def gen_get_trend_data():
    results = []
    for i in range(40):
        query = TREND_TEMPLATES[i % len(TREND_TEMPLATES)]
        vid = random.choice(VIDEO_IDS)
        plat = random.choice(TREND_PLATFORMS)

        results.append({
            "instruction": query,
            "output": {"tool": "get_trend_data", "params": {"video_id": vid, "platform": plat}}
        })
    return results


NO_TOOL_TEMPLATES = [
    # 概念解释
    "什么是爆款视频",
    "完播率是什么意思",
    "互动率怎么计算",
    "什么是短视频的黄金3秒",
    "AIDA模型是什么",
    "什么是内容矩阵",
    "短视频的定义是什么",
    "什么是用户画像",
    "CTR是什么指标",
    "什么是短视频SEO",
    # 创作建议
    "给我几个B站美妆区的选题建议",
    "怎么提高完播率",
    "短视频标题怎么写吸引人",
    "怎么设计视频封面",
    "视频开头怎么做才抓人",
    "给我一些创作灵感",
    "怎么做短视频才能火",
    "怎么提升视频质量",
    "短视频拍摄有什么技巧",
    "怎么让视频更容易上热门",
    # 总结分析
    "帮我总结一下分析结果",
    "根据已有数据，给出运营建议",
    "帮我写一个短视频脚本大纲",
    "帮我整理一下今天的分析要点",
    "总结一下刚才的发现",
    "根据这些数据给个结论",
    "帮我归纳一下主要观点",
    "整理一下分析报告的要点",
    # 闲聊
    "你好",
    "你是谁",
    "你是做什么的",
    "你能帮我做什么",
    "今天天气怎么样",
    "推荐几本书",
    "你是AI吗",
    # 通用问题
    "短视频行业前景怎么样",
    "做短视频赚钱吗",
    "短视频和直播哪个好",
    "怎么入行短视频",
    "短视频运营需要什么技能",
    # 边界case
    "帮我分析一下",
    "总结一下",
    "给我看看数据",
    "帮我做个报告",
    "分析一下这个",
]


def gen_no_tool():
    results = []
    for query in NO_TOOL_TEMPLATES:
        results.append({
            "instruction": query,
            "output": {"tool": "none", "params": {}}
        })
    return results


def main():
    print(f"\n{'='*50}")
    print("生成工具调用训练数据")
    print(f"{'='*50}\n")

    all_data = []
    generators = [
        ("search_videos", gen_search_videos, 80),
        ("rag_search", gen_rag_search, 50),
        ("get_transcript", gen_get_transcript, 40),
        ("get_trend_data", gen_get_trend_data, 40),
        ("no_tool", gen_no_tool, 50),
    ]

    for name, gen_fn, expected in generators:
        data = gen_fn()
        all_data.extend(data)
        print(f"  ✅ {name}: {len(data)} 条 (预期 {expected})")

    # 去重（按 instruction）
    seen = set()
    unique_data = []
    for item in all_data:
        if item["instruction"] not in seen:
            seen.add(item["instruction"])
            unique_data.append(item)
    print(f"\n  去重后: {len(unique_data)} 条 (原 {len(all_data)} 条)")

    # 打乱顺序
    random.shuffle(unique_data)

    # 保存
    output_path = PROJECT_ROOT / "data" / "raw" / "generated.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)

    # 统计
    tool_counts = {}
    for item in unique_data:
        tool = item["output"]["tool"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print(f"\n{'='*50}")
    print(f"✅ 生成完成！共 {len(unique_data)} 条")
    print(f"   输出: {output_path}")
    print(f"\n   各场景分布:")
    for tool, count in sorted(tool_counts.items()):
        print(f"     {tool}: {count} 条")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
