"""
补充高质量的 rag_search 和 none 训练数据，提升边界区分能力。

注意: generated_v2.json 是 handcrafted.json + generated.json + 边界补充样例的合并文件。
prepare_dataset.py 会在加载 handcrafted.json 和 generated_v2.json 后按 instruction 去重。

用法:
    python scripts/fix_rag_data.py
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ── 高质量 rag_search 样例（不依赖"知识库"关键词，靠语义判断）──
RAG_SAMPLES = [
    # 平台规则/算法（需要知识库里的平台文档）
    {"instruction": "抖音的推荐算法是什么原理", "output": {"tool": "rag_search", "params": {"query": "抖音推荐算法原理", "top_k": 5}}},
    {"instruction": "B站的流量是怎么分配的", "output": {"tool": "rag_search", "params": {"query": "B站流量分配机制", "top_k": 5}}},
    {"instruction": "快手和抖音的算法有什么区别", "output": {"tool": "rag_search", "params": {"query": "快手抖音算法区别", "top_k": 5}}},
    {"instruction": "抖音上什么内容容易被限流", "output": {"tool": "rag_search", "params": {"query": "抖音限流规则", "top_k": 5}}},
    {"instruction": "B站的权重机制是怎样的", "output": {"tool": "rag_search", "params": {"query": "B站权重机制", "top_k": 5}}},
    {"instruction": "短视频平台的审核规则有哪些", "output": {"tool": "rag_search", "params": {"query": "短视频审核规则", "top_k": 5}}},
    {"instruction": "抖音的完播率怎么计算的", "output": {"tool": "rag_search", "params": {"query": "抖音完播率计算方法", "top_k": 5}}},
    {"instruction": "B站的推荐机制和抖音有什么不同", "output": {"tool": "rag_search", "params": {"query": "B站抖音推荐机制对比", "top_k": 5}}},
    # 行业方法论（需要知识库里的方法论文档）
    {"instruction": "短视频选题有什么方法论", "output": {"tool": "rag_search", "params": {"query": "短视频选题方法论", "top_k": 5}}},
    {"instruction": "怎么做一个爆款视频的选题策划", "output": {"tool": "rag_search", "params": {"query": "爆款视频选题策划方法", "top_k": 5}}},
    {"instruction": "短视频的脚本结构怎么设计", "output": {"tool": "rag_search", "params": {"query": "短视频脚本结构设计", "top_k": 5}}},
    {"instruction": "怎么用AIDA模型写短视频脚本", "output": {"tool": "rag_search", "params": {"query": "AIDA模型短视频脚本", "top_k": 5}}},
    {"instruction": "短视频运营有哪些常见的方法论", "output": {"tool": "rag_search", "params": {"query": "短视频运营方法论", "top_k": 5}}},
    {"instruction": "怎么分析一个账号的运营策略", "output": {"tool": "rag_search", "params": {"query": "账号运营策略分析方法", "top_k": 5}}},
    {"instruction": "短视频的内容矩阵怎么搭建", "output": {"tool": "rag_search", "params": {"query": "短视频内容矩阵搭建", "top_k": 5}}},
    {"instruction": "怎么做短视频的竞品分析", "output": {"tool": "rag_search", "params": {"query": "短视频竞品分析方法", "top_k": 5}}},
    # 历史报告/案例（需要知识库里的历史数据）
    {"instruction": "之前有没有分析过美妆区的爆款", "output": {"tool": "rag_search", "params": {"query": "美妆区爆款分析", "top_k": 3}}},
    {"instruction": "有没有美食类视频的分析案例", "output": {"tool": "rag_search", "params": {"query": "美食类视频分析案例", "top_k": 5}}},
    {"instruction": "之前做过哪些行业的视频分析", "output": {"tool": "rag_search", "params": {"query": "行业视频分析历史", "top_k": 5}}},
    {"instruction": "有没有关于直播带货的分析报告", "output": {"tool": "rag_search", "params": {"query": "直播带货分析报告", "top_k": 5}}},
    # 数据分析方法（需要知识库里的分析框架）
    {"instruction": "怎么计算视频的互动率", "output": {"tool": "rag_search", "params": {"query": "视频互动率计算方法", "top_k": 5}}},
    {"instruction": "视频的健康度指标有哪些", "output": {"tool": "rag_search", "params": {"query": "视频健康度指标", "top_k": 5}}},
    {"instruction": "怎么判断一个视频是不是真的爆了", "output": {"tool": "rag_search", "params": {"query": "视频爆款判断标准", "top_k": 5}}},
    {"instruction": "短视频的数据分析框架是什么", "output": {"tool": "rag_search", "params": {"query": "短视频数据分析框架", "top_k": 5}}},
    # 创作技巧（需要知识库里的创作指南）
    {"instruction": "视频封面怎么设计更吸引人", "output": {"tool": "rag_search", "params": {"query": "视频封面设计技巧", "top_k": 5}}},
    {"instruction": "短视频标题有什么撰写技巧", "output": {"tool": "rag_search", "params": {"query": "短视频标题撰写技巧", "top_k": 5}}},
    {"instruction": "视频开头的钩子怎么设计", "output": {"tool": "rag_search", "params": {"query": "视频开头钩子设计", "top_k": 5}}},
    {"instruction": "怎么提高短视频的完播率", "output": {"tool": "rag_search", "params": {"query": "短视频完播率提升方法", "top_k": 5}}},
    {"instruction": "短视频的发布时间怎么选", "output": {"tool": "rag_search", "params": {"query": "短视频发布时间选择", "top_k": 5}}},
    {"instruction": "怎么写一个吸引人的短视频开头", "output": {"tool": "rag_search", "params": {"query": "短视频开头写法", "top_k": 5}}},
]

# ── 高质量 none 样例（容易被误判为需要工具的边界 case）──
NONE_SAMPLES = [
    # 看起来像搜索但其实是通用建议
    {"instruction": "给我推荐几个B站美妆区的UP主", "output": {"tool": "none", "params": {}}},
    {"instruction": "最近有什么好看的电影推荐", "output": {"tool": "none", "params": {}}},
    {"instruction": "帮我看看最近有什么新闻", "output": {"tool": "none", "params": {}}},
    {"instruction": "你觉得做美食视频好还是游戏视频好", "output": {"tool": "none", "params": {}}},
    {"instruction": "现在做什么类型的短视频最火", "output": {"tool": "none", "params": {}}},
    # 看起来像检索但其实是创作请求
    {"instruction": "帮我写一个关于美食的短视频脚本", "output": {"tool": "none", "params": {}}},
    {"instruction": "帮我起一个吸引人的标题", "output": {"tool": "none", "params": {}}},
    {"instruction": "帮我设计一个视频封面的文案", "output": {"tool": "none", "params": {}}},
    {"instruction": "帮我写一个视频的简介", "output": {"tool": "none", "params": {}}},
    # 看起来像数据分析但其实是建议
    {"instruction": "我的视频播放量很低怎么办", "output": {"tool": "none", "params": {}}},
    {"instruction": "怎么让视频上热门", "output": {"tool": "none", "params": {}}},
    {"instruction": "短视频账号怎么涨粉", "output": {"tool": "none", "params": {}}},
    {"instruction": "怎么做才能让视频被更多人看到", "output": {"tool": "none", "params": {}}},
    # 看起来像趋势查询但其实是通用问题
    {"instruction": "短视频行业的前景怎么样", "output": {"tool": "none", "params": {}}},
    {"instruction": "现在做短视频还来得及吗", "output": {"tool": "none", "params": {}}},
    {"instruction": "短视频和直播哪个更有前途", "output": {"tool": "none", "params": {}}},
    # 看起来像工具调用但其实是闲聊
    {"instruction": "你能帮我做什么", "output": {"tool": "none", "params": {}}},
    {"instruction": "你都会什么", "output": {"tool": "none", "params": {}}},
    {"instruction": "介绍一下你自己", "output": {"tool": "none", "params": {}}},
    {"instruction": "你是谁开发的", "output": {"tool": "none", "params": {}}},
]


def load_existing_data():
    """加载已有的原始数据。"""
    all_data = []
    for fname in ["handcrafted.json", "generated.json"]:
        path = PROJECT_ROOT / "data" / "raw" / fname
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_data.extend(json.load(f))
    return all_data


def main():
    # 加载已有数据
    existing = load_existing_data()
    existing_instructions = {item["instruction"] for item in existing}

    # 添加不重复的新样例
    new_data = []
    for item in RAG_SAMPLES + NONE_SAMPLES:
        if item["instruction"] not in existing_instructions:
            new_data.append(item)
            existing_instructions.add(item["instruction"])

    print(f"已有数据: {len(existing)} 条")
    print(f"新增数据: {len(new_data)} 条 (rag={len(RAG_SAMPLES)}, none={len(NONE_SAMPLES)})")

    # 合并
    all_data = existing + new_data
    random.shuffle(all_data)

    # 保存到新的生成文件
    output_path = PROJECT_ROOT / "data" / "raw" / "generated_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"总计: {len(all_data)} 条")
    print(f"保存到: {output_path}")

    # 统计
    tool_counts = {}
    for item in all_data:
        output = item.get("output", {})
        if isinstance(output, str):
            output = json.loads(output)
        tool = output.get("tool", "unknown")
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print(f"\n各工具分布:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()
