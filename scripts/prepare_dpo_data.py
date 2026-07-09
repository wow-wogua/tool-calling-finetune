"""
生成 DPO 偏好数据。

DPO 数据格式：每条包含 chosen（好的回答）和 rejected（坏的回答）。
重点生成 rag_search vs none 的边界 case。

用法:
    python scripts/prepare_dpo_data.py
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

SYSTEM_PROMPT = """你是研究员。根据任务选择工具和参数。

可用工具:
- search_videos(keyword, platforms, limit): 搜索视频数据。keyword=搜索关键词，platforms=平台列表，limit=返回数量
- rag_search(query, top_k): 从知识库检索参考文档。query=检索内容，top_k=返回数量
- get_transcript(video_url): 获取视频转写。video_url=视频链接
- get_trend_data(video_id, platform): 获取视频历史趋势数据。video_id=视频ID，platform=平台名
- 无需工具: 如果任务不需要调用任何工具，输出 {"tool": "none", "params": {}}

输出JSON: {"tool": "工具名", "params": {"参数名": "值"}}
只输出JSON，不要其他内容。"""

# ── DPO 偏好对 ──
# 格式：(instruction, chosen_tool, rejected_tool)
# chosen 是正确的工具选择，rejected 是常见的错误选择

DPO_PAIRS = [
    # rag_search 应该调但容易被判为 none 的 case
    ("抖音的推荐算法是什么原理", "rag_search", "none"),
    ("B站的流量是怎么分配的", "rag_search", "none"),
    ("怎么分析竞品账号", "rag_search", "none"),
    ("短视频选题有什么方法论", "rag_search", "none"),
    ("怎么用AIDA模型写脚本", "rag_search", "none"),
    ("怎么计算视频的互动率", "rag_search", "none"),
    ("视频封面怎么设计更吸引人", "rag_search", "none"),
    ("短视频标题有什么撰写技巧", "rag_search", "none"),
    ("怎么提高短视频的完播率", "rag_search", "none"),
    ("短视频的发布时间怎么选", "rag_search", "none"),
    ("抖音上什么内容容易被限流", "rag_search", "none"),
    ("B站的推荐机制和抖音有什么不同", "rag_search", "none"),
    ("短视频运营有哪些常见的方法论", "rag_search", "none"),
    ("怎么做短视频的竞品分析", "rag_search", "none"),
    ("短视频的内容矩阵怎么搭建", "rag_search", "none"),
    ("怎么判断一个视频是不是真的爆了", "rag_search", "none"),
    ("短视频的数据分析框架是什么", "rag_search", "none"),
    ("视频的健康度指标有哪些", "rag_search", "none"),
    ("怎么写一个吸引人的短视频开头", "rag_search", "none"),
    ("快手和抖音的算法有什么区别", "rag_search", "none"),

    # none 应该不调但容易被判为 rag_search 的 case
    ("帮我写一个短视频脚本大纲", "none", "rag_search"),
    ("给我几个选题建议", "none", "rag_search"),
    ("帮我总结一下分析结果", "none", "rag_search"),
    ("根据已有数据给出运营建议", "none", "rag_search"),
    ("帮我设计一个视频封面的文案", "none", "rag_search"),
    ("帮我起一个吸引人的标题", "none", "rag_search"),
    ("怎么让视频上热门", "none", "rag_search"),
    ("短视频账号怎么涨粉", "none", "rag_search"),
    ("我的视频播放量很低怎么办", "none", "rag_search"),
    ("现在做什么类型的短视频最火", "none", "rag_search"),

    # search_videos 应该调但容易被判为 none 的 case
    ("看看最近有什么好看的视频", "search_videos", "none"),
    ("帮我瞅瞅B站有啥火的", "search_videos", "none"),
    ("最近有什么热门内容", "search_videos", "none"),

    # none 应该不调但容易被判为 search_videos 的 case
    ("帮我推荐几个好看的电影", "none", "search_videos"),
    ("最近有什么新闻", "none", "search_videos"),
    ("你觉得做美食视频好还是游戏视频好", "none", "search_videos"),

    # rag_search vs search_videos 的区分
    ("有没有关于短视频运营的文档", "rag_search", "search_videos"),
    ("检索一下之前做过的分析报告", "rag_search", "search_videos"),
    ("有没有竞品分析的框架可以参考", "rag_search", "search_videos"),
    ("搜索B站美妆区的热门视频", "search_videos", "rag_search"),
    ("找一下最近的科技区热门", "search_videos", "rag_search"),
    ("分析B站游戏区的爆款", "search_videos", "rag_search"),
]


def make_dpo_item(instruction: str, chosen_tool: str, rejected_tool: str) -> dict:
    """生成一条 DPO 数据。"""
    # chosen 回答
    if chosen_tool == "none":
        chosen = '{"tool": "none", "params": {}}'
    elif chosen_tool == "rag_search":
        query = instruction.replace("有没有", "").replace("怎么", "").replace("什么", "").replace("？", "").replace("?", "").strip()[:20]
        chosen = json.dumps({"tool": "rag_search", "params": {"query": query, "top_k": 5}}, ensure_ascii=False)
    elif chosen_tool == "search_videos":
        chosen = json.dumps({"tool": "search_videos", "params": {"keyword": "", "platforms": ["bilibili"]}}, ensure_ascii=False)
    else:
        chosen = json.dumps({"tool": chosen_tool, "params": {}}, ensure_ascii=False)

    # rejected 回答
    if rejected_tool == "none":
        rejected = '{"tool": "none", "params": {}}'
    elif rejected_tool == "rag_search":
        query = instruction[:20]
        rejected = json.dumps({"tool": "rag_search", "params": {"query": query, "top_k": 5}}, ensure_ascii=False)
    elif rejected_tool == "search_videos":
        rejected = json.dumps({"tool": "search_videos", "params": {"keyword": "", "platforms": ["bilibili"]}}, ensure_ascii=False)
    else:
        rejected = json.dumps({"tool": rejected_tool, "params": {}}, ensure_ascii=False)

    human_value = f"{SYSTEM_PROMPT}\n\n任务: {instruction}"

    return {
        "conversations": [
            {"from": "human", "value": human_value},
        ],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
    }


def main():
    random.seed(42)

    # 生成 DPO 数据
    dpo_data = []
    for instruction, chosen_tool, rejected_tool in DPO_PAIRS:
        item = make_dpo_item(instruction, chosen_tool, rejected_tool)
        dpo_data.append(item)

    random.shuffle(dpo_data)

    # 分割 train/eval（90/10，DPO 数据量小，多留训练集）
    split_idx = max(1, int(len(dpo_data) * 0.9))
    train_data = dpo_data[:split_idx]
    eval_data = dpo_data[split_idx:]

    # 保存
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "dpo_train.json"
    eval_path = processed_dir / "dpo_eval.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    # 更新 dataset_info.json
    info_path = PROJECT_ROOT / "data" / "dataset_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    data_dir = PROJECT_ROOT / "data"
    info["tool_calling_dpo_train"] = {
        "file_name": str(train_path.relative_to(data_dir)).replace("\\", "/"),
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
    info["tool_calling_dpo_eval"] = {
        "file_name": str(eval_path.relative_to(data_dir)).replace("\\", "/"),
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"DPO 数据生成完成:")
    print(f"  训练集: {len(train_data)} 条")
    print(f"  评估集: {len(eval_data)} 条")
    print(f"  保存到: {train_path}")
    print(f"  dataset_info.json 已更新")


if __name__ == "__main__":
    main()
