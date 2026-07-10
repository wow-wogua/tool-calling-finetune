"""
将原始数据转为 LLaMA Factory 的 SFT 训练格式。

用法:
    python scripts/prepare_dataset.py

输入:
    data/raw/handcrafted.json  (手写样例)
    data/raw/generated_v2.json (程序化模板生成 + 边界补充后的合并数据，优先)
    data/raw/generated.json    (第一版程序化模板数据，fallback)

输出:
    data/processed/train.json      (80% 训练集)
    data/processed/eval.json       (20% 评估集)
    data/dataset_info.json         (LLaMA Factory 数据集配置)
"""

import json
import random
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent


# ── Researcher 工具 schema 与输出格式（和项目2对齐；项目2会额外注入任务与目标平台）──
SYSTEM_PROMPT = """你是研究员。根据任务选择工具和参数。

可用工具:
- search_videos(keyword, platforms, limit): 搜索视频数据。keyword=搜索关键词，platforms=平台列表，limit=返回数量
- rag_search(query, top_k): 从知识库检索参考文档。query=检索内容，top_k=返回数量
- get_transcript(video_url): 获取视频转写。video_url=视频链接
- get_trend_data(video_id, platform): 获取视频历史趋势数据。video_id=视频ID，platform=平台名
- 无需工具: 如果任务不需要调用任何工具，输出 {"tool": "none", "params": {}}

输出JSON: {"tool": "工具名", "params": {"参数名": "值"}}
只输出JSON，不要其他内容。"""


def load_raw_data() -> list:
    """加载所有原始数据，并按 instruction 去重。

    generated_v2.json 是在 generated.json 基础上补充边界样例得到的合并文件，
    其中可能包含 handcrafted.json 的样例。这里优先保留先加载的手写样例。
    """
    all_data = []

    # 加载手写样例
    handcrafted_path = PROJECT_ROOT / "data" / "raw" / "handcrafted.json"
    if handcrafted_path.exists():
        with open(handcrafted_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"  加载手写样例: {len(data)} 条")

    # 加载生成的数据（优先用 v2，fallback 到 v1）
    generated_path = PROJECT_ROOT / "data" / "raw" / "generated_v2.json"
    if not generated_path.exists():
        generated_path = PROJECT_ROOT / "data" / "raw" / "generated.json"
    if generated_path.exists():
        with open(generated_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
            print(f"  加载生成数据: {len(data)} 条")

    seen_instructions = set()
    unique_data = []
    duplicate_count = 0
    for item in all_data:
        instruction = item.get("instruction", "").strip()
        if instruction and instruction in seen_instructions:
            duplicate_count += 1
            continue
        if instruction:
            seen_instructions.add(instruction)
        unique_data.append(item)

    if duplicate_count:
        print(f"  去重: 移除重复 instruction {duplicate_count} 条，保留 {len(unique_data)} 条")

    return unique_data


def parse_output(output):
    """解析 output 字段，兼容 str 和 dict 两种格式。"""
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output
    return output


def convert_to_sft(item: dict) -> dict:
    """将单条数据转为 LLaMA Factory SFT 格式。

    LLaMA Factory 格式:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
    instruction = item.get("instruction", "")
    output = parse_output(item.get("output", ""))

    # output 可能是 dict 或 str
    if isinstance(output, dict):
        output_str = json.dumps(output, ensure_ascii=False)
    else:
        output_str = str(output)

    # 构建 human 消息：system prompt + 用户查询
    human_value = f"{SYSTEM_PROMPT}\n\n任务: {instruction}"

    return {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": output_str},
        ]
    }


def split_dataset(data: list, eval_ratio: float = 0.2, seed: int = 42) -> tuple:
    """按比例分割训练集和评估集，保证各工具类型均衡。"""
    random.seed(seed)

    # 按工具类型分组
    by_tool = {}
    for item in data:
        output = parse_output(item.get("output", ""))
        tool = output.get("tool", "unknown") if isinstance(output, dict) else "unknown"
        if tool not in by_tool:
            by_tool[tool] = []
        by_tool[tool].append(item)

    train_data = []
    eval_data = []

    for tool, items in by_tool.items():
        random.shuffle(items)
        split_idx = max(1, int(len(items) * (1 - eval_ratio)))
        train_data.extend(items[:split_idx])
        eval_data.extend(items[split_idx:])

    random.shuffle(train_data)
    random.shuffle(eval_data)

    return train_data, eval_data


def generate_dataset_info(train_path: str, eval_path: str) -> dict:
    """生成 LLaMA Factory 的 dataset_info.json。"""
    return {
        "tool_calling_train": {
            "file_name": train_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        },
        "tool_calling_eval": {
            "file_name": eval_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        },
    }


def main():
    print(f"\n{'='*50}")
    print("准备 LLaMA Factory 训练数据")
    print(f"{'='*50}\n")

    # 加载原始数据
    raw_data = load_raw_data()
    if not raw_data:
        print("❌ 没有找到原始数据！请先运行 generate_data.py")
        return

    # 转换格式
    print(f"\n转换 {len(raw_data)} 条数据为 SFT 格式...")
    sft_data = [convert_to_sft(item) for item in raw_data]

    # 统计各工具类型
    tool_counts = {}
    for item in raw_data:
        output = parse_output(item.get("output", ""))
        tool = output.get("tool", "unknown") if isinstance(output, dict) else "unknown"
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    print(f"\n各工具类型分布:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count} 条")

    # 分割数据集
    train_raw, eval_raw = split_dataset(raw_data, eval_ratio=0.2)
    train_sft = [convert_to_sft(item) for item in train_raw]
    eval_sft = [convert_to_sft(item) for item in eval_raw]

    print(f"\n分割结果:")
    print(f"  训练集: {len(train_sft)} 条 ({len(train_sft)/len(sft_data)*100:.0f}%)")
    print(f"  评估集: {len(eval_sft)} 条 ({len(eval_sft)/len(sft_data)*100:.0f}%)")

    # 保存
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.json"
    eval_path = processed_dir / "eval.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_sft, f, ensure_ascii=False, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_sft, f, ensure_ascii=False, indent=2)

    # 生成 dataset_info.json（路径相对于 data/ 目录）
    data_dir = PROJECT_ROOT / "data"
    dataset_info = generate_dataset_info(
        str(train_path.relative_to(data_dir)).replace("\\", "/"),
        str(eval_path.relative_to(data_dir)).replace("\\", "/"),
    )
    info_path = PROJECT_ROOT / "data" / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            existing_info = json.load(f)
    else:
        existing_info = {}
    existing_info.update(dataset_info)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ 数据准备完成！")
    print(f"   训练集: {train_path}")
    print(f"   评估集: {eval_path}")
    print(f"   配置:   {info_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
