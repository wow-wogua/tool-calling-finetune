"""
评测微调后的工具调用准确率。

复用项目2（viral-video-agent）的 BFCL 评测用例，
对比基座模型 vs 微调模型的工具调用准确率。

用法:
    # 评测基座模型
    python scripts/evaluate.py --model Qwen/Qwen3-4B

    # 评测已合并的微调模型
    python scripts/evaluate.py --model outputs/qwen3_dpo_tool_calling_merged

    # 如果只有 LoRA adapter，--model 传基座模型，--adapter 传 adapter 目录
    python scripts/evaluate.py --model Qwen/Qwen3-4B --adapter outputs/qwen3_lora_tool_calling

    # 对比两个完整模型
    python scripts/evaluate.py --compare Qwen/Qwen3-4B outputs/qwen3_dpo_tool_calling_merged

环境变量:
    HF_ENDPOINT=https://hf-mirror.com  (国内镜像加速)
"""

import json
import sys
import os
import argparse
import re
from pathlib import Path
from datetime import datetime

# 离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Windows 编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent

# ── 评测用例（与 viral-video-agent/src/eval/bfcl_eval.py 一致，扩展到 50 条）──
TEST_CASES = [
    # ── search_videos 工具（20 条）──
    {"input": "搜索B站美妆类视频", "expected_tool": "search_videos", "expected_params": {"keyword": "美妆", "platforms": ["bilibili"]}},
    {"input": "搜索B站美食类视频TOP10", "expected_tool": "search_videos", "expected_params": {"keyword": "美食", "platforms": ["bilibili"], "limit": 10}},
    {"input": "帮我找B站最近的科技区热门视频", "expected_tool": "search_videos", "expected_params": {"keyword": "科技", "platforms": ["bilibili"]}},
    {"input": "看看最近有什么热门视频", "expected_tool": "search_videos", "expected_params": {"keyword": "", "platforms": ["bilibili"]}},
    {"input": "分析B站游戏区最近的爆款", "expected_tool": "search_videos", "expected_params": {"keyword": "游戏", "platforms": ["bilibili"]}},
    {"input": "找一下B站生活区点赞最高的视频", "expected_tool": "search_videos", "expected_params": {"keyword": "生活", "platforms": ["bilibili"]}},
    {"input": "获取B站热门排行榜前20名", "expected_tool": "search_videos", "expected_params": {"keyword": "", "platforms": ["bilibili"], "limit": 20}},
    {"input": "帮我找B站音乐区的热门MV", "expected_tool": "search_videos", "expected_params": {"keyword": "音乐", "platforms": ["bilibili"]}},
    {"input": "分析B站舞蹈区最近的爆款视频", "expected_tool": "search_videos", "expected_params": {"keyword": "舞蹈", "platforms": ["bilibili"]}},
    {"input": "看看B站知识区有什么热门内容", "expected_tool": "search_videos", "expected_params": {"keyword": "知识", "platforms": ["bilibili"]}},
    {"input": "搜索抖音美食区热门", "expected_tool": "search_videos", "expected_params": {"keyword": "美食", "platforms": ["douyin"]}},
    {"input": "找B站影视区最近的热门解说", "expected_tool": "search_videos", "expected_params": {"keyword": "影视", "platforms": ["bilibili"]}},
    {"input": "分析B站动画区的热门番剧", "expected_tool": "search_videos", "expected_params": {"keyword": "动画", "platforms": ["bilibili"]}},
    {"input": "看看B站汽车区有什么测评视频", "expected_tool": "search_videos", "expected_params": {"keyword": "汽车", "platforms": ["bilibili"]}},
    {"input": "找B站体育区最近的赛事集锦", "expected_tool": "search_videos", "expected_params": {"keyword": "体育", "platforms": ["bilibili"]}},
    {"input": "分析B站搞笑区的热门视频", "expected_tool": "search_videos", "expected_params": {"keyword": "搞笑", "platforms": ["bilibili"]}},
    {"input": "帮我瞅瞅B站生活区最近有啥火的", "expected_tool": "search_videos", "expected_params": {"keyword": "生活", "platforms": ["bilibili"]}},
    {"input": "看看B站和抖音的科技区热门", "expected_tool": "search_videos", "expected_params": {"keyword": "科技", "platforms": ["bilibili", "douyin"]}},
    {"input": "B站时尚区最近有什么爆款", "expected_tool": "search_videos", "expected_params": {"keyword": "时尚", "platforms": ["bilibili"]}},
    {"input": "搜索快手萌宠区热门视频", "expected_tool": "search_videos", "expected_params": {"keyword": "萌宠", "platforms": ["kuaishou"]}},

    # ── rag_search 工具（10 条）──
    {"input": "检索知识库中关于爆款公式的文档", "expected_tool": "rag_search", "expected_params": {"query": "爆款公式"}},
    {"input": "有没有历史爆款分析报告可以参考", "expected_tool": "rag_search", "expected_params": {"query": "爆款分析报告"}},
    {"input": "查一下抖音的算法规则", "expected_tool": "rag_search", "expected_params": {"query": "抖音算法规则"}},
    {"input": "知识库里有没有关于AIDA框架的内容", "expected_tool": "rag_search", "expected_params": {"query": "AIDA框架"}},
    {"input": "帮我找一下短视频运营的方法论", "expected_tool": "rag_search", "expected_params": {"query": "短视频运营方法论"}},
    {"input": "有没有关于竞品分析的文档", "expected_tool": "rag_search", "expected_params": {"query": "竞品分析"}},
    {"input": "检索一下2024年的行业趋势报告", "expected_tool": "rag_search", "expected_params": {"query": "2024年行业趋势"}},
    {"input": "知识库里有没有关于钩子设计的内容", "expected_tool": "rag_search", "expected_params": {"query": "钩子设计"}},
    {"input": "B站推荐算法是什么原理", "expected_tool": "rag_search", "expected_params": {"query": "B站推荐算法原理"}},
    {"input": "怎么分析竞品账号，有没有框架可以用", "expected_tool": "rag_search", "expected_params": {"query": "竞品分析框架"}},

    # ── get_transcript 工具（5 条）──
    {"input": "把这个视频的文案转写出来", "expected_tool": "get_transcript", "expected_params": {}},
    {"input": "获取这个视频的字幕内容", "expected_tool": "get_transcript", "expected_params": {}},
    {"input": "帮我转写一下这个视频的口播文案", "expected_tool": "get_transcript", "expected_params": {}},
    {"input": "提取视频里的文案，我想看看话术", "expected_tool": "get_transcript", "expected_params": {}},
    {"input": "把视频里说的话整理成文字", "expected_tool": "get_transcript", "expected_params": {}},

    # ── get_trend_data 工具（5 条）──
    {"input": "这个视频的播放量趋势怎么样", "expected_tool": "get_trend_data", "expected_params": {}},
    {"input": "看看这个视频最近的数据变化", "expected_tool": "get_trend_data", "expected_params": {}},
    {"input": "查一下这个抖音视频的涨粉趋势", "expected_tool": "get_trend_data", "expected_params": {}},
    {"input": "这个视频是什么时候开始爆的", "expected_tool": "get_trend_data", "expected_params": {}},
    {"input": "对比一下这个视频上周和这周的数据", "expected_tool": "get_trend_data", "expected_params": {}},

    # ── 不需要工具的场景（10 条）──
    {"input": "帮我总结一下分析结果", "expected_tool": None, "expected_params": {}},
    {"input": "根据已有数据，给出运营建议", "expected_tool": None, "expected_params": {}},
    {"input": "解释一下什么是爆款视频", "expected_tool": None, "expected_params": {}},
    {"input": "帮我写一个短视频脚本大纲", "expected_tool": None, "expected_params": {}},
    {"input": "什么是爆款视频", "expected_tool": None, "expected_params": {}},
    {"input": "给我几个选题建议", "expected_tool": None, "expected_params": {}},
    {"input": "怎么提高完播率", "expected_tool": None, "expected_params": {}},
    {"input": "你好，你是做什么的", "expected_tool": None, "expected_params": {}},
    {"input": "短视频行业前景怎么样", "expected_tool": None, "expected_params": {}},
    {"input": "帮我分析一下", "expected_tool": None, "expected_params": {}},
]


SYSTEM_PROMPT = """你是研究员。根据任务选择工具和参数。

可用工具:
- search_videos(keyword, platforms, limit): 搜索视频数据。keyword=搜索关键词，platforms=平台列表，limit=返回数量
- rag_search(query, top_k): 从知识库检索参考文档。query=检索内容，top_k=返回数量
- get_transcript(video_url): 获取视频转写。video_url=视频链接
- get_trend_data(video_id, platform): 获取视频历史趋势数据。video_id=视频ID，platform=平台名
- 无需工具: 如果任务不需要调用任何工具，输出 {"tool": "none", "params": {}}

输出JSON: {"tool": "工具名", "params": {"参数名": "值"}}
只输出JSON，不要其他内容。"""


def parse_model_output(text: str) -> dict:
    """解析模型输出为 JSON，兼容多种格式。"""
    clean = text.strip()

    # 去掉 thinking tokens（Qwen3 的 <think>...</think>）
    clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL).strip()

    # 去掉 markdown 代码块
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # 尝试直接解析
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 部分
    match = re.search(r'\{[^{}]*"tool"[^{}]*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"tool": "parse_error", "params": {}}


def check_tool_match(actual: dict, expected_tool: str | None, expected_params: dict) -> dict:
    """检查工具调用是否正确。"""
    actual_tool = actual.get("tool")

    # 工具名匹配
    if expected_tool is None:
        tool_correct = actual_tool in (None, "none")
    else:
        tool_correct = actual_tool == expected_tool

    # 参数匹配（只检查 expected_params 中有的字段）
    params_correct = True
    if tool_correct and expected_params:
        actual_params = actual.get("params", {})
        for key, expected_val in expected_params.items():
            actual_val = actual_params.get(key)
            if isinstance(expected_val, list):
                if not isinstance(actual_val, list) or set(expected_val) != set(actual_val):
                    params_correct = False
                    break
            elif actual_val != expected_val:
                params_correct = False
                break

    return {
        "tool_correct": tool_correct,
        "params_correct": params_correct,
        "fully_correct": tool_correct and params_correct,
    }


def load_cases(path: str | None = None) -> list[dict]:
    """加载评测用例。不传路径时使用内置 50 条 BFCL 用例。"""
    if not path:
        return TEST_CASES
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if isinstance(cases, dict):
        cases = cases.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"评测用例为空或格式错误: {path}")
    return cases


def run_eval_local(model_path: str, adapter_path: str = None, test_cases: list[dict] = None) -> dict:
    """本地模型评测（需要 transformers + torch）。"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("❌ 请安装依赖: pip install transformers torch")
        sys.exit(1)

    print(f"  加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    if adapter_path:
        # 加载 LoRA 适配器（与训练时相同的 4-bit 量化）
        from peft import PeftModel
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )

    model.eval()

    results = []
    correct_tool = 0
    correct_full = 0

    test_cases = test_cases or TEST_CASES

    for i, case in enumerate(test_cases):
        prompt = f"{SYSTEM_PROMPT}\n\n任务: {case['input']}"

        messages = [{"role": "user", "content": prompt}]
        # 关闭 thinking 模式：在 prompt 末尾加 /no_think
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        actual = parse_model_output(response)
        match = check_tool_match(actual, case["expected_tool"], case.get("expected_params", {}))

        if match["tool_correct"]:
            correct_tool += 1
        if match["fully_correct"]:
            correct_full += 1

        status = "✅" if match["fully_correct"] else ("⚠️" if match["tool_correct"] else "❌")
        print(f"  {status} [{i+1:02d}] {case['input'][:30]}... → 期望={case['expected_tool']}, 实际={actual.get('tool')}")

        results.append({
            "case": i + 1,
            "input": case["input"],
            "expected_tool": case["expected_tool"],
            "actual_tool": actual.get("tool"),
            "tool_correct": match["tool_correct"],
            "fully_correct": match["fully_correct"],
        })

    total = len(test_cases)
    return {
        "model": model_path,
        "adapter": adapter_path,
        "total": total,
        "tool_accuracy": correct_tool / total * 100,
        "full_accuracy": correct_full / total * 100,
        "tool_correct": correct_tool,
        "full_correct": correct_full,
        "details": results,
    }


def run_eval_api(model_name: str, base_url: str = None, api_key: str = None, test_cases: list[dict] = None) -> dict:
    """API 模式评测（用于对比 MiMo 等 API 模型）。"""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 请安装 openai: pip install openai")
        sys.exit(1)

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key

    client = OpenAI(**client_kwargs)

    results = []
    correct_tool = 0
    correct_full = 0

    test_cases = test_cases or TEST_CASES

    for i, case in enumerate(test_cases):
        prompt = f"{SYSTEM_PROMPT}\n\n任务: {case['input']}"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            text = response.choices[0].message.content
        except Exception as e:
            print(f"  ❌ [{i+1:02d}] API 错误: {e}")
            text = ""

        actual = parse_model_output(text)
        match = check_tool_match(actual, case["expected_tool"], case.get("expected_params", {}))

        if match["tool_correct"]:
            correct_tool += 1
        if match["fully_correct"]:
            correct_full += 1

        status = "✅" if match["fully_correct"] else ("⚠️" if match["tool_correct"] else "❌")
        print(f"  {status} [{i+1:02d}] {case['input'][:30]}... → 期望={case['expected_tool']}, 实际={actual.get('tool')}")

        results.append({
            "case": i + 1,
            "input": case["input"],
            "expected_tool": case["expected_tool"],
            "actual_tool": actual.get("tool"),
            "tool_correct": match["tool_correct"],
            "fully_correct": match["fully_correct"],
        })

    total = len(test_cases)
    return {
        "model": model_name,
        "total": total,
        "tool_accuracy": correct_tool / total * 100,
        "full_accuracy": correct_full / total * 100,
        "tool_correct": correct_tool,
        "full_correct": correct_full,
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description="评测工具调用准确率")
    parser.add_argument("--model", type=str, help="模型路径或名称")
    parser.add_argument("--adapter", type=str, help="LoRA 适配器路径")
    parser.add_argument("--api", action="store_true", help="使用 API 模式评测")
    parser.add_argument("--base-url", type=str, help="API base URL")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--cases", type=str, help="外部评测用例 JSON 路径；不填则使用内置 50 条 BFCL 用例")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"), help="对比两个模型")
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print("工具调用准确率评测")
    print(f"{'='*50}")
    test_cases = load_cases(args.cases)
    print(f"评测用例数: {len(test_cases)}")
    print(f"  search_videos: {sum(1 for c in test_cases if c['expected_tool'] == 'search_videos')}")
    print(f"  rag_search: {sum(1 for c in test_cases if c['expected_tool'] == 'rag_search')}")
    print(f"  get_transcript: {sum(1 for c in test_cases if c['expected_tool'] == 'get_transcript')}")
    print(f"  get_trend_data: {sum(1 for c in test_cases if c['expected_tool'] == 'get_trend_data')}")
    print(f"  无需工具: {sum(1 for c in test_cases if c['expected_tool'] is None)}")
    if args.cases:
        print(f"  用例文件: {args.cases}")
    print()

    all_results = {}

    if args.compare:
        # 对比模式
        for model in args.compare:
            print(f"\n评测模型: {model}")
            result = run_eval_api(model, args.base_url, args.api_key, test_cases) if args.api else run_eval_local(model, test_cases=test_cases)
            all_results[model] = result
            print(f"\n  工具准确率: {result['tool_accuracy']:.1f}% ({result['tool_correct']}/{result['total']})")
            print(f"  完全准确率: {result['full_accuracy']:.1f}% ({result['full_correct']}/{result['total']})")

        print(f"\n{'='*50}")
        print("对比结果:")
        for model, result in all_results.items():
            print(f"  {model}: 工具 {result['tool_accuracy']:.1f}% / 完全 {result['full_accuracy']:.1f}%")
        print(f"{'='*50}")

    elif args.model:
        # 单模型评测
        print(f"\n评测模型: {args.model}")
        if args.api:
            result = run_eval_api(args.model, args.base_url, args.api_key, test_cases)
        else:
            result = run_eval_local(args.model, args.adapter, test_cases)
        all_results[args.model] = result

        print(f"\n{'='*50}")
        print(f"工具准确率: {result['tool_accuracy']:.1f}% ({result['tool_correct']}/{result['total']})")
        print(f"完全准确率: {result['full_accuracy']:.1f}% ({result['full_correct']}/{result['total']})")
        print(f"{'='*50}")

    else:
        print("请指定 --model 或 --compare")
        print("示例:")
        print("  python scripts/evaluate.py --model Qwen/Qwen3-4B")
        print("  python scripts/evaluate.py --model outputs/qwen3_dpo_tool_calling_merged")
        print("  python scripts/evaluate.py --model Qwen/Qwen3-4B --adapter outputs/qwen3_lora_tool_calling")
        print("  python scripts/evaluate.py --api --model mimo-v2.5-pro --base-url http://localhost:8000/v1")
        return

    # 保存结果
    output_path = args.output or str(PROJECT_ROOT / "results" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_cases": len(test_cases),
            "cases_file": args.cases or "builtin",
            "results": all_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
