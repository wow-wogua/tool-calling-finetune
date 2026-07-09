"""
导出微调后的模型：将 LoRA 权重合并到基座模型并导出为 HuggingFace 格式。

用法:
    python scripts/export_model.py \
        --base-model Qwen/Qwen3-4B \
        --adapter outputs/qwen3_dpo_tool_calling \
        --output outputs/qwen3_dpo_tool_calling_merged

导出后接入项目2的 LLM 网关:
    1. 用 FastAPI 部署导出的模型（serve_model.py，端口 8002）
    2. 在 viral-video-agent 中注册:
       model_registry.switch_to_finetuned(
           "researcher",
           model="qwen3-tool-calling",
           base_url="http://localhost:8002/v1",
           api_key="not-needed"
       )
"""

import argparse
import sys
from pathlib import Path


def merge_lora(base_model: str, adapter_path: str, output_path: str):
    """合并 LoRA 权重到基座模型。"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
    except ImportError:
        print("❌ 请安装依赖: pip install transformers peft torch")
        sys.exit(1)

    print(f"加载基座模型: {base_model}")
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    print(f"加载 LoRA 适配器: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("合并权重...")
    merged = model.merge_and_unload()

    print(f"保存合并后的模型: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_path)

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"\n{'='*50}")
    print(f"✅ 模型导出完成！")
    print(f"   输出路径: {output_path}")
    print(f"\n部署方式:")
    print(f"  FastAPI (推荐，轻量):")
    print(f"    python scripts/serve_model.py")
    print(f"    API 地址: http://localhost:8002/v1")
    print(f"\n接入项目2 LLM 网关:")
    print(f"    model_registry.switch_to_finetuned(")
    print(f"        'researcher',")
    print(f"        model='qwen3-tool-calling',")
    print(f"        base_url='http://localhost:8002/v1',")
    print(f"        api_key='not-needed'")
    print(f"    )")
    print(f"{'='*50}")


def create_modelfile(output_path: str):
    """生成 Modelfile（可选，用于其他部署方式）。"""
    modelfile = f"""FROM {output_path}

PARAMETER temperature 0
PARAMETER top_p 1.0
PARAMETER num_ctx 2048

SYSTEM \"\"\"你是研究员。根据任务选择工具和参数。

可用工具:
- search_videos(keyword, platforms, limit): 搜索视频数据
- rag_search(query, top_k): 从知识库检索参考文档
- get_transcript(video_url): 获取视频转写
- get_trend_data(video_id, platform): 获取视频历史趋势数据
- 无需工具: 输出 {{"tool": "none", "params": {{}}}}

输出JSON: {{"tool": "工具名", "params": {{"参数名": "值"}}}}
只输出JSON，不要其他内容。\"\"\"
"""
    modelfile_path = Path(output_path).parent / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    print(f"  Modelfile 已生成: {modelfile_path}")


def main():
    parser = argparse.ArgumentParser(description="导出微调后的模型")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B", help="基座模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--modelfile", action="store_true", help="同时生成 Modelfile（可选）")
    args = parser.parse_args()

    output_path = args.output or str(
        Path(args.adapter).parent / f"{Path(args.adapter).name}_merged"
    )

    merge_lora(args.base_model, args.adapter, output_path)

    if args.modelfile:
        create_modelfile(output_path)


if __name__ == "__main__":
    main()
