"""
导出微调后的模型：将 LoRA 权重合并到基座模型并导出为 HuggingFace 格式。

用法:
    python scripts/export_model.py \
        --base-model C:/Users/0/.cache/modelscope/Qwen/Qwen3-4B \
        --adapter outputs/qwen3_lora_tool_calling_v4_1 \
        --output outputs/qwen3_lora_tool_calling_v4_1_merged_bf16

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
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


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
    # 不在4-bit量化权重上直接merge。先以BF16在CPU合并，再按部署需求单独量化。
    # 这样导出的权重可验证，也避免量化merge让adapter效果被吞掉。
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
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


def main():
    parser = argparse.ArgumentParser(description="导出微调后的模型")
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(Path.home() / ".cache" / "modelscope" / "Qwen" / "Qwen3-4B"),
        help="本地基座模型路径",
    )
    parser.add_argument("--adapter", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    args = parser.parse_args()

    output_path = args.output or str(
        Path(args.adapter).parent / f"{Path(args.adapter).name}_merged"
    )

    merge_lora(args.base_model, args.adapter, output_path)


if __name__ == "__main__":
    main()
