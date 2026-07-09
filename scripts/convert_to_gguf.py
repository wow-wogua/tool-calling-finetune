"""
将 HuggingFace 模型转为 GGUF 格式，用于 Ollama 部署。

用法:
    python scripts/convert_to_gguf.py

需要先安装:
    pip install llama-cpp-python gguf transformers torch
"""

import os
import sys
import json
import struct
from pathlib import Path

MODEL_PATH = "outputs/qwen3_dpo_tool_calling_merged"
OUTPUT_PATH = "outputs/qwen3-tool-calling.gguf"

def convert():
    print("=" * 50)
    print("转换 HuggingFace 模型为 GGUF 格式")
    print("=" * 50)

    # 检查模型文件
    model_dir = Path(MODEL_PATH)
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {MODEL_PATH}")
        sys.exit(1)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"❌ config.json 不存在")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"模型类型: {config.get('model_type', 'unknown')}")
    print(f"隐藏层: {config.get('hidden_size', 'unknown')}")
    print(f"层数: {config.get('num_hidden_layers', 'unknown')}")
    print()

    # 使用 llama-cpp-python 的转换工具
    try:
        from llama_cpp import llama_model_quantize
        print("llama-cpp-python 转换工具可用")
    except ImportError:
        print("❌ llama-cpp-python 未安装")
        sys.exit(1)

    # 方法：直接用 Ollama 导入（不需要手动转换）
    print("提示：Ollama 支持直接从 HuggingFace 导入，不需要手动转换。")
    print()
    print("步骤：")
    print("1. 启动 Ollama: ollama serve")
    print("2. 创建 Modelfile:")
    print(f'   FROM ./{MODEL_PATH}')
    print("3. 导入: ollama create qwen3-tool-calling -f Modelfile")
    print("4. 测试: ollama run qwen3-tool-calling")

    # 生成 Modelfile
    modelfile_content = f"""FROM ./{MODEL_PATH}

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

    modelfile_path = Path("outputs/Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"\n✅ Modelfile 已生成: {modelfile_path}")
    print(f"\n运行命令:")
    print(f"  cd D:\\internship\\tool-calling-finetune")
    print(f"  ollama serve")
    print(f"  ollama create qwen3-tool-calling -f outputs/Modelfile")
    print(f"  ollama run qwen3-tool-calling")


if __name__ == "__main__":
    convert()
