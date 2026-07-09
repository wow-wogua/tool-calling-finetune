"""
直接调用 LLaMA Factory 训练，绕过 CLI。

用法:
    python scripts/train.py
"""

import sys
import os

# 强制离线模式，不触发任何下载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 确保在项目根目录运行
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen3_lora.yaml"

print("=" * 50)
print("开始 LoRA 微调训练")
print("=" * 50)
print(f"工作目录: {os.getcwd()}")
print(f"Python: {sys.version}")
print()

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
except Exception as e:
    print(f"PyTorch 导入失败: {e}")
    sys.exit(1)

try:
    from llamafactory.train.tuner import run_exp
    print("LLaMA Factory train 模块导入成功")
    print()
except Exception as e:
    print(f"LLaMA Factory 导入失败: {e}")
    sys.exit(1)

# 加载配置
try:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    print(f"配置加载成功: {config_path}")
    print(f"模型: {args.get('model_name_or_path')}")
    print(f"数据集: {args.get('dataset')}")
    print(f"输出目录: {args.get('output_dir')}")
    print()
except Exception as e:
    print(f"配置加载失败: {e}")
    sys.exit(1)

print("=" * 50)
print("开始训练...")
print("=" * 50)

try:
    run_exp(args)
    print("\n训练完成！")
except Exception as e:
    print(f"\n训练失败: {e}")
    import traceback
    traceback.print_exc()
