"""
DPO 训练脚本。在 SFT 后的 LoRA 适配器基础上做偏好优化。

v4/v4.1 没有执行 DPO；本入口仅保留给显式提供的新配置或历史复核。

用法:
    python scripts/train_dpo.py <config.yaml>
"""

import sys
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if len(sys.argv) <= 1:
    print("用法: python scripts/train_dpo.py <config.yaml>")
    print("v4/v4.1 未执行 DPO；v1-v3 配置已归档到 results/legacy/v1-v3/configs/。")
    sys.exit(2)
config_path = sys.argv[1]

print("=" * 50)
print("开始 DPO 偏好优化训练")
print("=" * 50)
print(f"工作目录: {os.getcwd()}")
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

try:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    print(f"配置加载成功: {config_path}")
    print(f"模型: {args.get('model_name_or_path')}")
    print(f"SFT 适配器: {args.get('adapter_name_or_path')}")
    print(f"数据集: {args.get('dataset')}")
    print(f"输出目录: {args.get('output_dir')}")
    print()
except Exception as e:
    print(f"配置加载失败: {e}")
    sys.exit(1)

print("=" * 50)
print("开始 DPO 训练...")
print("=" * 50)

try:
    run_exp(args)
    print("\nDPO 训练完成！")
except Exception as e:
    print(f"\nDPO 训练失败: {e}")
    import traceback
    traceback.print_exc()
