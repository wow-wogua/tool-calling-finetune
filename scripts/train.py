"""
直接调用 LLaMA Factory 训练，绕过 CLI。

用法:
    python scripts/train.py configs/qwen3_lora_v4.yaml
"""

import os
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# 强制离线模式，不触发任何下载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 确保在项目根目录运行
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen3_lora_v4.yaml"

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


def sha256_file(path: str) -> str | None:
    target = Path(path)
    if not target.exists():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if "_v4" in str(args.get("dataset", "")):
    print("运行 v4 数据冻结、Schema 与泄漏检查...")
    subprocess.run([sys.executable, "scripts/validate_dataset.py"], check=True)
    packages = {}
    for package in ("torch", "bitsandbytes", "transformers", "peft", "datasets", "accelerate", "llamafactory"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    environment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "packages": packages,
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file("data/v4/manifests/v4_manifest.json"),
        "holdout_lock_sha256": sha256_file("data/v4/manifests/holdout_lock.json"),
        "seed": args.get("seed"),
        "data_seed": args.get("data_seed"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3) if torch.cuda.is_available() else None,
    }
    environment_path = Path("results/v4/training_environment.json")
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"环境摘要已保存: {environment_path}")

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
    sys.exit(1)
