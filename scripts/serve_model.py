"""
用 FastAPI 部署微调后的模型为 OpenAI 兼容 API。

用法:
    python scripts/serve_model.py

启动后 API 地址: http://localhost:8002/v1
接入项目2:
    model_registry.switch_to_finetuned(
        "researcher",
        model="qwen3-tool-calling",
        base_url="http://localhost:8002/v1",
        api_key="not-needed"
    )
"""

import os
import sys
import json
import re
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).parent.parent
BASE_MODEL_CANDIDATES = (
    PROJECT_ROOT / "outputs" / "Qwen3-4B-base",
    Path.home() / ".cache" / "modelscope" / "Qwen" / "Qwen3-4B",
    Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen3-4B",
)
DEFAULT_BASE_MODEL = next(
    (path for path in BASE_MODEL_CANDIDATES if (path / "config.json").exists()),
    BASE_MODEL_CANDIDATES[0],
)
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", str(DEFAULT_BASE_MODEL))
ADAPTER_PATH = os.getenv("FINETUNED_ADAPTER_PATH", "outputs/qwen3_lora_tool_calling_v4_1")
MERGED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH")
MODEL_PORT = int(os.getenv("FINETUNED_MODEL_PORT", "8002"))

print("=" * 50)
print("加载模型...")
print("=" * 50)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from researcher_prompt_v4 import build_researcher_prompt, is_complete_project2_prompt

load_path = MERGED_MODEL_PATH or BASE_MODEL_PATH
quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(
    load_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quantization,
)
model = (
    base_model
    if MERGED_MODEL_PATH
    else PeftModel.from_pretrained(base_model, ADAPTER_PATH)
)
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
model.eval()

print("模型加载完成！")
print(f"加载方式: {'merged model' if MERGED_MODEL_PATH else 'base + direct adapter'}")
print(f"模型路径: {load_path}")
if not MERGED_MODEL_PATH:
    print(f"Adapter: {ADAPTER_PATH}")
print(f"设备: {model.device}")
print()

PROMPT_VARIANT = os.getenv("RESEARCHER_PROMPT_VARIANT", "contract")


def _available_tools(prompt: str) -> set[str]:
    tools = set(re.findall(r"(?m)^- ([a-z_]+)(?:\(|:)", prompt or ""))
    return tools or {"search_videos", "rag_search", "get_transcript", "none"}


def _safe_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        return min(maximum, max(minimum, int(value)))
    except (TypeError, ValueError):
        return default


def _sanitize_response(response: str, prompt: str) -> str:
    """Apply project-2's frozen Schema before returning the model decision."""
    try:
        decision = json.loads((response or "").strip())
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"tool": "none", "params": {}}, ensure_ascii=False, separators=(",", ":"))
    if not isinstance(decision, dict):
        return json.dumps({"tool": "none", "params": {}}, ensure_ascii=False, separators=(",", ":"))
    tool = decision.get("tool")
    params = decision.get("params") if isinstance(decision.get("params"), dict) else {}
    available = _available_tools(prompt)
    if tool in (None, "none") or tool not in available:
        normalized = {"tool": "none", "params": {}}
    elif tool == "search_videos":
        platforms = params.get("platforms", ["bilibili"])
        if not isinstance(platforms, list):
            platforms = [platforms]
        normalized_platforms = [
            "bilibili" if str(item).lower() in {"bilibili", "b站", "哔哩哔哩"} else str(item).lower()
            for item in platforms
        ]
        if set(normalized_platforms) - {"bilibili"}:
            normalized = {"tool": "none", "params": {}}
        else:
            normalized = {
                "tool": tool,
                "params": {
                    "keyword": str(params.get("keyword", "")),
                    "platforms": ["bilibili"],
                    "limit": _safe_int(params.get("limit"), 10, 1, 20),
                },
            }
    elif tool == "rag_search":
        query = str(params.get("query", "")).strip()
        if not query:
            normalized = {"tool": "none", "params": {}}
        else:
            rag_params = {"query": query, "top_k": _safe_int(params.get("top_k"), 5, 1, 10)}
            platform = params.get("platform")
            if platform in {"bilibili", "douyin", "kuaishou", "xiaohongshu", "generic"}:
                rag_params["platform"] = platform
            normalized = {"tool": tool, "params": rag_params}
    elif tool == "get_transcript":
        url = str(params.get("video_url", "")).strip()
        valid_url = url in prompt and (
            url.startswith("https://www.bilibili.com/") or url.startswith("https://b23.tv/")
        )
        normalized = {"tool": tool, "params": {"video_url": url}} if valid_url else {"tool": "none", "params": {}}
    elif tool == "get_trend_data":
        video_id = str(params.get("video_id", "")).strip()
        normalized = (
            {"tool": tool, "params": {"video_id": video_id, "platform": "bilibili"}}
            if video_id and video_id in prompt
            else {"tool": "none", "params": {}}
        )
    else:
        normalized = {"tool": "none", "params": {}}
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))

# ── FastAPI 服务 ──
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ChatRequest(BaseModel):
    model: str = "qwen3-tool-calling"
    messages: list
    temperature: float = 0.0
    max_tokens: int = 256


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 提取用户消息
    user_msg = ""
    for msg in request.messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")

    # 项目2发送的是完整 Researcher Prompt；直接复用以保持训练/推理输入结构一致。
    # 对普通 OpenAI 客户端只传裸任务的情况，再补统一工具 schema。
    prompt = (
        user_msg
        if is_complete_project2_prompt(user_msg)
        else build_researcher_prompt(user_msg, variant=PROMPT_VARIANT)
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    response = _sanitize_response(raw_response, prompt)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = int(outputs[0].shape[0] - inputs["input_ids"].shape[1])

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "qwen3-tool-calling", "object": "model", "owned_by": "local"}
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "load_mode": "merged" if MERGED_MODEL_PATH else "base+direct-adapter",
        "adapter": None if MERGED_MODEL_PATH else ADAPTER_PATH,
        "prompt_variant": PROMPT_VARIANT,
    }

if __name__ == "__main__":
    print("=" * 50)
    print(f"启动 API 服务: http://localhost:{MODEL_PORT}/v1")
    print("=" * 50)
    print()
    print("接入项目2 LLM 网关:")
    print('  model_registry.switch_to_finetuned(')
    print('      "researcher",')
    print('      model="qwen3-tool-calling",')
    print('      base_url="http://localhost:8002/v1",')
    print('      api_key="not-needed"')
    print('  )')
    print()
    uvicorn.run(app, host="0.0.0.0", port=MODEL_PORT)
