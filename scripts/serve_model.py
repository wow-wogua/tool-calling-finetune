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
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "outputs/qwen3_dpo_tool_calling_merged")
MODEL_PORT = int(os.getenv("FINETUNED_MODEL_PORT", "8002"))

print("=" * 50)
print("加载模型...")
print("=" * 50)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from researcher_prompt import build_researcher_prompt

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()

print("模型加载完成！")
print(f"设备: {model.device}")
print()

PROMPT_VARIANT = os.getenv("RESEARCHER_PROMPT_VARIANT", "base")

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


def _is_full_researcher_prompt(text: str) -> bool:
    """项目2已经发送完整 Researcher Prompt 时，不再重复包一层。"""
    return "可用工具:" in text and "输出JSON" in text and "任务:" in text

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 提取用户消息
    user_msg = ""
    for msg in request.messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")

    # 项目2发送的是完整 Researcher Prompt；直接复用以保持训练/推理输入结构一致。
    # 对普通 OpenAI 客户端只传裸任务的情况，再补统一工具 schema。
    prompt = user_msg if _is_full_researcher_prompt(user_msg) else build_researcher_prompt(user_msg, PROMPT_VARIANT)
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

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
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
