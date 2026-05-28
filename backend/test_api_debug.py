"""直接测试智谱 API 响应，绕过 LangChain"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4")
MODEL = os.getenv("LLM_MODEL", "glm-4.7")

messages = [
    {"role": "system", "content": "你是一个健身教练。用中文回答，简洁明了。"},
    {"role": "user", "content": "帮我制定一个简单的腿部训练计划"}
]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 测试1: 非流式
print("=== 非流式测试 ===")
payload = {"model": MODEL, "messages": messages, "temperature": 0.7, "stream": False}
try:
    resp = requests.post(f"{API_BASE}/chat/completions", json=payload, headers=headers, timeout=60)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    if "error" in data:
        print(f"Error: {data['error']}")
    elif "choices" in data:
        choice = data["choices"][0]
        msg = choice.get("message", {})
        print(f"content: {repr(msg.get('content', ''))[:200]}")
        print(f"tool_calls: {msg.get('tool_calls')}")
        print(f"finish_reason: {choice.get('finish_reason')}")
        print(f"usage: {data.get('usage', {})}")
    else:
        print(f"Unexpected response: {str(data)[:500]}")
except Exception as e:
    print(f"Exception: {e}")

# 测试2: 流式
print("\n=== 流式测试 ===")
payload["stream"] = True
try:
    resp = requests.post(f"{API_BASE}/chat/completions", json=payload, headers=headers, timeout=60, stream=True)
    print(f"Status: {resp.status_code}")
    chunk_count = 0
    content_total = ""
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        import json
        try:
            data = json.loads(data_str)
            delta = data["choices"][0].get("delta", {})
            content = delta.get("content", "")
            tool_calls = delta.get("tool_calls")
            if chunk_count < 5 or content:
                print(f"  chunk[{chunk_count}]: content={repr(content)[:50]}, tool_calls={bool(tool_calls)}")
            if content:
                content_total += content
            chunk_count += 1
        except json.JSONDecodeError:
            print(f"  parse error: {data_str[:100]}")
    print(f"Total chunks: {chunk_count}, content length: {len(content_total)}")
    print(f"Content preview: {content_total[:200]}")
except Exception as e:
    print(f"Exception: {e}")
