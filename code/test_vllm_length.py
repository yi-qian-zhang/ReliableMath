# test_vllm_length.py
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

# 要求模型生成很长的输出
response = client.chat.completions.create(
    model="deepseek-r1-distill-qwen-7b",
    messages=[
        {"role": "user", "content": "Count from 1 to 1000, one number per line."}
    ],
    max_tokens=8192,
    temperature=0.0
)

print(f"Response length: {len(response.choices[0].message.content)}")
print(f"Finish reason: {response.choices[0].finish_reason}")
print(f"Last line: {response.choices[0].message.content.split(chr(10))[-1]}")
