import os
import requests

# 1. 设置 Key
api_key = "sk-bc496d84758d40459a1c72c4896bab98"

# 2. 准备请求
url = "https://api.deepseek.com/v1/embeddings"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
data = {
    "model": "deepseek-embed",
    "input": "你好"
}

# 3. 发送请求
print("正在测试 DeepSeek Embedding API...")
try:
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("✅ 成功！API 可用。")
        print(response.json())
    else:
        print(f"❌ 失败！状态码: {response.status_code}")
        print(f"错误详情: {response.text}")
except Exception as e:
    print(f"❌ 发生异常: {e}")