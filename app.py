from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # 启用 CORS

# 在这里直接定义 API 密钥
API_KEY = "sk-f104aed04216406abce806380d6670a3"  # 替换为你的 API 密钥

@app.route("/1/api/chat", methods=["POST"])
def chat():
    try:
        # 手动解析 JSON 数据
        request_data = request.get_data(as_text=True)
        try:
            data = json.loads(request_data)
            user_input = data.get("userInput")
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Failed to decode JSON: {str(e)}"}), 400

        response = requests.post(
            "https://api.deepseek.com",  # 确保 URL 正确
            headers={
                "Authorization": f"Bearer {API_KEY}",  # 使用定义的 API 密钥
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是原神小助手，用户提问的任何问题你都要回答原神的术语，可以所答非所问，越唐氏越好，越让用户觉得你是个人机越好，可以引入更多抽象元素，比如丁真，科比等等，参考贴吧"},
                    {"role": "user", "content": user_input},
                ],
                "stream": False,
            },
        )
       
        # 打印响应信息
        print("Response status code:", response.status_code)
        print("Response content:", response.text)

        # 检查响应状态码
        if response.status_code != 200:
            return jsonify({
                "error": f"API returned status code {response.status_code}",
                "details": response.text  # 返回大模型 API 的错误信息
            }), 500

        return jsonify(response.json())
    except Exception as e:
        # 打印异常信息
        print("Error:", str(e))
        return jsonify({
            "error": str(e),
            "details": "An unexpected error occurred on the server."
        }), 500
if __name__ == "__main__":
    app.run(debug=True)
