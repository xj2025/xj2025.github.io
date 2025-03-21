from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 启用 CORS

# 在这里直接定义 API 密钥
API_KEY = "sk-f104aed04216406abce806380d6670a3"  # 替换为你的 API 密钥
@app.route("/")
def home():
    return "Welcome to the Flask API!"
@app.route("/api/chat", methods=["GET"])  # 使用 GET 请求
def chat():
    try:
        # 从查询参数中获取用户输入
        user_input = request.args.get("userInput")
        if not user_input:
            return jsonify({"error": "userInput is required"}), 400

        # 打印用户输入
        print(f"用户输入: {user_input}")

        # 调用大模型 API
        response = requests.post(
            "https://api.deepseek.com",  # 确保 URL 正确
            headers={
                "Authorization": f"Bearer {API_KEY}",  # 使用定义的 API 密钥
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是政治老师"},
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
    app.run(host="0.0.0.0", port=10000)  # 确保端口为 10000
