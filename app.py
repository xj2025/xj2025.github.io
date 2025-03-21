from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI  # 导入 OpenAI SDK

app = Flask(__name__)
CORS(app)  # 启用 CORS

# 在这里直接定义 API 密钥
API_KEY = "sk-f104aed04216406abce806380d6670a3"  # 替换为你的 API 密钥

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

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

        # 调用 DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是玩贴吧抽象的，梗很多，比如丁真，科比等，你说话很抽象，但是不会骂人"},
                {"role": "user", "content": user_input},
            ],
            stream=False,
        )

        # 打印响应信息
        print("Response content:", response)

        # 提取大模型的回复
        reply = response.choices[0].message.content

        # 返回响应
        return jsonify({
            "choices": [
                {
                    "message": {
                        "content": reply
                    }
                }
            ]
        })
    except Exception as e:
        # 打印异常信息
        print("Error:", str(e))
        return jsonify({
            "error": str(e),
            "details": "An unexpected error occurred on the server."
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # 确保端口为 10000
