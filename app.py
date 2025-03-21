from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("userInput")
    try:
        response = requests.post(
            "https://api.deepseek.com",
            headers={
                "Authorization": "Bearer YOUR_API_KEY",
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
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
