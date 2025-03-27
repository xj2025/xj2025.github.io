from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行化
app = Flask(__name__)

CORS(app)  # 允许所有域名（生产环境应限制）
# ========== 初始化配置 ==========
class ChatConfig:
    def __init__(self):
        self.api_key = "sk-f104aed04216406abce806380d6670a3"
        self.base_url = "https://api.deepseek.com"
        self.embedding_model = "BAAI/bge-small-zh-v1.5"
        self.retrieve_top_k = 3
        self.memory_window = 6
        self.model_cache_dir = "./model"  # 本地模型缓存路径

config = ChatConfig()
client = OpenAI(api_key=config.api_key, base_url=config.base_url)

# 初始化嵌入模型（从本地加载）
embedding_model = SentenceTransformer(
    config.embedding_model,
    device='cpu',
    cache_folder=config.model_cache_dir,  # 指定缓存目录
    local_files_only=True
)

# 加载知识库
with open('1.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# 构建FAISS索引
vectors = np.array([embedding_model.encode(doc["text"]) for doc in knowledge_base])
faiss_index = faiss.IndexFlatL2(vectors.shape[1])
faiss_index.add(vectors)

# ========== 核心功能 ==========
def retrieve_documents(query):
    query_vec = embedding_model.encode([query])
    _, indices = faiss_index.search(query_vec, config.retrieve_top_k)
    return [knowledge_base[idx]["text"] for idx in indices[0]]

def build_rag_prompt(query, docs):
    return f"""【参考知识】\n{chr(10).join(f'- {doc}' for doc in docs)}\n\n【用户问题】{query}"""

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("userInput")
        messages = data.get("messages", [])
        
        if not user_input:
            return jsonify({"error": "userInput is required"}), 400

        # RAG检索
        retrieved_docs = retrieve_documents(user_input)
        rag_prompt = build_rag_prompt(user_input, retrieved_docs)
        
        # 构建消息历史（保留原始对话流）
        messages.append({"role": "user", "content": rag_prompt})

        # 调用API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
        )

        # 处理响应
        ai_message = {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content
        }

        # 更新消息历史（原始用户输入+AI回复）
        updated_messages = messages[:-1] + [
            {"role": "user", "content": user_input},
            ai_message
        ]

        return jsonify({
            "reply": ai_message["content"],
            "updatedMessages": updated_messages,
            "relatedKnowledge": retrieved_docs  # 新增：返回检索到的知识
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
