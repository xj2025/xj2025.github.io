from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
import logging
from datetime import datetime
import sys
import io
#import locale
#locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 在文件最开头添加

# 其他现有代码保持不变...
# 强制设置系统默认编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
app = Flask(__name__)
# 精确配置CORS（必须放在所有路由定义前）
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://xj2025.github.io"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
app.config['JSON_AS_ASCII'] = False  # 允许非ASCII字符
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
# ========== 初始化日志配置 ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 初始化配置 ==========
class ChatConfig:
    def __init__(self):
        self.api_key = "sk-f104aed04216406abce806380d6670a3"
        self.base_url = "https://api.deepseek.com"
        self.embedding_model = "BAAI/bge-small-zh-v1.5"
        self.retrieve_top_k = 3
        self.memory_window = 6
        self.model_cache_dir = "./model"

config = ChatConfig()

# ========== 初始化组件 ==========
def initialize_components():
    """初始化所有关键组件并记录状态"""
    components = {}
    
    try:
        # 1. 初始化OpenAI客户端
        components['openai_client'] = OpenAI(api_key=config.api_key, base_url=config.base_url)
        logger.info("✅ OpenAI客户端初始化成功")
        
        # 2. 加载嵌入模型
        components['embedding_model'] = SentenceTransformer(
            config.embedding_model,
            device='cpu',
            cache_folder=config.model_cache_dir,
            local_files_only=True
        )
        logger.info("✅ 嵌入模型加载成功")
        
        # 3. 加载知识库
        if not os.path.exists('1.json'):
            raise FileNotFoundError("知识库文件 1.json 不存在")
            
        with open('1.json', 'r', encoding='utf-8') as f:
            components['knowledge_base'] = json.load(f)
        logger.info(f"✅ 知识库加载成功，共 {len(components['knowledge_base'])} 条数据")
        
        # 4. 构建FAISS索引
        vectors = np.array([components['embedding_model'].encode(doc["text"]) for doc in components['knowledge_base']])
        components['faiss_index'] = faiss.IndexFlatL2(vectors.shape[1])
        components['faiss_index'].add(vectors)
        logger.info("✅ FAISS索引构建成功")
        
        return components
    
    except Exception as e:
        logger.error(f"❌ 组件初始化失败: {str(e)}")
        raise

try:
    components = initialize_components()
    client = components['openai_client']
    embedding_model = components['embedding_model']
    knowledge_base = components['knowledge_base']
    faiss_index = components['faiss_index']
except Exception as e:
    logger.critical("⚠️ 服务启动失败，关键组件未初始化！")
    # 设置为None以便后续检查
    client = embedding_model = knowledge_base = faiss_index = None
# ========== 模型预热 ==========
@app.cli.command("warmup")
def warmup_command():
    """命令行预热模型"""
    warmup_models()
    print("✅ 模型预热完成")
# ========== 健康检查接口 ==========
@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "api_ready": client is not None,
        "model_ready": embedding_model is not None,
        "knowledge_base_ready": knowledge_base is not None,
        "faiss_ready": faiss_index is not None,
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status)

# ========== 核心功能 ==========
def retrieve_documents(query):
    """增强的检索函数，包含错误处理"""
    try:
        start_time = datetime.now()
        query_vec = embedding_model.encode([query])
        _, indices = faiss_index.search(query_vec, config.retrieve_top_k)
        results = [knowledge_base[idx]["text"] for idx in indices[0]]
        logger.info(f"检索完成: 查询='{query}', 耗时={(datetime.now()-start_time).total_seconds():.2f}s, 结果数={len(results)}")
        return results
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        return []

def build_rag_prompt(query, docs):
    """构建提示词，添加空值检查"""
    if not docs:
        logger.warning("检索结果为空，将使用空上下文")
        docs = ["未找到相关知识点"]
    return f"""【参考知识】\n{chr(10).join(f'- {doc}' for doc in docs)}\n\n【用户问题】{query}"""

@app.route("/api/chat", methods=["POST"])
def chat():
    # 1. 请求验证
    start_time = datetime.now()
    logger.info(f"收到新请求: {request.method} {request.path}")
    
    # 1.1 强制检查Content-Type
    content_type = request.content_type or ''
    if 'application/json' not in content_type.lower():
        logger.warning(f"无效的Content-Type: {content_type}")
        return jsonify({"error": "Content-Type必须是application/json"}), 400
    
    # 1.2 高级编码处理
    try:
        # 获取原始字节数据
        raw_data = request.get_data(cache=True)
        
        # 尝试多种编码解码
        encodings = ['utf-8', 'gbk', 'gb2312', 'big5']  # 常见中文编码
        decoded_data = None
        
        for encoding in encodings:
            try:
                decoded_data = raw_data.decode(encoding)
                # 验证是否是合法JSON
                data = json.loads(decoded_data)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
                
        if not decoded_data:
            raise ValueError("无法解码请求体")
            
    except Exception as e:
        logger.error(f"请求解析失败: {str(e)}")
        return jsonify({
            "error": "请求解析失败",
            "suggestion": "请确保使用UTF-8编码的合法JSON格式",
            "detail": str(e)
        }), 400
    
    # 2. 参数检查
    user_input = data.get("userInput")
    messages = data.get("messages", [])

    if not user_input:
        return jsonify({"error": "userInput是必填字段"}), 400

    # 3. 组件状态检查
    if not all([client, embedding_model, knowledge_base, faiss_index]):
        logger.error("服务组件未就绪")
        return jsonify({"error": "系统正在初始化，请稍后再试"}), 503

    try:
        # 4. RAG检索（添加中文处理）
        retrieved_docs = []
        try:
            retrieved_docs = retrieve_documents(user_input)
            if not retrieved_docs:
                logger.warning("未检索到相关内容")
        except Exception as e:
            logger.error(f"检索过程出错: {str(e)}")
            # 不中断流程，使用空结果继续

        # 5. 构建对话（增强中文提示词）
        rag_prompt = build_rag_prompt(user_input, retrieved_docs)
        messages.append({"role": "user", "content": rag_prompt})

        # 6. 调用大模型（添加超时处理）
        llm_start = datetime.now()
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                timeout=30  # 设置超时
            )
            llm_time = (datetime.now() - llm_start).total_seconds()
            logger.info(f"大模型调用成功，耗时{llm_time:.2f}s")
        except Exception as e:
            logger.error(f"大模型调用失败: {str(e)}")
            return jsonify({
                "error": "AI服务暂时不可用",
                "retrieved_knowledge": retrieved_docs  # 返回已检索到的知识
            }), 503

        # 7. 构造响应（确保中文正确返回）
        ai_message = {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content
        }

        # 确保消息历史编码正确
        updated_messages = []
        try:
            updated_messages = messages[:-1] + [
                {"role": "user", "content": user_input},
                ai_message
            ]
        except Exception as e:
            logger.error(f"消息历史构造失败: {str(e)}")
            updated_messages = []  # 不影响主流程

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"请求处理完成，总耗时{total_time:.2f}s")

        return jsonify({
            "reply": ai_message["content"],
            "updatedMessages": updated_messages,
            "relatedKnowledge": retrieved_docs,
            "processing_time": total_time,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        return jsonify({
            "error": "系统处理错误",
            "detail": "请稍后再试",
            "status": "error"
        }), 500

if __name__ == "__main__":
    # 启动前再次检查组件
    if None in [client, embedding_model, knowledge_base, faiss_index]:
        logger.critical("⚠️ 有组件未初始化，服务可能无法正常工作！")
    
    # 运行服务
    app.run(host="0.0.0.0", port=10000, debug=False)
