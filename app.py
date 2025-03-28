from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import numpy as np
import faiss
import json
import os
import logging
from datetime import datetime, timedelta
import sys
import io
import aiohttp
import asyncio
import platform
import requests
import traceback

# Windows系统需要特殊设置事件循环策略
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 强制设置系统默认编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# ========== 初始化日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== 百度ERNIE配置 ==========
class ChatConfig:
    def __init__(self):
        # 百度ERNIE配置
        self.api_key = "qehpo02v3xq42r0lrNy3VpZR"
        self.secret_key = "BmPgo9ghoxdLpHv3IMWEkshF83VoZDVW"
        self.embed_api_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1"
        self.access_token = None
        self.token_expire = None
        
        # OpenAI配置
        self.openai_api_key = "sk-f104aed04216406abce806380d6670a3"
        self.openai_base_url = "https://api.deepseek.com"
        
        # 性能参数
        self.llm_timeout = 25
        self.retrieve_timeout = 8
        self.max_response_tokens = 100
        
        # 检索参数
        self.retrieve_top_k = 10  # 临时调大检索数量
        self.similarity_threshold = 0.7
        self.debug_mode = False  # 调试开关

         # 对话管理
        self.memory_window = 4  # 历史消息轮次

         # 新增调试参数
        self.min_similarity = 0.3  # 最低记录阈值
        self.max_similarity = 0.9  # 最高记录阈值 = 4

config = ChatConfig()

# ========== 全局组件 ==========
client = None
knowledge_base = None
faiss_index = None

def initialize_components():
    """初始化必要组件（同步版本）"""
    global client, knowledge_base, faiss_index
    
    try:
        # 1. 获取百度Access Token（同步请求）
        auth_url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={config.api_key}&client_secret={config.secret_key}"
        token_data = requests.get(auth_url, timeout=10).json()  # 添加超时
        if 'error' in token_data:
            raise ValueError(f"百度认证失败: {token_data.get('error_description')}")
        
        config.access_token = token_data["access_token"]
        config.token_expire = datetime.now() + timedelta(seconds=token_data["expires_in"] - 60)
        logger.info(f"百度ERNIE认证成功，Token有效期至: {config.token_expire}")

        # 2. OpenAI客户端
        client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # 3. 加载知识库（使用绝对路径）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        knowledge_path = os.path.join(base_dir, '1.json')
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
            logger.info(f"加载知识库，共{len(knowledge_base)}条数据")
        
        # 4. 加载并归一化向量
        vectors_path = os.path.join(base_dir, 'knowledge_vectors.npy')
        vectors = np.load(vectors_path)
        faiss.normalize_L2(vectors)
        
        # 5. 构建FAISS索引
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)
        logger.info(f"FAISS索引构建完成，维度: {vectors.shape[1]}")
        def check_vector_health(vectors):
            """向量系统健康诊断"""
            try:
                # 1. 检查向量范数
                norms = np.linalg.norm(vectors, axis=1)
                logger.info(f"向量范数范围: {norms.min():.4f}-{norms.max():.4f}")
                
                # 2. 检查索引类型
                assert faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT, "必须使用内积索引"
                
                # 3. 抽样检查相似度
                sample = vectors[:5] @ vectors[:5].T
                np.fill_diagonal(sample, np.nan)
                logger.info(f"样本相似度范围: {np.nanmin(sample):.2f}-{np.nanmax(sample):.2f}")
                
            except Exception as e:
                logger.error(f"健康检查失败: {str(e)}")
                raise
                # 健康检查
        check_vector_health(vectors)
        
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}\n{traceback.format_exc()}")
        raise


#相似度分布分析
async def analyze_similarity_distribution():
    """系统启动时自动分析相似度分布"""
    test_queries = [
        "常见问题", 
        "操作指南",
        "错误解决方法",
        "系统要求",
        "如何使用"
    ]
    
    logger.info("开始相似度分布分析...")
    for query in test_queries:
        embedding = await get_embeddings(query)
        distances, _ = faiss_index.search(embedding.reshape(1, -1), 50)  # 检查前50个结果
        similarities = 1 - distances[0]
        
        logger.info(
            f"查询: '{query}'\n"
            f"相似度分布: min={similarities.min():.2f} | "
            f"max={similarities.max():.2f} | "
            f"mean={similarities.mean():.2f}\n"
            f"高于当前阈值({config.similarity_threshold})的结果: "
            f"{sum(similarities > config.similarity_threshold)}个"
        )
async def refresh_access_token():
    """异步刷新百度Token"""
    try:
        auth_url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={config.api_key}&client_secret={config.secret_key}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(auth_url) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP错误: {resp.status}")
                token_data = await resp.json()
                config.access_token = token_data["access_token"]
                config.token_expire = datetime.now() + timedelta(seconds=token_data["expires_in"] - 60)
                logger.info("百度Token刷新成功")
    except Exception as e:
        logger.error(f"Token刷新失败: {str(e)}")
        config.access_token = None  # 强制下次重新初始化

async def get_embeddings(text):
    """使用百度ERNIE获取文本嵌入（异步安全版）"""
    try:
        # Token检查（线程安全）
        if not config.access_token or (config.token_expire and datetime.now() > config.token_expire):
            await refresh_access_token()
            if not config.access_token:
                raise ValueError("无法获取有效Token")
        
        # API调用
        url = f"{config.embed_api_url}?access_token={config.access_token}"
        payload = json.dumps({"input": [text], "user_id": "rag_system"}, ensure_ascii=False)
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.post(url, headers={'Content-Type': 'application/json'}, data=payload) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP错误: {resp.status}")
                
                data = await resp.json()
                if "error_code" in data:
                    raise ValueError(f"API错误 {data['error_code']}: {data.get('error_msg')}")
                
                embedding = np.array(data['data'][0]['embedding'])
                return embedding / np.linalg.norm(embedding)  # 归一化
                
    except Exception as e:
        logger.error(f"获取embedding失败: {str(e)}")
        return np.zeros(384)  # 返回零向量保底


# ========== 检索函数 ==========
async def retrieve_documents(query):
    try:
        query_embedding = await get_embeddings(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)  # 查询向量也需归一化

        similarities, indices = faiss_index.search(query_embedding, 20)  # 扩大检索范围

        # 分数修正和过滤
        valid_results = [
            {"text": knowledge_base[idx]["text"], "score": float((score + 1)/2)}  # 映射到[0,1]
            for idx, score in zip(indices[0], similarities[0])
            if score > -0.5  # 容忍部分负相关
        ]
        # 确保返回结构一致
        return [{
            "text": str(knowledge_base[idx]["text"]),  # 强制转为字符串
            "score": float(score)
        } for idx, score in zip(indices[0], similarities[0]) if score > config.similarity_threshold]
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        return [{
            "text": str(doc["text"]), 
            "score": 0.0
        } for doc in knowledge_base[:config.retrieve_top_k]]  # 保底返回
     
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        return []
# ========== 核心路由 ==========
@app.route("/api/chat", methods=["POST"])
def chat():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = request.get_json()
        if not data or "userInput" not in data:
            return jsonify({"error": "需要提供userInput参数"}), 400

        return loop.run_until_complete(async_chat_handler(data))
    except Exception as e:
        logger.error(f"请求处理失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
    finally:
        loop.close()


async def async_chat_handler(data):
    start_time = datetime.now()
    user_input = data["userInput"].strip()
    messages = data.get("messages", [])
    
    try:
        # 保存原始阈值
        original_threshold = config.similarity_threshold
        
        # 首次检索
        retrieved_docs = await retrieve_documents(user_input)
        
        # 动态调整阈值（紧急修复）
        if len(retrieved_docs) == 0:
            logger.warning(f"首次检索失败，当前阈值: {config.similarity_threshold}")
            
            # 逐步降低阈值直到获得结果或达到最低阈值
            for attempt in range(3):
                new_threshold = max(0.3, config.similarity_threshold - 0.1*(attempt+1))
                logger.warning(f"尝试降低阈值至: {new_threshold}")
                config.similarity_threshold = new_threshold
                retrieved_docs = await retrieve_documents(user_input)
                
                if len(retrieved_docs) > 0:
                    logger.warning(f"在阈值 {new_threshold} 下检索到 {len(retrieved_docs)} 条结果")
                    break
                    
            if len(retrieved_docs) == 0:
                logger.error("即使阈值降至0.3仍无结果，返回默认知识")
                retrieved_docs = [doc["text"] for doc in knowledge_base[:config.retrieve_top_k]]
        
        # 恢复原始阈值（避免影响后续请求）
        config.similarity_threshold = original_threshold
        
        # 记录最终使用的阈值和结果数
        logger.info(f"最终使用阈值: {config.similarity_threshold} | 结果数: {len(retrieved_docs)}")
       
        debug_info = f"[阈值: {config.similarity_threshold:.2f} | 结果数: {len(retrieved_docs)}]"
        knowledge_str = chr(10).join(
            f"- [相似度: {doc['score']:.2f}] {doc['text'][:100]}..." 
            for doc in retrieved_docs
        )
        
        rag_prompt = f"""【参考知识】{debug_info}{knowledge_str}【问题】{user_input}"""
        messages.append({"role": "user", "content": rag_prompt})
        
        # 控制历史长度
        if len(messages) > config.memory_window * 2:
            messages = messages[-config.memory_window * 2:]

        # 调用大模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=config.max_response_tokens,
            timeout=config.llm_timeout
        )
        
        ai_message = response.choices[0].message
        
        return jsonify({
            "reply": ai_message.content,
            "updatedMessages": messages[:-1] + [
                {"role": "user", "content": user_input},
                {"role": ai_message.role, "content": ai_message.content}
            ],
            "relatedKnowledge": retrieved_docs,
            "status": "success",
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "debug": {  # 添加调试信息
                "final_threshold": config.similarity_threshold,
                "retrieved_count": len(retrieved_docs)
            }
        })
    except Exception as e:
        logger.error(f"处理失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "debug": {
                "threshold": config.similarity_threshold,
                "retrieved_docs": []
            }
        })
# ========== 调试路由 ========== （新增这部分）
@app.route("/api/debug_search", methods=["POST"])
async def debug_search():
    """可视化调试检索过程"""
    data = request.get_json()
    query = data.get("query", "测试查询")
    
    # 1. 获取查询向量
    query_embedding = await get_embeddings(query)
    
    # 2. 执行搜索
    distances, indices = faiss_index.search(
        query_embedding.reshape(1, -1).astype('float32'),
        config.retrieve_top_k * 3  # 获取更多结果分析
    )
    
    # 3. 分析结果
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = 1 - dist
        results.append({
            "rank": len(results) + 1,
            "similarity": float(similarity),
            "text": knowledge_base[idx]["text"][:100] + "...",
            "passed": similarity > config.similarity_threshold
        })
    
    return jsonify({
        "query": query,
        "embedding_shape": query_embedding.shape,
        "threshold": config.similarity_threshold,
        "results": sorted(results, key=lambda x: -x["similarity"])
    })    
# ========== 测试路由 ==========
@app.route("/api/test_baidu_embedding", methods=["GET"])
async def test_baidu_embedding():
    """测试百度embedding服务状态"""
    test_text = "这是一个测试文本"
    try:
        token_url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={config.api_key}&client_secret={config.secret_key}"
        token_res = requests.get(token_url)
        if token_res.status_code != 200:
            return jsonify({"error": "Token获取失败", "detail": token_res.text}), 500
        
        embedding = await get_embeddings(test_text)
        
        return jsonify({
            "status": "success",
            "embedding_shape": embedding.shape,
            "sample_values": embedding[:3].tolist()
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/api/system_status", methods=["GET"])
def system_status():
    """系统健康检查"""
    return jsonify({
        "status": "running",
        "components": {
            "baidu_token_valid": config.token_expire > datetime.now() if config.access_token else False,
            "knowledge_base_loaded": bool(knowledge_base),
            "faiss_index_ready": bool(faiss_index),
            "openai_client_ready": bool(client)
        },
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    try:
        
        async def main():
            await initialize_components()
            app.run(host="0.0.0.0", port=10000)
    
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"服务启动失败: {str(e)}\n{traceback.format_exc()}")
