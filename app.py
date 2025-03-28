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
    """初始化必要组件"""
    global client, knowledge_base, faiss_index
    
    try:
        # 1. 获取百度Access Token
        auth_url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={config.api_key}&client_secret={config.secret_key}"
        token_data = requests.get(auth_url).json()
        config.access_token = token_data["access_token"]
        config.token_expire = datetime.now() + timedelta(seconds=token_data["expires_in"] - 60)
        logger.info("百度ERNIE认证成功")

        # 2. OpenAI客户端
        client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # 3. 加载知识库
        with open('1.json', 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
            logger.info(f"加载知识库，共{len(knowledge_base)}条数据")
        
        # 4. 加载FAISS索引
        vectors = np.load('knowledge_vectors.npy')
        
        # 关键修复步骤
        faiss.normalize_L2(vectors)  # 必须归一化
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])  # 必须用内积
        faiss_index.add(vectors)
        # 验证索引类型
        assert faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT, "索引类型错误！"
        logger.info("✅ 所有组件初始化完成")
        avg_norm = np.mean(np.linalg.norm(vectors, axis=1))
        logger.info(f"向量平均范数: {avg_norm:.4f} (正常范围: 0.8-1.2)")



        # 加载向量后添加归一化检查
        vectors = np.load('knowledge_vectors.npy')
        norms = np.linalg.norm(vectors, axis=1)
        logger.info(f"原始向量范数范围: min={norms.min():.4f}, max={norms.max():.4f}")

        # 归一化处理
        vectors = vectors / norms[:, np.newaxis]
        np.save('knowledge_vectors_normalized.npy', vectors)  # 保存归一化后的向量

        # 使用归一化后的向量构建索引
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])  # 改用内积（余弦相似度）
        faiss_index.add(vectors.astype('float32'))
        logger.info(f"归一化后向量范数: {np.linalg.norm(vectors[0]):.4f}")

        # ===== 新增健康检查 =====
        def check_vector_health():
            """向量系统健康诊断"""
            # 检查样本向量
            sample_vectors = np.load('knowledge_vectors.npy')[:50]  # 检查前50个向量
            faiss.normalize_L2(sample_vectors)
            
            # 1. 自相似度检查（对角线应为1.0）
            self_sim = np.diag(sample_vectors @ sample_vectors.T)
            if not np.allclose(self_sim, 1.0, atol=0.01):
                raise ValueError(f"向量未正确归一化！自相似度范围: {self_sim.min():.2f}-{self_sim.max():.2f}")
            
            # 2. 向量间相似度分布
            cross_sim = sample_vectors @ sample_vectors.T
            np.fill_diagonal(cross_sim, np.nan)  # 忽略对角线
            avg_sim = np.nanmean(cross_sim)
            logger.info(f"向量健康状态 - 平均相似度: {avg_sim:.3f} (正常范围: 0.05~0.3)")
            
            # 3. 索引类型验证
            if not hasattr(faiss_index, 'metric_type'):
                logger.warning("无法确认FAISS索引类型！")
            elif faiss_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                logger.error(f"索引类型错误！当前: {faiss_index.metric_type}, 应为内积(METRIC_INNER_PRODUCT)")
        
        check_vector_health()  # 执行检查
        # ======================
        
        logger.info("✅ 所有组件初始化完成并通过健康检查")
    

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
# ========== 百度ERNIE向量化 ==========
async def get_embeddings(text):
    """使用百度ERNIE获取文本嵌入"""
    if datetime.now() > config.token_expire:
        logger.info("Token过期，自动刷新")
        initialize_components()
        asyncio.run(analyze_similarity_distribution())  # 新增此行
    url = f"{config.embed_api_url}?access_token={config.access_token}"
    
    payload = json.dumps({
        "input": [text],
        "user_id": "rag_system"
    }, ensure_ascii=False)
    
    headers = {
        'Content-Type': 'application/json'
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url,
                headers=headers,
                data=payload.encode('utf-8'),
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP错误: {response.status}")
                
                data = await response.json()
                
                if "error_code" in data:
                    error_msg = data.get("error_msg", "未知错误")
                    logger.error(f"百度API错误 {data['error_code']}: {error_msg}")
                    raise ValueError(f"API错误: {error_msg}")
                
                if not data.get("data") or not isinstance(data["data"], list):
                    raise ValueError("返回数据格式异常")
                
                embedding = np.array(data['data'][0]['embedding'])
                # 添加归一化（关键修复）
                embedding = embedding / np.linalg.norm(embedding)
                logger.debug(f"归一化后向量范数: {np.linalg.norm(embedding):.4f}")
                return embedding
        
                
        except Exception as e:
            logger.error(f"百度API调用失败: {str(e)}\n{traceback.format_exc()}")
            avg_vector = np.mean(np.load('knowledge_vectors.npy'), axis=0)
            logger.warning(f"使用备用向量，维度: {avg_vector.shape}")
            return np.zeros(384) / np.linalg.norm(np.zeros(384))  # 返回归一化的零向量

# ========== 检索函数 ==========
# async def retrieve_documents(query):

#     try:
#         start_time = datetime.now()
        
#         # 1. 获取查询向量
#         query_embedding = await get_embeddings(query)
#         logger.debug(f"查询向量示例值: {query_embedding[:3]}")  # 打印前3个值
        
#         # 2. FAISS搜索
#         distances, indices = faiss_index.search(query_embedding.reshape(1, -1).astype('float32'), 
#                                              config.retrieve_top_k)
#         logger.debug(f"原始搜索结果 - 距离: {distances}, 索引: {indices}")
        
#         # 3. 结果过滤
#         results = []
#         for idx, dist in zip(indices[0], distances[0]):
#             similarity = 1 - dist
#             if similarity > config.similarity_threshold:
#                 results.append({
#                     "text": knowledge_base[idx]["text"],
#                     "similarity": float(similarity),
#                     "index": int(idx)
#                 })
#             logger.debug(f"文档{idx} - 相似度: {similarity:.4f} (阈值: {config.similarity_threshold})")
        
#         logger.info(f"检索完成: 耗时{(datetime.now()-start_time).total_seconds():.2f}s, 有效结果数: {len(results)}")
#         return [item["text"] for item in results]
#     except Exception as e:
#         logger.error(f"检索失败: {str(e)}\n{traceback.format_exc()}")
#         return []


# ========== 检索函数 ==========
# async def retrieve_documents(query):
#     try:
#         query_embedding = await get_embeddings(query)
#         query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
#         # 使用FAISS内积搜索（余弦相似度）
#         similarities, indices = faiss_index.search(query_embedding, config.retrieve_top_k)
        
#         results = [
#             {"text": knowledge_base[idx]["text"], "score": float(score)}
#             for idx, score in zip(indices[0], similarities[0])
#             if score > config.similarity_threshold
#         ]
#         logger.info(f"检索结果分数范围: {min(r['score'] for r in results):.2f}-{max(r['score'] for r in results):.2f}")
#         return results
#     except Exception as e:
#         logger.error(f"检索失败: {str(e)}")
#         return []
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
        # query_embedding = await get_embeddings(query)
        # query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # # 执行搜索（临时调大返回数量）
        # similarities, indices = faiss_index.search(query_embedding, 20)  # 改为返回20个结果
        
        # if len(similarities[0]) == 0:
        #     raise ValueError("FAISS返回空结果，请检查索引")
            
        # # 打印所有结果的原始分数（无论是否通过阈值）
        # logger.info("===== 相似度分数详情 =====")
        # for idx, score in zip(indices[0], similarities[0]):
        #     logger.info(f"文档{idx}: {score:.4f} {'✅' if score > config.similarity_threshold else '❌'}")
        
        # # 过滤结果
        # results = [
        #     {"text": knowledge_base[idx]["text"], "score": float(score)}
        #     for idx, score in zip(indices[0], similarities[0])
        #     if score > config.similarity_threshold
        # ]
        
        # return results
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

# async def async_chat_handler(data):
#     start_time = datetime.now()
#     user_input = data["userInput"].strip()
#     messages = data.get("messages", [])
    
#     try:
#         retrieved_docs = await retrieve_documents(user_input)
#         # 动态调整参数（示例）
#         if len(retrieved_docs) == 0:
#            logger.warning("未检索到结果，自动降低阈值")
#            config.similarity_threshold = max(0.3, config.similarity_threshold - 0.1)
#            retrieved_docs = await retrieve_documents(user_input)
#         rag_prompt = f"""【参考知识】\n{chr(10).join(f'- {doc[:100]}...' for doc in retrieved_docs)}\n\n【问题】{user_input}"""
#         messages.append({"role": "user", "content": rag_prompt})
        
#         if len(messages) > config.memory_window * 2:
#             messages = messages[-config.memory_window * 2:]

#         response = client.chat.completions.create(
#             model="deepseek-chat",
#             messages=messages,
#             max_tokens=config.max_response_tokens,
#             timeout=config.llm_timeout
#         )
        
#         ai_message = response.choices[0].message
        
#         return jsonify({
#             "reply": ai_message.content,
#             "updatedMessages": messages[:-1] + [
#                 {"role": "user", "content": user_input},
#                 {"role": ai_message.role, "content": ai_message.content}
#             ],
#             "relatedKnowledge": retrieved_docs,
#             "status": "success",
#             "processing_time": (datetime.now() - start_time).total_seconds()
#         })
#     except Exception as e:
#         logger.error(f"处理失败: {str(e)}\n{traceback.format_exc()}")
#         raise
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
        
        # # 构建提示词（添加调试信息）
        # debug_info = f"[调试] 使用阈值: {config.similarity_threshold:.2f} | 结果数: {len(retrieved_docs)}"
        # rag_prompt = f"""【参考知识】{debug_info}\n{chr(10).join(f'- {doc[:100]}...' for doc in retrieved_docs)}\n\n【问题】{user_input}"""
        # 构建提示词（修复字典访问问题）
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
        initialize_components()
        app.run(host="0.0.0.0", port=10000, debug=False)
    except Exception as e:
        logger.critical(f"服务启动失败: {str(e)}\n{traceback.format_exc()}")
