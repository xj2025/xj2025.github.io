# 使用 Python 3.10 官方镜像
FROM python:3.10-slim

# 设置工作目录（容器内部路径）
WORKDIR /app

# 复制 requirements.txt 并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码到容器
COPY . .

# 暴露端口（必须和 fly.toml 的 internal_port 一致）
EXPOSE 8000

# 启动异步 Flask（用 Uvicorn 或 Hypercorn）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
