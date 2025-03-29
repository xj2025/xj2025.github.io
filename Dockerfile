


FROM python:3.10-slim  # 基础镜像（不要用 docker:latest）

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
# 异步 Flask（推荐）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
