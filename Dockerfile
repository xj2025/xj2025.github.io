


FROM python:3.10-slim  
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8080
# 异步 Flask（推荐）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
