FROM python:3.10-slim

# Cho phép truyền biến môi trường khi build
ARG GROQ_API_KEY
ENV GROQ_API_KEY=$GROQ_API_KEY

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chỉ copy các file cần thiết
COPY main.py .
COPY rag_services.py .
COPY chunks.json .
COPY metadata.json .
COPY index.bin .

# Chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8182"]
