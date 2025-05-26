FROM python:3.10-slim

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép file dependency
COPY requirements.txt .

# Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY . .

# Chạy FastAPI app bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8182"]
