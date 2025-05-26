FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY rag_services.py .
COPY chunks.json .
COPY metadata.json .
COPY index.bin .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8182"]
