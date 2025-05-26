import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

index = faiss.read_index("index.bin")
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def is_chitchat(query: str) -> bool:
    technical_keywords = ["tách chuỗi", "regex", "mảng", "chuỗi", "hàm", "python", "cách xử lý", "code", "câu lệnh"]
    if any(kw in query.lower() for kw in technical_keywords):
        return False

    chitchat_keywords = [
        "bạn là ai", "chào", "hello", "hi", "cảm ơn", "bạn làm gì",
        "khỏe không", "tạm biệt", "ai tạo ra bạn"
    ]
    return any(kw in query.lower() for kw in chitchat_keywords)


def search_context(query, k=5):
    q_vec = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_vec, dtype="float32"), k)
    return [metadata[i] | {"content": chunks[i]} for i in I[0]]

def ask_groq(query, context_chunks=None, model="llama-3.3-70b-versatile"):
    if context_chunks is None:
        context_chunks = []

    context_text = "\n\n".join([
        f"Đoạn {i+1}:\n{chunk['content']}" for i, chunk in enumerate(context_chunks)
    ])

    if context_text.strip():
        prompt = f"""Dựa trên các đoạn sau, hãy trả lời câu hỏi bên dưới một cách chính xác và rõ ràng.\n\n{context_text}\n\nCâu hỏi: {query}\n\nTrả lời:"""
    else:
        prompt = query

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là một trợ lý AI thân thiện và chính xác, là tư vấn viên của AnhKhoaDev. "
                    "Bạn có thể trả lời câu hỏi liên quan đến tài liệu kỹ thuật hoặc trò chuyện lịch sự."
                )
            },
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
