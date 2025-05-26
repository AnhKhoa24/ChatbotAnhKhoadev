from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_services import is_chitchat, search_context, ask_groq
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
load_dotenv()

app = FastAPI(title="Chatbot RAG cho AnhKhoaDev")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
class HistoryItem(BaseModel):
    role: str  # 'user' hoặc 'bot'
    content: str

class Query(BaseModel):
    question: str
    history: Optional[List[HistoryItem]] = []

@app.post("/chat")
def chat(query: Query):
    question = query.question.strip()

    if is_chitchat(question):
        response = ask_groq(question, history=query.history)
    else:
        context = search_context(question)
        response = ask_groq(question, context, history=query.history)

    return {"answer": response}


# if __name__ == "__main__":
  
#     uvicorn.run("api:app", host="0.0.0.0", port=8222, reload=True)
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
