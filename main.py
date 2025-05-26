from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_services import is_chitchat, search_context, ask_groq

load_dotenv()

app = FastAPI(title="Chatbot RAG cho AnhKhoaDev")

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    question = query.question.strip()

    if is_chitchat(question):
        response = ask_groq(question)
    else:
        context = search_context(question)
        response = ask_groq(question, context)

    return {"answer": response}

# if __name__ == "__main__":
  
#     uvicorn.run("api:app", host="0.0.0.0", port=8222, reload=True)
