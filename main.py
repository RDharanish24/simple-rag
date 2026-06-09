import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
app=FastAPI()
model=SentenceTransformer("all-MiniLM-L6-v2")
client=chromadb.PersistentClient(path="./local_db")

collection=client.get_or_create_collection(
    name="document_store",
    metadata={"hnsw:space":"cosine"}
)

class Queryrequest(BaseModel):
    query:str

@app.get("/")
def home():
    return {"message":"RAG API is running"}

@app.post("/ask")

def ask_question(request:Queryrequest):
    query=request.query
    query_embedding=model.encode(query).tolist()
    query_results=collection.query(query_embeddings=[query_embedding],n_results=3)
    retrieved_docs= query_results.get("documents",[[]])[0]
    if not retrieved_docs:
        return {"answer":"no relevent answers found"}
    context = " ".join(retrieved_docs)




    prompt = f"""
You are an AI assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    response=ollama.generate(model="llama3.1",prompt=prompt)
    return {"answer":response.get("response","no response")}

