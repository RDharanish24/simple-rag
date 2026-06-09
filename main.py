import ollama
import chromadb
from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")
client=chromadb.PersistentClient(path="./local_db")

collection=client.get_or_create_collection(
    name="document_store",
    metadata={"hnsw:space":"cosine"}
)

query=input("enter your query")
query_embedding=model.encode(query).tolist()
query_results=collection.query(query_embeddings=[query_embedding],n_results=3)

retrieved_docs= query_results["documents"][0]
context = " ".join(retrieved_docs)

prompt = f"""
Use the below context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

response=ollama.generate(model="llama3.1",prompt=prompt)
print(response['response'])

