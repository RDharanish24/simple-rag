import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
model=SentenceTransformer("all-MiniLM-L6-v2")
with open(r"C:\Users\DELL\Documents\ml projects\simple-RAG\genai_notes.txt","r",encoding="utf-8") as a:
    raw_text=a.read()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50,separators=["\n\n","\n"," ",""]  )
notes=text_splitter.split_text(raw_text)
embeddings=model.encode(notes)

client=chromadb.PersistentClient(path="./local_db")

collection=client.get_or_create_collection(
    name="document_store",
    metadata={"hnsw:space":"cosine"}
)

collection.add(documents=notes,embeddings=embeddings,ids=[f"id{i}" for i in range(len(notes))])

query=input("enter your query")
query_embedding=model.encode(query)
query_results=collection.query(query_embeddings=[query_embedding],n_results=3)

retrieved_docs= query_results["documents"][0]
context = " ".join(retrieved_docs)
print(context)