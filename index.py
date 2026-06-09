import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
model=SentenceTransformer("all-MiniLM-L6-v2")
with open(r"C:\Users\DELL\Documents\ml projects\simple-RAG\genai_notes.txt","r",encoding="utf-8") as a:
    raw_text=a.read()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50,separators=["\n\n","\n"," ",""]  )
notes=text_splitter.split_text(raw_text)
embeddings=model.encode(notes)

client=chromadb.PersistentClient(path="./local_db")

collection=client.get_or_create_collection(
    name="document_store",
    metadata={"hnsw:space":"cosine"}
)
if collection.count()==0:
    collection.add(documents=notes,embeddings=embeddings.tolist(),ids=[f"id{i}" for i in range(len(notes))])



