import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
model=SentenceTransformer("all-MiniLM-L6-v2")
with open(r"C:\Users\DELL\Documents\ml projects\simple-RAG\genai_notes.txt","r",encoding="utf-8") as a:
    raw_text=a.read()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50,separators=["\n\n","\n"," ",""]  )
notes=text_splitter.split_text(raw_text)
embeddings=model.encode(notes)

print(f"Number of documents processed: {len(notes)}")
print(f"Shape of embeddings array: {embeddings.shape}")
