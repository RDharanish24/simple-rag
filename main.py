import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

model=SentenceTransformer("all-MiniLM-L6-v2")
with open(r"C:\Users\DELL\Documents\ml projects\simple-RAG\genai_notes.txt","r",encoding="utf-8") as a:
    notes=a.read().splitlines()

embeddings=model.encode(notes)

print(f"Number of documents processed: {len(notes)}")
print(f"Shape of embeddings array: {embeddings.shape}")
