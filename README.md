# simple-RAG

A lightweight Retrieval-Augmented Generation (RAG) demo using:
- `sentence-transformers` for embedding generation
- `chromadb` for persistent vector storage
- `FastAPI` for query API
- `Streamlit` for chat UI
- `ollama` for LLM response generation

## Files
- `index.py` - loads text from `genai_notes.txt`, splits into chunks, embeds them, and stores vectors in `local_db`
- `main.py` - FastAPI service that queries the vector store and uses `ollama` to generate answers
- `ui.py` - Streamlit frontend that sends user questions to the API
- `requirements.txt` - Python dependencies

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure `genai_notes.txt` exists in the project root.

## Run
1. Start the API:
   ```bash
   uvicorn main:app --reload
   ```
2. Start the UI:
   ```bash
   streamlit run ui.py
   ```

## Notes
- The `local_db` folder stores `chromadb` persistent data.
- Update paths in `ui.py` and `index.py` if the project is moved.
- `ollama` requires a local Ollama installation and compatible model.
