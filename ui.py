import streamlit as st
import requests

st.set_page_config(page_title="RAG CHATBOT",layout="centered")

st.title("RAG CHATBOT")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg["content"])

user_input=st.chat_input("Ask Something....")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role":"user","content":user_input})

    try:
        response=requests.post("http://127.0.0.1:8000/ask",
            json={"query": user_input}
        )

        answer=response.json().get("answer","no answer")


    except Exception as e:
        answer=f"error {e}"

    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})