import streamlit as st
from core.query_and_get_response import ask_query


st.title("Document Q&A")
question = st.text_input("Ask your question about PDFs:")
if question:
     answer = ask_query(question)
     st.write(answer)