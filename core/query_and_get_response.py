import os

from sentence_transformers import SentenceTransformer
import faiss
import json
# from getpass import getpass
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()


EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, good for local use
API_KEY = os.getenv("API_KEY")
CLIENT = Mistral(api_key=API_KEY)

def _create_embeddings_for_query(query):
    embeddings = EMBEDDING_MODEL.encode([query], show_progress_bar=True)
    return embeddings


def query_api(prompt, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": prompt
        }
    ]
    chat_response = CLIENT.chat.complete(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)


def prepare_prompt(question, retrieved_context):
    return f"""
     Context information is below.
     ---------------------
     {retrieved_context}
     ---------------------
     Given the context information and not prior knowledge, answer the query.
     Query: {question}
     Answer:
     """


def ask_query(query:str ):
    with open('extracted_pdfs/extracts.json', 'r') as fp:
        chunks = json.load(fp)

    index = faiss.read_index("vector_store/faiss_index.bin")
    question_embeddings = _create_embeddings_for_query(query)
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunks = [chunks[i] for i in I.tolist()[0]]

    context = ". ".join([item['text'] for item in retrieved_chunks])

    prompt = prepare_prompt(query, context)
    response = query_api(prompt)
    return response

if __name__ == '__main__':
    question = "when did Green attend the 2008 Budget Plan meeting in which Davis presented sales projections for upcoming year"
    ask_query(question)