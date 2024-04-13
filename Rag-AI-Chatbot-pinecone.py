from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import time
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
# from pinecone import Pinecone, ServerlessSpec

import os

from langchain.schema import Document
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

embeddings = OpenAIEmbeddings(openai_api_key="open-ai-key",model="text-embedding-ada-002")
pinecone.init(
api_key='877efaad-c217-4026-a0eb-728cab79a098', environment='gcp-starter'
)
chat = ChatOpenAI(
    openai_api_key='openai-key',
    model='gpt-3.5-turbo'
)
messages = [
    SystemMessage(content="You are a helpful assistant.")
]


def fetch_docs_text():
    loader = PyPDFLoader("python_tutorial.pdf")
    pages = loader.load_and_split()
    return pages
def text_spliters():
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   docs = text_splitter.split_documents(fetch_docs_text())
   return docs

def create_database():

    if 'python' not in pinecone.list_indexes():
        pinecone.create_index("python", dimension=1536,metric='dotproduct')
        vectorstores = Pinecone.from_documents(text_spliters(), embeddings, index_name='python')
        return vectorstores
    else:
        vectorstores = Pinecone.from_documents(text_spliters(), embeddings, index_name='python')
        return vectorstores
db=create_database()


def augment_prompt(query: str):
    results = db.similarity_search(query)
    source_knowledge = results[0].page_content
    augmented_prompt = f"""Using the contexts below, answer the query.
    Contexts:
    {source_knowledge}
    Query: {query}"""
    return augmented_prompt



print("Exit for prees 'e' \n")
while True:
    user=input("You: ")
    if user.lower() == 'e':
        break
    messages.append(HumanMessage(
        content=augment_prompt(user)
    ))
    res = chat(messages)
    messages.append(AIMessage(
        content=str(res.content)
    ))
    print(f"AI: {str(res.content)}")