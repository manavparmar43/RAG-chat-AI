from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgembedding import PGEmbedding 
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import time
embedding_funcation=OpenAIEmbeddings(openai_api_key="open-ai-key")
connection_string="postgresql+psycopg2://postgres:root@localhost:5432/langchain_embanding_data"
collection_name = "state_of_the_union"
def fetch_docs_text():
   loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
   docs = loader.load()
   return docs

def text_spliters():
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
   docs = text_splitter.split_documents(fetch_docs_text())
   return docs

def create_database():
   try:

         db = PGEmbedding.from_documents(
               embedding=embedding_funcation,
               documents=text_spliters(),
               collection_name=collection_name,
               connection_string=connection_string,
               pre_delete_collection=False,
         )
         return db
   except Exception as e:
      print(f"Error: {e}")
      db = None
      return db


db=create_database()
while True:
   if db is not None:
      query = input("\nAsk Question: ")
      print("\nProcessing.... \n")
      time.sleep(2)
      data = db.similarity_search(query)
      print(data[0].page_content)
   else:
      create_database() 