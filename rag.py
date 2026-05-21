from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma

load_dotenv()

# 1. Cargar PDF
loader = PyPDFLoader("documento.pdf")

documents = loader.load()

print(f"Paginas cargadas: {len(documents)}")

# 2. Dividir texto en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks creados: {len(chunks)}")

# 3. Crear embeddings
embeddings = OpenAIEmbeddings()

# 4. Crear vector DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Base vectorial creada")

# 5. Buscar información
query = "¿De qué trata el documento?"

results = vectorstore.similarity_search(query, k=3)

print("\nResultados encontrados:\n")

for result in results:
    print(result.page_content)
    print("\n-----------------\n")

# 6. Modelo LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini"
)

# 7. Crear contexto
context = "\n".join([doc.page_content for doc in results])

prompt = f"""
Responde usando únicamente la información proporcionada.

Contexto:
{context}

Pregunta:
{query}
"""

response = llm.invoke(prompt)

print("\nRESPUESTA FINAL:\n")

print(response.content)