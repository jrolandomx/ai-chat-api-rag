from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import shutil

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

load_dotenv()

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vectorstore = None
llm = ChatOpenAI(model="gpt-4.1-mini")


class ChatRequest(BaseModel):
    prompt: str


class TextRequest(BaseModel):
    text: str


class PDFQuestionRequest(BaseModel):
    question: str


conversation_history = [
    {
        "role": "system",
        "content": (
            "Eres un asistente académico y técnico. "
            "Respondes con claridad, de forma breve y paso a paso. "
            "No inventes información; si no sabes algo, dilo."
        )
    }
]


@app.get("/")
def home():
    return {"message": "AI Chat API funcionando"}


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    file_path = f"uploaded_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return {
        "message": "PDF cargado y procesado correctamente",
        "filename": file.filename,
        "pages": len(documents),
        "chunks": len(chunks)
    }


@app.post("/ask-pdf")
def ask_pdf(request: PDFQuestionRequest):
    global vectorstore

    if vectorstore is None:
        return {
            "error": "Primero debes subir un PDF usando /upload-pdf"
        }

    results = vectorstore.similarity_search(request.question, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Responde usando únicamente la información del contexto.

Si la respuesta no está en el contexto, responde:
"No encontré esa información en el documento."

Contexto:
{context}

Pregunta:
{request.question}
"""

    response = llm.invoke(prompt)

    return {
        "question": request.question,
        "answer": response.content,
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:500]
            }
            for doc in results
        ]
    }


@app.post("/chat")
def chat(request: ChatRequest):
    conversation_history.append({
        "role": "user",
        "content": request.prompt
    })

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=conversation_history
    )

    assistant_response = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": assistant_response
    })

    return {
        "response": assistant_response,
        "history": conversation_history
    }


@app.post("/summarize")
def summarize(request: TextRequest):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Resume el texto en máximo 5 líneas."},
            {"role": "user", "content": request.text}
        ]
    )

    return {
        "summary": response.choices[0].message.content
    }


@app.post("/translate")
def translate(request: TextRequest):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Traduce el texto al inglés."},
            {"role": "user", "content": request.text}
        ]
    )

    return {
        "translation": response.choices[0].message.content
    }


@app.post("/keywords")
def keywords(request: TextRequest):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Extrae de 5 a 10 palabras clave del texto."},
            {"role": "user", "content": request.text}
        ]
    )

    return {
        "keywords": response.choices[0].message.content
    }