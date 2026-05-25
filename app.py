from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import shutil
import traceback

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ai-chat-frontend-one.vercel.app",
        "https://ai-chat-frontend-h92larnka-jrolandomxs-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = None
uploaded_documents = []

conversation_history = [
    {
        "role": "system",
        "content": (
            "Eres un asistente académico y técnico. "
            "Respondes de manera clara, breve y paso a paso."
        ),
    }
]


class ChatRequest(BaseModel):
    prompt: str


class TextRequest(BaseModel):
    text: str


class PDFQuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI Chat API funcionando correctamente"
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


@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):
    conversation_history.append({
        "role": "user",
        "content": request.prompt
    })

    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=conversation_history,
        stream=True
    )

    async def generate():
        full_response = ""

        for chunk in stream:
            content = chunk.choices[0].delta.content

            if content:
                full_response += content
                yield content

        conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


@app.post("/summarize")
def summarize(request: TextRequest):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Resume el texto en máximo 5 líneas."
            },
            {
                "role": "user",
                "content": request.text
            },
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
            {
                "role": "system",
                "content": "Traduce el texto al inglés."
            },
            {
                "role": "user",
                "content": request.text
            },
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
            {
                "role": "system",
                "content": "Extrae de 5 a 10 palabras clave del texto."
            },
            {
                "role": "user",
                "content": request.text
            },
        ]
    )

    return {
        "keywords": response.choices[0].message.content
    }


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, uploaded_documents

    try:
        file_path = f"uploaded_{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"PDF guardado: {file_path}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        uploaded_documents = documents

        print(f"Páginas cargadas: {len(documents)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(documents)

        print(f"Chunks creados: {len(chunks)}")

        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        print("Embeddings inicializados")

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        print("Vectorstore FAISS creado correctamente")

        return {
            "message": "PDF cargado correctamente",
            "filename": file.filename,
            "pages": len(documents),
            "chunks": len(chunks)
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


@app.post("/ask-pdf")
def ask_pdf(request: PDFQuestionRequest):
    global vectorstore

    try:
        if vectorstore is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Primero debes subir un PDF"
                }
            )

        results = vectorstore.similarity_search(
            request.question,
            k=3
        )

        context = "\n\n".join([
            doc.page_content for doc in results
        ])

        prompt = f"""
Responde usando únicamente la información del contexto.

Si la respuesta no está en el contexto responde:
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
                    "content": doc.page_content[:300]
                }
                for doc in results
            ]
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


@app.post("/review-article")
def review_article():
    global uploaded_documents

    try:
        if not uploaded_documents:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Primero debes subir un artículo en PDF"
                }
            )

        full_text = "\n\n".join([
            doc.page_content for doc in uploaded_documents
        ])

        max_chars = 45000

        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]

        review_prompt = f"""
Eres un revisor académico experto de artículos científicos con experiencia en publicación,
arbitraje y evaluación editorial en revistas indexadas nacionales e internacionales.

Tu función es actuar como un dictaminador riguroso, crítico, objetivo y constructivo,
proporcionando observaciones detalladas que permitan fortalecer la calidad científica,
metodológica, teórica y editorial del manuscrito evaluado.

Mantén un tono profesional, académico y humano. Evita frases genéricas, superficiales
o excesivamente duras. Señala fortalezas, áreas de mejora y recomendaciones concretas.

Analiza el siguiente artículo científico:

ARTÍCULO:
{full_text}

Elabora una revisión académica profesional con esta estructura:

# Observaciones generales

Evalúa la estructura general del artículo:
- Título
- Resumen
- Palabras clave
- Introducción
- Planteamiento del problema
- Objetivos
- Justificación
- Marco teórico
- Metodología
- Resultados
- Discusión
- Conclusiones
- Referencias
- Formato APA
- Coherencia general

# Observaciones metodológicas

Analiza:
- Claridad metodológica
- Congruencia entre objetivos, metodología y resultados
- Tipo de estudio
- Técnicas de recolección y análisis
- Validez de resultados
- Posibles sesgos
- Limitaciones

# Observaciones teóricas

Analiza:
- Sustento teórico y conceptual
- Pertinencia y actualidad de las fuentes
- Originalidad y aporte científico
- Profundidad del análisis
- Relación entre teoría y hallazgos

# Observaciones de resultados y discusión

Analiza:
- Claridad de resultados
- Correspondencia entre resultados, tablas, figuras y texto
- Interpretación de hallazgos
- Profundidad de discusión
- Relación con objetivos

# Observaciones de redacción

Identifica:
- Problemas de redacción académica
- Repeticiones
- Ambigüedades
- Párrafos extensos o poco claros
- Falta de cohesión
- Transiciones débiles
- Posible uso artificial o excesivamente genérico del lenguaje

# Observaciones de formato y referencias

Analiza:
- Uso de citas
- Referencias
- Cumplimiento APA 7
- Correspondencia entre citas y referencias
- Tablas y figuras

# Fortalezas del artículo

Menciona fortalezas concretas y específicas.

# Debilidades principales

Menciona debilidades relevantes que deberían corregirse.

# Recomendaciones concretas

Incluye recomendaciones accionables y específicas para mejorar el manuscrito.

# Dictamen final

Incluye:
- Nivel de aporte científico
- Nivel de rigor metodológico
- Nivel de claridad editorial
- Recomendación editorial, eligiendo solo una:
  - Aceptado sin cambios
  - Aceptado con cambios menores
  - Requiere cambios mayores
  - Rechazado

Redacta en español académico, con tono profesional y constructivo.
"""

        response = llm.invoke(review_prompt)

        return {
            "review": response.content
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )