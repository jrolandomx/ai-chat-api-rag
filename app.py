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


def run_review_agent(agent_name: str, instructions: str, article_text: str):
    prompt = f"""
Actúa como {agent_name}.

{instructions}

Analiza el siguiente artículo científico:

{article_text}

Redacta observaciones académicas profundas, específicas, constructivas y útiles.
Evita comentarios genéricos. Señala problemas, justifica por qué deben corregirse
y propone mejoras concretas.
"""

    response = llm.invoke(prompt)

    return response.content


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

        max_chars = 35000

        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]

        methodology_review = run_review_agent(
            "un revisor metodológico experto",
            """
Evalúa exclusivamente el rigor metodológico del artículo.

Considera:
- claridad del diseño metodológico;
- tipo de estudio;
- congruencia entre objetivos, método y resultados;
- población, muestra o corpus;
- técnicas de recolección de datos;
- técnicas de análisis;
- validez, confiabilidad o credibilidad;
- limitaciones;
- posibles sesgos;
- suficiencia de la explicación metodológica.

Entrega tu revisión con el encabezado:
# Revisión metodológica
""",
            full_text
        )

        theoretical_review = run_review_agent(
            "un revisor teórico experto",
            """
Evalúa exclusivamente la dimensión teórica y conceptual del artículo.

Considera:
- pertinencia del marco teórico;
- actualidad de la literatura;
- profundidad conceptual;
- claridad de categorías o variables;
- relación entre teoría, problema y hallazgos;
- originalidad del aporte;
- diálogo con literatura reciente;
- solidez argumentativa.

Entrega tu revisión con el encabezado:
# Revisión teórica
""",
            full_text
        )

        editorial_review = run_review_agent(
            "un revisor editorial académico",
            """
Evalúa exclusivamente la estructura, redacción y coherencia editorial del manuscrito.

Considera:
- título;
- resumen;
- palabras clave;
- introducción;
- planteamiento del problema;
- objetivos;
- justificación;
- resultados;
- discusión;
- conclusiones;
- coherencia general;
- cohesión entre apartados;
- claridad argumentativa;
- redacción académica;
- repeticiones;
- ambigüedades;
- transiciones débiles.

Entrega tu revisión con el encabezado:
# Revisión editorial y de redacción
""",
            full_text
        )

        apa_review = run_review_agent(
            "un revisor de formato APA 7 y estilo editorial",
            """
Evalúa exclusivamente el uso de citas, referencias, tablas, figuras y formato académico.

Considera:
- correspondencia entre citas y referencias;
- suficiencia de referencias;
- actualidad de fuentes;
- posibles ausencias de citas;
- tablas y figuras;
- mención de tablas y figuras en el texto;
- consistencia de formato APA 7;
- errores editoriales visibles.

No inventes referencias. Si no puedes verificar algo con el texto disponible, indícalo.

Entrega tu revisión con el encabezado:
# Revisión de formato, citas y referencias
""",
            full_text
        )

        chief_editor_prompt = f"""
Actúa como editor en jefe de una revista científica indexada.

Tienes cuatro dictámenes especializados:

DICTAMEN METODOLÓGICO:
{methodology_review}

DICTAMEN TEÓRICO:
{theoretical_review}

DICTAMEN EDITORIAL:
{editorial_review}

DICTAMEN APA:
{apa_review}

Con base en ellos, elabora un dictamen final integrado en español académico.

Usa esta estructura:

# Observaciones generales

# Observaciones metodológicas

# Observaciones teóricas

# Observaciones editoriales y de redacción

# Observaciones de formato APA y referencias

# Tabla sintética de observaciones

Incluye una tabla en texto Markdown con columnas:
Apartado | Problema detectado | Recomendación concreta | Prioridad

# Fortalezas del artículo

# Debilidades principales

# Recomendaciones concretas para mejora

# Evaluación por criterios

Incluye una tabla en texto Markdown con columnas:
Criterio | Valoración cualitativa | Puntuación /10

Criterios:
- Originalidad
- Pertinencia del tema
- Rigor metodológico
- Sustento teórico
- Claridad de resultados
- Discusión
- Redacción académica
- Formato APA

# Dictamen final

Incluye:
- Nivel de aporte científico
- Nivel de rigor metodológico
- Nivel de claridad editorial
- Recomendación editorial final, eligiendo solo una:
  - Aceptado sin cambios
  - Aceptado con cambios menores
  - Requiere cambios mayores
  - Rechazado

Mantén un tono riguroso, constructivo, académico y humano.
"""

        final_response = llm.invoke(chief_editor_prompt)

        review = f"""
# Dictamen académico multiagente

## Agente metodológico

{methodology_review}

---

## Agente teórico

{theoretical_review}

---

## Agente editorial

{editorial_review}

---

## Agente APA

{apa_review}

---

## Dictamen consolidado del editor en jefe

{final_response.content}
"""

        return {
            "review": review
        }

    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )