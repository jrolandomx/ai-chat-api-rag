# AI Academic Reviewer Backend

Backend oficial de la plataforma inteligente para revisión académica, análisis documental y arbitraje científico asistido por inteligencia artificial.

Desarrollado para:

**Instituto de Investigaciones en Contaduría**  
**Universidad Veracruzana**

---

# Overview

AI Academic Reviewer Backend es una API desarrollada con FastAPI orientada al procesamiento inteligente de artículos científicos, documentos PDF y generación automatizada de dictámenes académicos mediante inteligencia artificial.

El sistema integra:

- OpenAI GPT
- LangChain
- FAISS
- Retrieval-Augmented Generation (RAG)
- análisis semántico,
- revisión académica multiagente,
- exportación de dictámenes.

---

# Main Features

## Academic Peer Review Engine

- Revisión metodológica
- Revisión teórica
- Revisión editorial
- Revisión APA 7
- Dictamen editorial automatizado
- Evaluación multiagente
- Score académico
- Badge editorial

---

## PDF Intelligence

- Carga de documentos PDF
- Extracción automática de texto
- Chunking inteligente
- Embeddings semánticos
- Búsqueda contextual
- Question Answering sobre documentos

---

## Blind Review System

La plataforma puede ocultar automáticamente:

- nombres de autores,
- afiliaciones,
- ORCID,
- correos,
- agradecimientos,

para simular arbitraje académico ciego.

---

## Supported Review Types

- Scopus
- WoS
- CONAHCYT
- Latindex
- Tesis doctoral
- Tesis maestría

---

## Export Features

- Exportación a Word (.docx)
- Exportación a PDF
- Reportes académicos automatizados

---

# Tech Stack

## Backend Framework

- FastAPI
- Python

---

## AI Stack

- OpenAI GPT-4.1-mini
- LangChain
- OpenAI Embeddings
- FAISS Vector Database

---

## PDF Processing

- PyPDF
- Recursive Text Splitting
- Semantic Retrieval

---

## Document Generation

- python-docx
- reportlab

---

# Backend Architecture

```text
FastAPI API
      ↓
PDF Processing
      ↓
Chunking
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
OpenAI GPT Models
      ↓
Academic Multi-Agent Review
```

---

# Multi-Agent Academic Review System

La plataforma simula un sistema editorial compuesto por:

1. Revisor metodológico
2. Revisor teórico
3. Revisor editorial
4. Revisor APA
5. Editor en jefe

---

# Generated Academic Sections

El sistema genera automáticamente:

- Badge editorial
- Score académico
- Revisión metodológica
- Revisión teórica
- Revisión editorial
- Revisión APA
- Tabla de observaciones
- Fortalezas
- Debilidades
- Recomendaciones
- Dictamen final

---

# Project Structure

```text
app.py
requirements.txt
.env
uploaded_files/
```

---

# Installation

## Clone repository

```bash
git clone https://github.com/jrolandomx/ai-chat-api.git
```

---

## Enter project folder

```bash
cd ai-chat-api
```

---

# Create Virtual Environment

## macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
```

---

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

# Run Development Server

```bash
uvicorn app:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

---

# API Documentation

FastAPI Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# Main Endpoints

## General Chat

```http
POST /chat
```

General AI assistant.

---

## Streaming Chat

```http
POST /chat-stream
```

Streaming AI responses.

---

## Upload PDF

```http
POST /upload-pdf
```

Upload academic articles and scientific PDFs.

---

## Ask PDF

```http
POST /ask-pdf
```

Ask questions directly about uploaded documents.

---

## Academic Review

```http
POST /review-article
```

Generate academic peer review.

Parameters:

```text
review_type
blind_review
```

---

## Export Word

```http
POST /export-review-word
```

Download review as Word document.

---

## Export PDF

```http
POST /export-review-pdf
```

Download review as PDF document.

---

## Summarize Text

```http
POST /summarize
```

Generate text summary.

---

## Translate Text

```http
POST /translate
```

Translate content.

---

## Extract Keywords

```http
POST /keywords
```

Extract academic keywords.

---

# RAG Workflow

```text
Upload PDF
    ↓
Extract Text
    ↓
Chunk Text
    ↓
Generate Embeddings
    ↓
Store in FAISS
    ↓
Semantic Search
    ↓
Context Retrieval
    ↓
LLM Response
```

---

# Requirements

## Core Dependencies

```txt
fastapi
uvicorn
openai
python-dotenv
python-multipart
langchain
langchain-openai
langchain-community
faiss-cpu
pypdf
python-docx
reportlab
tiktoken
```

---

# Deployment

## Recommended Backend Hosting

- Render

---

# Production Command

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

---

# Future Improvements

- Database integration
- User authentication
- ORCID integration
- Review history
- Collaborative editorial review
- Citation validation
- Statistical analysis agent
- AI writing detection
- Multi-document comparison
- Reviewer analytics dashboard

---

# Academic Purpose

This backend was designed for:

- scientific peer review,
- editorial processes,
- thesis evaluation,
- academic research support,
- institutional scientific analysis.

---

# Institutional Version

Developed for:

**Instituto de Investigaciones en Contaduría**  
**Universidad Veracruzana**

---

# Author

Rolando Ramirez Rueda

GitHub:
https://github.com/jrolandomx