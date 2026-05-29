import os
import sys

# Obtener rutas absolutas
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Cambiar el directorio de trabajo actual al directorio raíz del proyecto.
# Esto asegura que todas las lecturas de archivos relativos (ej. "text_to_sql/...")
# en los scripts importados funcionen correctamente sin importar desde dónde se ejecute el servidor.
os.chdir(parent_dir)

# Agregar la raíz del proyecto a sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from transcripcion.transcriptor import transcribe
from text_to_sql.chroma import text_to_sql as text_to_sql_chroma
from text_to_sql.o_llama import text_to_sql as text_to_sql_ollama
from text_to_sql.pinec import text_to_sql as text_to_sql_pinecone

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="Voice to Report API",
    description="API para transcripción de audio y conversión de texto a SQL mediante RAG usando diferentes backends (ChromaDB, Ollama y Pinecone).",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "mensaje": "Bienvenido a la API de Voice to Report",
        "endpoints": {
            "/transcribe": "Transcripción de audio (GET)",
            "/chromadb": "Texto a SQL usando ChromaDB (GET)",
            "/ollama": "Texto a SQL usando Ollama local (GET)",
            "/pinecone": "Texto a SQL usando Pinecone (GET)"
        }
    }

@app.get("/transcribe")
def get_transcribe(audio_path: str = Query(..., description="Ruta absoluta o relativa al archivo de audio (.ogg)")):
    """
    Recibe la ruta de un archivo de audio y devuelve la transcripción en texto.
    """
    try:
        resultado = transcribe(audio_path)
        return {"resultado": resultado}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la transcripción: {str(e)}")

@app.get("/chromadb")
def get_chromadb(consulta: str = Query(..., description="Consulta en lenguaje natural a convertir en SQL")):
    """
    Convierte lenguaje natural a SQL utilizando ChromaDB como base de datos vectorial para RAG (Gemini).
    """
    try:
        resultado = text_to_sql_chroma(consulta)
        return {"resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento con ChromaDB: {str(e)}")

@app.get("/ollama")
def get_ollama(consulta: str = Query(..., description="Consulta en lenguaje natural a convertir en SQL (Local)")):
    """
    Convierte lenguaje natural a SQL utilizando ChromaDB + Ollama (modelo Gemma local).
    """
    try:
        resultado = text_to_sql_ollama(consulta)
        return {"resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento con Ollama: {str(e)}")

@app.get("/pinecone")
def get_pinecone(consulta: str = Query(..., description="Consulta en lenguaje natural a convertir en SQL (Pinecone)")):
    """
    Convierte lenguaje natural a SQL utilizando Pinecone como base de datos vectorial para RAG (Gemini).
    """
    try:
        resultado = text_to_sql_pinecone(consulta)
        return {"resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento con Pinecone: {str(e)}")
