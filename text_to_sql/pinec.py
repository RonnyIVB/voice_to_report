import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pinecone import Pinecone, ServerlessSpec

# 1. Configuración de Credenciales
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    print("❌ Error: GEMINI_API_KEY o PINECONE_API_KEY no encontradas en el archivo .env")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# 2. Configuración del Índice en Pinecone
nombre_indice = "esquemas-olap-contable"

# Crear el índice si no existe (Configuración para nivel gratuito)
if nombre_indice not in pc.list_indexes().names():
    pc.create_index(
        name=nombre_indice,
        dimension=768, # Dimensión específica para el modelo 'gemini-embedding-001'
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

indice = pc.Index(nombre_indice)

# 3. Función para cargar y procesar el archivo de entrenamiento
def cargar_contexto_entrenamiento(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        contenido = f.read()
        # Separamos por el delimitador de tres guiones
        bloques = [b.strip() for b in contenido.split('---') if b.strip()]
    return bloques

# 4. Generación de Embeddings e Indexación
def indexar_esquemas(bloques):
    print(f"Indexando {len(bloques)} estructuras en Pinecone...")
    for i, bloque in enumerate(bloques):
        print(f"  Procesando esquema {i+1}/{len(bloques)}...")
        resultado_embedding = client.models.embed_content(
            model="gemini-embedding-001",
            contents=bloque,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        vector = resultado_embedding.embeddings[0].values
        
        # Insertar en Pinecone
        indice.upsert(vectors=[{
            "id": f"esquema_{i}",
            "values": vector,
            "metadata": {"texto_sql": bloque}
        }])

# 5. Proceso de Transformación: Texto a SQL
def transformar_texto_a_sql(consulta_usuario):
    # A. Convertir la consulta del usuario en vector
    query_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=consulta_usuario,
        config=types.EmbedContentConfig(output_dimensionality=768),
    ).embeddings[0].values
    
    # B. Buscar los 3 esquemas más relevantes en Pinecone
    resultados = indice.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    contexto_recuperado = "\n\n".join([match['metadata']['texto_sql'] for match in resultados['matches']])

    # Cargar instrucción del sistema
    ruta_instruccion = os.path.join("text_to_sql", "__pycache__", "instruccion_sistema.txt")
    try:
        with open(ruta_instruccion, "r", encoding="utf-8") as archivo:
            instruccion_sistema = archivo.read()
    except FileNotFoundError:
        print(f"❌ Error: El archivo de instrucción no fue encontrado en {ruta_instruccion}")
        sys.exit(1)

    # C. Generar la consulta SQL con Gemini (nuevo SDK)
    prompt = f"""
    Actúa como un experto en PostgreSQL. Genera una consulta SQL utilizando este esquema OLAP:
    {contexto_recuperado}
    Solicitud del usuario: {consulta_usuario}
    Instrucciones: {instruccion_sistema}
    """
    
    respuesta = client.models.generate_content(
        # model="gemini-2.0-flash",
        model="gemini-2.5-flash",
        contents=prompt
    )
    return respuesta.text

# --- EJECUCIÓN DEL FLUJO ---
try:
    # Paso A: Cargar e indexar (Solo se necesita hacer una vez o cuando cambie el archivo)
    esquemas = cargar_contexto_entrenamiento("query_entrenamiento.txt")
    indexar_esquemas(esquemas)

    # Paso B: Ejemplo de uso
    consulta = "Dame una lista de los 10 productos más vendidos en el año 2025, indícame el nombre y el monto."
    sql_generado = transformar_texto_a_sql(consulta)

    print("\n--- CONSULTA SQL GENERADA ---")
    print(sql_generado)

except ClientError as e:
    if e.code == 429:
        print("\n[ERROR] Se agotaron las peticiones disponibles de la API de Gemini.")
        print("   Has alcanzado el límite de uso (rate limit) de tu plan actual.")
        print("   Soluciones:")
        print("   1. Espera unos minutos e intenta de nuevo.")
        print("   2. Revisa tu cuota en: https://aistudio.google.com/apikey")
        print("   3. Considera actualizar tu plan en Google AI Studio.")
        sys.exit(1)
    else:
        raise