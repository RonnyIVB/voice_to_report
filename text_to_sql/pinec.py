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
    print("Error: GEMINI_API_KEY o PINECONE_API_KEY no encontradas en el archivo .env")
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
    ruta_instruccion = os.path.join("text_to_sql", "instruccion_sistema.txt")
    try:
        with open(ruta_instruccion, "r", encoding="utf-8") as archivo:
            instruccion_sistema = archivo.read()
    except FileNotFoundError:
        print(f"Error: El archivo de instrucción no fue encontrado en {ruta_instruccion}")
        sys.exit(1)

    # C. Generar la consulta SQL con Gemini (nuevo SDK)
    prompt = f"""
    Actúa como un experto en PostgreSQL. Genera una consulta SQL utilizando este esquema OLAP:
    {contexto_recuperado}
    Solicitud del usuario: {consulta_usuario}
    Instrucciones: {instruccion_sistema}
    """
    
    config_determinista = types.GenerateContentConfig(
        temperature=0,       # Sin aleatoriedad: siempre elige el token más probable
        top_p=1,             # Desactiva muestreo por nucleus
        top_k=1,             # Solo considera el token con mayor probabilidad
        system_instruction=instruccion_sistema,  # Instrucción del sistema separada
    )

    from google.genai import errors as genai_errors

    modelos_archivo = "text_to_sql/modelos_chromadb.txt"
    modelos_disponibles = []

    if os.path.exists(modelos_archivo):
        with open(modelos_archivo, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    # Tomamos el nombre del modelo (primera columna, sin comillas simples)
                    partes = linea.split(",")
                    nombre_modelo = partes[0].strip().strip("'")
                    if nombre_modelo and nombre_modelo not in modelos_disponibles:
                        modelos_disponibles.append(nombre_modelo)
    else:
        # Lista por defecto en caso de que falte el archivo
        modelos_disponibles = [
            'gemini-3.5-flash',
            'gemini-3.1-flash-lite-preview',
            'gemini-3-flash-preview'
        ]

    respuesta_generada = None
    modelo_exitoso = None
    modelos_fallidos = ""

    for model_name in modelos_disponibles:
        try:
            respuesta_generada = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config_determinista
            )
            modelo_exitoso = model_name
            break  # Si funciona uno, ya no probar con los otros modelos
        except genai_errors.ClientError as e:
            if '429' in str(e):
                modelos_fallidos += f"Error (429) del modelo {model_name}: Cuota de la API excedida o requiere facturación.\n"
            elif '403' in str(e):
                modelos_fallidos += f"Error (403) del modelo {model_name}: Clave de API inválida o sin permisos.\n"
            else:
                modelos_fallidos += f"Error del cliente (400) del modelo {model_name}: {e}\n"
        except genai_errors.ServerError as e:
            modelos_fallidos += f"Error del servidor (500) del modelo {model_name}: {e}\n"
        except ConnectionError as e:
            modelos_fallidos += f"Error de conexión del modelo {model_name}: {e}\n"
        except Exception as e:
            modelos_fallidos += f"Error inesperado del modelo {model_name}: {e}\n"

    if respuesta_generada:
        return respuesta_generada.text
    else:
        return "Error:\n" + modelos_fallidos

# --- EJECUCIÓN DEL FLUJO ---
try:
    # Paso A: Cargar e indexar (Solo se necesita hacer una vez o cuando cambie el archivo)
    esquemas = cargar_contexto_entrenamiento("text_to_sql/query_entrenamiento.txt")
    if not esquemas:
        print("No se encontraron esquemas para indexar. Verifica el archivo de entrenamiento.")
        sys.exit(1)
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