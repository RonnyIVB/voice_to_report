import chromadb
from google import genai
from google.genai import errors as genai_errors
import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# 1. Inicialización de clientes y configuración
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY no encontrada en las variables de entorno.")
    cliente_genai = genai.Client(api_key=api_key)
except Exception as e:
    print(f"❌ Error al inicializar el cliente de Gemini: {e}")
    sys.exit(1)

try:
    cliente_db = chromadb.Client()
    coleccion_esquemas = cliente_db.create_collection(name="esquemas_sistema_contable")
    coleccion_queries = cliente_db.create_collection(name="queries_entrenamiento")
except Exception as e:
    print(f"❌ Error al inicializar ChromaDB: {e}")
    sys.exit(1)

# 2. Lectura de metadatos desde archivo externo
def cargar_metadatos_desde_archivo(nombre_archivo):
    if not os.path.exists(nombre_archivo):
        print(f"Error: El archivo {nombre_archivo} no fue encontrado.")
        return []
    
    with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        # Se lee el archivo y se separa por el delimitador definido
        contenido = archivo.read()
        bloques_tablas = [bloque.strip() for bloque in contenido.split('---') if bloque.strip()]
    return bloques_tablas

# Ejecución de la carga
lista_de_esquemas = cargar_metadatos_desde_archivo("query_entrenamiento.txt")

if not lista_de_esquemas:
    print("❌ No se encontraron esquemas para indexar. Verifica el archivo de entrenamiento.")
    sys.exit(1)

# 3. Indexación en ChromaDB
# Se generan IDs automáticos para cada tabla encontrada en el archivo
try:
    ids_tablas = [f"tabla_{i}" for i in range(len(lista_de_esquemas))]
    coleccion_esquemas.add(
        documents=lista_de_esquemas,
        ids=ids_tablas
    )
except Exception as e:
    print(f"❌ Error al indexar los esquemas de las tablas en ChromaDB: {e}")
    sys.exit(1)

# Ejecución de la carga
lista_de_queries = cargar_metadatos_desde_archivo("ejemplos_entrenamiento.txt")

if not lista_de_queries:
    print("❌ No se encontraron queries para indexar. Verifica el archivo de entrenamiento.")
    sys.exit(1)

# 3. Indexación en ChromaDB
# Se generan IDs automáticos para cada tabla encontrada en el archivo
try:
    ids_queries = [f"query_{i}" for i in range(len(lista_de_queries))]
    coleccion_queries.add(
        documents=lista_de_queries,
        ids=ids_queries
    )
except Exception as e:
    print(f"❌ Error al indexar las queries en ChromaDB: {e}")
    sys.exit(1)

# 4. Solicitud de entrada
#solicitud_usuario = "Dame una lista de los 10 artículos más vendidos en el año 2023, indícame el nombre del artículo y el monto de ventas."
#solicitud_usuario = "Dame un resumen de las ventas totales en cada mes del año 2024"
#solicitud_usuario = "Dame una lista de los vendedores que tengan más ventas en el año 2024 junto con el monto de sus ventas"
#solicitud_usuario = "Dame una lista de los nombres de los vendedores que tengan más ventas en el año 2024 junto con el monto de sus ventas"
#solicitud_usuario = "Dame el una lista de los productos que son del tipo Ferretería que más se vendieron en el año 2024"
solicitud_usuario = "Muéstrame el monto mensual de compras que ha realizado Doménica en cada mes del año 2025"
#solicitud_usuario = "El total de compras en cada mes del año 2025 que ha realizado Doménica"

# 5. Recuperación (Retrieval) del contexto pertinente
resultados_busqueda = coleccion_esquemas.query(
    query_texts=[solicitud_usuario],
    n_results = 5 # Traer las 2 tablas más relevantes
)

contexto_tablas = "\n".join(resultados_busqueda['documents'][0])

resultados_busqueda = coleccion_queries.query(
    query_texts=[solicitud_usuario],
    n_results = 10 # Traer las 10 queries más relevantes
)

contexto_queries = "\n".join(resultados_busqueda['documents'][0])

# 6. Construcción del Prompt del Sistema
instruccion_sistema = """
Eres un traductor de lenguaje natural a SQL para PostgreSQL. Tu ÚNICO trabajo es convertir solicitudes en consultas SQL usando EXCLUSIVAMENTE los esquemas proporcionados.

REGLAS ABSOLUTAS E INQUEBRANTABLES:
1. Usa SOLAMENTE los nombres de tablas y columnas EXACTOS que aparecen en los esquemas. NO inventes, NO asumas, NO modifiques ningún nombre.
2. Si una columna se llama "nombre_completo", NUNCA uses "nombre", "name", ni ninguna variación. Copia el nombre EXACTO.
3. Si la información solicitada NO puede resolverse con los esquemas dados, responde ÚNICAMENTE: "-- ERROR: No es posible generar esta consulta con los esquemas disponibles."
4. NO agregues columnas que no existan en los esquemas, aunque parezcan lógicas o útiles.
5. Antes de escribir la consulta, verifica mentalmente que CADA tabla y CADA columna que uses exista textualmente en los esquemas.
6. Devuelve SOLO el código SQL puro. Sin explicaciones, sin comentarios, sin markdown.
7. NO uses funciones o sintaxis que no sea estándar de PostgreSQL.
8. Para nombres propios de personas, tipos de artículos, etc, transforma a minúsculas y usa el operador LIKE para buscar coincidencias
"""

prompt_usuario = f"""
ESQUEMAS DE TABLAS DISPONIBLES (esta es tu ÚNICA fuente de verdad):
{contexto_tablas}

EJEMPLOS DE QUERIES:
{contexto_queries}

SOLICITUD DEL USUARIO:
{solicitud_usuario}
"""

# instruccion_sistema = f"""
# Actúa como un experto en PostgreSQL.

# INSTRUCCIÓN CRÍTICA: Debes usar EXACTAMENTE los nombres de columnas tal como aparecen en los esquemas.
# Por ejemplo, si un esquema define "nombre_completo", NO lo reemplaces por "nombre". 
# Cada columna debe coincidir letra por letra con la definición del esquema.

# Genera una consulta SQL basada en estos esquemas:
# {contexto_tablas}

# Solicitud: {solicitud_usuario}
# Solo devuelve el código SQL.
# """

# 7. Configuración determinística y generación de la consulta
from google.genai import types

config_determinista = types.GenerateContentConfig(
    temperature=0,       # Sin aleatoriedad: siempre elige el token más probable
    top_p=1,             # Desactiva muestreo por nucleus
    top_k=1,             # Solo considera el token con mayor probabilidad
    system_instruction=instruccion_sistema,  # Instrucción del sistema separada
)

# 8. Generación de la consulta
try:
    respuesta_generada = cliente_genai.models.generate_content(
        #model='gemini-2.5-pro',
        #model='gemini-2.5-flash',
        #model='gemma-4-31b-it',
        #model='gemini-3.1-flash-lite-preview',
        model='gemini-3-flash-preview',

        contents=prompt_usuario,
        #contents=instruccion_sistema,
        config=config_determinista
    )

    # Mostrar todos los modelos ejecutando en el terminal:
    # python -c "import os; from google import genai; from dotenv import load_dotenv; load_dotenv(); client = genai.Client(api_key=os.getenv('GEMINI_API_KEY')); [print(m.name) for m in client.models.list()]"

    print(respuesta_generada.text)

except genai_errors.ClientError as e:
    if '429' in str(e):
        print("❌ Error: Has excedido la cuota de la API de Gemini.")
        print("   Opciones: espera a que se restablezca la cuota, usa 'gemini-2.5-flash', o activa la facturación.")
    elif '403' in str(e):
        print("❌ Error: API key inválida o sin permisos. Verifica tu clave de API.")
    elif '400' in str(e):
        print(f"❌ Error en la solicitud a Gemini: {e}")
    else:
        print(f"❌ Error del cliente Gemini: {e}")
except genai_errors.ServerError as e:
    print(f"❌ Error del servidor de Gemini (intenta de nuevo más tarde): {e}")
except ConnectionError:
    print("❌ Error de conexión. Verifica tu conexión a internet.")
except Exception as e:
    print(f"❌ Error inesperado al generar la consulta: {e}")