from asyncio import timeouts
import chromadb
from google import genai
from google.genai import errors as genai_errors
import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

def cargar_metadatos_desde_archivo(nombre_archivo):
    # Lectura de metadatos desde archivo externo
    if not os.path.exists(nombre_archivo):
        return f"Error: El archivo {nombre_archivo} no fue encontrado."
    
    with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
        # Se lee el archivo y se separa por el delimitador definido
        contenido = archivo.read()
        bloques_tablas = [bloque.strip() for bloque in contenido.split('---') if bloque.strip()]
    return bloques_tablas

def text_to_sql(solicitud_usuario):
    
    # 1. Inicialización de clientes y configuración
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no encontrada en las variables de entorno.")
        cliente_genai = genai.Client(api_key=api_key)
    except Exception as e:
        return f"Error al inicializar el cliente de Gemini: {e}"

    try:
        cliente_db = chromadb.Client()
        coleccion_esquemas = cliente_db.create_collection(name="esquemas_sistema_contable")
        coleccion_queries = cliente_db.create_collection(name="queries_entrenamiento")
    except Exception as e:
        return f"Error al inicializar ChromaDB: {e}"

    # 2. Carga de metadatos desde archivo externo
    lista_de_esquemas = cargar_metadatos_desde_archivo("text_to_sql/query_entrenamiento.txt")

    # Si no es una lista y tiene un string con la palabra 'Error:', retornará el mensaje de error
    if not isinstance(lista_de_esquemas, list) and "Error" in lista_de_esquemas:
        return lista_de_esquemas 

    # 3. Indexación en ChromaDB
    # Se generan IDs automáticos para cada tabla encontrada en el archivo
    try:
        ids_tablas = [f"tabla_{i}" for i in range(len(lista_de_esquemas))]
        coleccion_esquemas.add(
            documents=lista_de_esquemas,
            ids=ids_tablas
        )
    except Exception as e:
        return f"Error al indexar los esquemas de las tablas en ChromaDB: {e}"

    # Ejecución de la carga
    lista_de_queries = cargar_metadatos_desde_archivo("text_to_sql/ejemplos_entrenamiento.txt")

    # Si no es una lista y tiene un string con la palabra 'Error:', retornará el mensaje de error
    if not isinstance(lista_de_queries, list) and "Error" in lista_de_queries:
        return lista_de_queries

    # 3. Indexación en ChromaDB
    # Se generan IDs automáticos para cada tabla encontrada en el archivo
    try:
        ids_queries = [f"query_{i}" for i in range(len(lista_de_queries))]
        coleccion_queries.add(
            documents=lista_de_queries,
            ids=ids_queries
        )
    except Exception as e:
        return f"Error al indexar las queries en ChromaDB: {e}"

    # 4. Recuperación (Retrieval) del contexto pertinente
    n_results_esquemas = min(10, coleccion_esquemas.count())
    resultados_busqueda = coleccion_esquemas.query(
        query_texts=[solicitud_usuario],
        n_results=n_results_esquemas  # Traer las tablas más relevantes (máx. disponibles)
    )

    contexto_tablas = "\n".join(resultados_busqueda['documents'][0])

    n_results_queries = min(10, coleccion_queries.count())
    resultados_queries = coleccion_queries.query(
        query_texts=[solicitud_usuario],
        n_results=n_results_queries  # Traer las queries más relevantes (máx. disponibles)
    )

    contexto_queries = "\n".join(resultados_queries['documents'][0])

    # 5. Construcción del Prompt del Sistema
    ruta_instruccion = os.path.join("text_to_sql", "instruccion_sistema.txt")
    try:
        with open(ruta_instruccion, "r", encoding="utf-8") as archivo:
            instruccion_sistema = archivo.read()
    except FileNotFoundError:
        return f"Error: El archivo de instrucción no fue encontrado en {ruta_instruccion}"

    prompt_usuario = f"""
    ESQUEMAS DE TABLAS DISPONIBLES (esta es tu ÚNICA fuente de verdad):
    {contexto_tablas}

    EJEMPLOS DE QUERIES:
    {contexto_queries}

    SOLICITUD DEL USUARIO:
    {solicitud_usuario}
    """

    # 6. Configuración determinística y generación de la consulta
    from google.genai import types

    config_determinista = types.GenerateContentConfig(
        temperature=0,       # Sin aleatoriedad: siempre elige el token más probable
        top_p=1,             # Desactiva muestreo por nucleus
        top_k=1,             # Solo considera el token con mayor probabilidad
        system_instruction=instruccion_sistema,  # Instrucción del sistema separada
    )

    # 7. Generación de la consulta
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
            #print(f"Intentando generar consulta con el modelo: {model_name}...")
            respuesta_generada = cliente_genai.models.generate_content(
                model=model_name,
                contents=prompt_usuario,
                config=config_determinista
            )
            modelo_exitoso = model_name
            #print(f"¡Éxito con el modelo: {model_name}!\n")
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
        except ConnectionError:
            modelos_fallidos += f"Error de conexión del modelo {model_name}: {e}\n"
        except Exception as e:
            modelos_fallidos += f"Error inesperado del modelo {model_name}: {e}\n"

    # Mostrar el resultado final de la consulta generada
    if respuesta_generada:
        return "CromaDB\n" + respuesta_generada.text
    else:
        return "Error:\n" + modelos_fallidos


# --- PRUEBAS DE FUNCIONAMIENTO ---
if __name__ == "__main__":
    #print(text_to_sql("Dame una lista de los 10 artículos más vendidos en el año 2023, indícame el nombre del artículo y el monto de ventas."))
    #print(text_to_sql("Dame un resumen de las ventas totales en cada mes del año 2024"))
    print(text_to_sql("Dame una lista de los vendedores que tengan más ventas en el año 2024 junto con el monto de sus ventas"))
    #print(text_to_sql("Dame una lista de los nombres de los vendedores que tengan más ventas en el año 2024 junto con el monto de sus ventas"))
    #print(text_to_sql("Dame el una lista de los productos que son del tipo Ferretería que más se vendieron en el año 2024"))
    #print(text_to_sql("Dame el total de compras de cada mes que ha realizado Sampa en el año 2025"))
    #print(text_to_sql("El total de compras en cada mes del año 2019 que ha realizado Doménica"))