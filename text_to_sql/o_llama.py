import chromadb
import ollama
import os
import sys

def cargar_metadatos_desde_archivo(nombre_archivo):
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

    # 4. Carga de metadatos desde archivo externo
    lista_de_queries = cargar_metadatos_desde_archivo("text_to_sql/ejemplos_entrenamiento.txt")

    if not isinstance(lista_de_queries, list) and "Error" in lista_de_queries:
        return lista_de_queries

    # 5. Indexación en ChromaDB
    # Se generan IDs automáticos para cada tabla encontrada en el archivo
    try:
        ids_queries = [f"query_{i}" for i in range(len(lista_de_queries))]
        coleccion_queries.add(
            documents=lista_de_queries,
            ids=ids_queries
        )
    except Exception as e:
        return f"Error al indexar las queries en ChromaDB: {e}"

    # 6. Recuperación (Retrieval) del contexto pertinente
    n_results_esquemas = min(10, coleccion_esquemas.count())
    resultados_busqueda = coleccion_esquemas.query(
        query_texts=[solicitud_usuario],
        n_results = n_results_esquemas # Traer las tablas más relevantes
    )

    contexto_tablas = "\n\n".join(resultados_busqueda['documents'][0])

    n_results_queries = min(10, coleccion_queries.count())
    resultados_busqueda = coleccion_queries.query(
        query_texts=[solicitud_usuario],
        n_results=n_results_queries  # Traer las queries más relevantes (máx. disponibles)
    )

    contexto_queries = "\n\n".join(resultados_busqueda['documents'][0])

    # 7. Construcción del Prompt del Sistema
    ruta_instruccion = os.path.join("text_to_sql", "instruccion_sistema.txt")
    try:
        with open(ruta_instruccion, "r", encoding="utf-8") as archivo:
            instruccion_sistema = archivo.read()
    except FileNotFoundError:
        return f"Error: El archivo de instrucción no fue encontrado en {ruta_instruccion}"

    # Prompt mejorado: inyectar los esquemas COMPLETOS directamente en el system prompt
    # para que el modelo los tenga como fuente de verdad absoluta.
    prompt_sistema_completo = f"""{instruccion_sistema}

    A continuación se muestran los ÚNICOS esquemas de tablas que existen.
    NO uses ninguna tabla ni columna que no aparezca aquí:

    {contexto_tablas}
    """

    prompt_usuario = f"""
    A continuación hay ejemplos de consultas SQL correctas que usan los esquemas anteriores.
    Úsalos como referencia de estilo y estructura:

    {contexto_queries}

    Ahora genera la consulta SQL para la siguiente solicitud.
    Recuerda: usa SOLO las tablas y columnas de los esquemas. No inventes nada.

    SOLICITUD DEL USUARIO:
    {solicitud_usuario}
    """

    # 8. Generación de la consulta con Ollama (modelo local)
    try:
        
        respuesta_generada = ollama.generate(
            #model='gemma2', # Asegúrate de haber descargado este modelo con 'ollama run gemma2'
            model='gemma4:e4b',
            prompt=prompt_usuario,
            system=prompt_sistema_completo,
            options={
                "temperature": 0.0,  # Sin aleatoriedad (determinístico)
                "top_k": 1,
                "top_p": 1.0,
                "num_ctx": 8192  # Ventana de contexto más amplia para que entren todos los esquemas
            }
        )

        return "\nOllama\n" + respuesta_generada['response']
        
    except ConnectionError:
        return "Error de conexión. Asegúrate de que Ollama esté ejecutándose en tu sistema."
    except Exception as e:
        return f"Error inesperado al generar la consulta localmente: {e}"


# --- PRUEBAS DE FUNCIONAMIENTO ---
if __name__ == "__main__":
    #print(text_to_sql("Dame una lista de los 10 artículos más vendidos en el año 2023, indícame el nombre del artículo y el monto de ventas."))
    #print(text_to_sql("Dame un resumen de las ventas totales en cada mes del año 2024"))
    print(text_to_sql("Dame una lista de los vendedores que tengan más ventas en el año 2024 junto con el monto de sus ventas"))
    #print(text_to_sql("Dame una lista de los nombres de los vendedores que tengan más ventas en el año 2024 junto con el monto de sus ventas"))
    #print(text_to_sql("Dame el una lista de los productos que son del tipo Ferretería que más se vendieron en el año 2024"))
    #print(text_to_sql("Dame el total de compras de cada mes que ha realizado Sampa en el año 2025"))
    #print(text_to_sql("El total de compras en cada mes del año 2019 que ha realizado Doménica"))