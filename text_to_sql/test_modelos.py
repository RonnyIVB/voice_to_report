import os
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

# Cargar .env de la ruta del proyecto
load_dotenv(dotenv_path="d:/VSCode Proyectos/voice_to_report/.env")

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

modelos_archivo = "text_to_sql/modelos_chromadb.txt"
models_to_test = []

if os.path.exists(modelos_archivo):
    with open(modelos_archivo, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                # Tomamos la primera columna antes de la primera coma
                partes = linea.split(",")
                nombre_modelo = partes[0].strip().strip("'")
                # Evitamos duplicados en la prueba
                if nombre_modelo and nombre_modelo not in models_to_test:
                    models_to_test.append(nombre_modelo)
else:
    print(f"Advertencia: No se encontró el archivo {modelos_archivo}. Usando lista por defecto.")
    models_to_test = [
        'gemini-3.5-flash',
        'gemini-3.1-flash-lite-preview',
        'gemini-3-flash-preview',
        'gemini-2.5-flash'
    ]

print("Probando modelos con la clave de API actual:")
print("-" * 50)

for model_name in models_to_test:
    try:
        # Hacemos una llamada simple
        response = client.models.generate_content(
            model=model_name,
            contents="Hola, responde con solo una palabra para confirmar que funcionas."
        )
        print(f"✅ {model_name}: FUNCIONA. Respuesta: {response.text.strip()}")
    except genai_errors.APIError as e:
        print(f"❌ {model_name}: Error de API ({type(e).__name__}) -> {e}")
    except Exception as e:
        print(f"❌ {model_name}: Otro error ({type(e).__name__}) -> {e}")
