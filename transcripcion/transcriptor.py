import whisper
import os

def transcribe(audio_path: str) -> str:
    """
    Toma un path de un archivo de audio (.ogg) en idioma español y
    devuelve un String con la transcripción completa del audio.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"El archivo no existe: {audio_path}")

    # Cargamos el modelo 'base', el cual es un buen balance entre velocidad y precisión.
    # Se descargará automáticamente la primera vez (aprox. 140MB).
    model = whisper.load_model("base")

    # Realizamos la transcripción. 
    # Especificamos 'es' (español) para mejorar la precisión y evitar detecciones erróneas.
    result = model.transcribe(audio_path, language="es")

    # Retornamos el texto transcrito sin espacios innecesarios al inicio o final.
    return result["text"].strip()

if __name__ == "__main__":
    # Ejemplo de uso (descomenta para probar si tienes un archivo .ogg):
    # try:
    #     texto = transcribe("mi_audio.ogg")
    #     print(f"Transcripción:\n{texto}")
    # except Exception as e:
    #     print(f"Error: {e}")
    pass
