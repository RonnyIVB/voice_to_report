import uvicorn
import os

if __name__ == "__main__":
    print("Iniciando el servidor de la API Voice to Report...")
    print("La documentación interactiva estará disponible en: http://127.0.0.1:8000/docs")
    
    # Iniciamos uvicorn apuntando al controlador de la API
    uvicorn.run("API.controller:app", host="127.0.0.1", port=8000, reload=True)
