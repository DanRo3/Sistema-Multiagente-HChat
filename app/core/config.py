# app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    GEMINI_API_KEY: str = "NO_API_KEY" # Valor predeterminado seguro
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_FOLDER: str = "vector_store_index"
    FAISS_INDEX_NAME: str = "data_index" # Confirma este nombre
    GEMINI_MODEL_NAME: str = "gemini-2.0-flash" # O el modelo experimental que uses
    # Opcional: Configuración de seguridad para ejecución de código
    CODE_EXECUTION_TIMEOUT: int = 40 # Segundos

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        # Manejar caso donde la variable no está en .env o entorno
        validate_assignment = True # Asegura que los tipos coincidan
        extra = 'ignore'

settings = Settings()

# Validación simple al cargar
if settings.GEMINI_API_KEY == "NO_API_KEY":
    print("ADVERTENCIA: GEMINI_API_KEY no encontrada en el entorno.")

# Puedes añadir más validaciones (e.g., existencia de carpeta FAISS)