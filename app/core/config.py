import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # ... (otras configuraciones)

    # --- CAMBIA ESTA LÍNEA ---
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    # ------------------------

    FAISS_INDEX_FOLDER: str = "data_index"
    FAISS_INDEX_NAME: str = "data" # Asegúrate que coincida con tus archivos

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()

# (Validación opcional de existencia de archivos/carpetas aquí...)