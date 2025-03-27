# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Define la ruta base del proyecto para construir otros paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Carga las variables desde el archivo .env
    model_config = SettingsConfigDict(env_file=BASE_DIR / '.env', extra='ignore')

    # --- API Keys ---
    gemini_api_key: str = "NO_KEY_LOADED" # Valor por defecto si no está en .env

    # --- Modelos ---
    embedding_model_name: str = "hiiamsid/sentence_similarity_spanish_es"
    generative_model_name: str = "gemini-1.5-flash-latest" # O el nombre específico del experimental si es distinto

    # --- Vector Store ---
    # Construye los paths relativos al directorio base
    faiss_index_path: Path = BASE_DIR / "vector_store_index/index.faiss"
    faiss_metadata_path: Path = BASE_DIR / "vector_store_index/index.pkl"

    # --- Otros ---
    # Puedes añadir más configuraciones según necesites

# Crea una instancia única de la configuración para importar en otros módulos
settings = Settings()

# Opcional: Imprimir para verificar al inicio (solo para depuración)
# print("Configuration loaded:")
# print(f"  FAISS Index Path: {settings.faiss_index_path}")
# print(f"  Embedding Model: {settings.embedding_model_name}")
# print(f"  Generative Model: {settings.generative_model_name}")
# print(f"  Gemini Key Loaded: {'Yes' if settings.gemini_api_key != 'NO_KEY_LOADED' else 'No'}")