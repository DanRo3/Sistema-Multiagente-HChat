from typing import Optional
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from dotenv import load_dotenv
from typing import Literal # Para tipos literales
import logging

logger = logging.getLogger(__name__)
load_dotenv()

# Definir los proveedores soportados explícitamente
LLMProvider = Literal["google", "openai", "huggingface_local"]

class Settings(BaseSettings):
    """
    Configuración centralizada de la aplicación. Carga desde .env y variables de entorno.
    Determina qué configuraciones de LLM son relevantes según LLM_PROVIDER.
    """
    # --- Proveedor LLM Principal ---
    # Cambia esta variable en .env para seleccionar el LLM a usar
    LLM_PROVIDER: LLMProvider = Field(default="google", description="Proveedor de LLM a usar ('google', 'openai', 'huggingface_local')")

    # --- Configuración Google Gemini ---
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="API Key para Google Gemini (requerido si LLM_PROVIDER='google')")
    GEMINI_MODEL_NAME: str = Field(default="gemini-1.5-flash-latest", description="Modelo específico de Gemini a usar")

    # --- Configuración OpenAI ---
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="API Key para OpenAI (requerido si LLM_PROVIDER='openai')")
    OPENAI_MODEL_NAME: str = Field(default="gpt-4o", description="Modelo específico de OpenAI a usar (e.g., gpt-4o, gpt-3.5-turbo)")

    # --- Configuración Hugging Face Local ---
    HUGGINGFACE_MODEL_ID: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2", description="ID del modelo en Hugging Face Hub (si LLM_PROVIDER='huggingface_local')")
    HF_MODEL_DEVICE: str = Field(default="auto", description="Dispositivo para HF local ('auto', 'cuda', 'cpu', 'mps')")
    HF_MODEL_LOAD_IN_8BIT: bool = Field(default=False, description="Usar cuantización 8-bit (requiere bitsandbytes)")
    HF_MODEL_LOAD_IN_4BIT: bool = Field(default=False, description="Usar cuantización 4-bit (requiere bitsandbytes)")

    # --- Configuración Embeddings (Independiente del LLM) ---
    EMBEDDING_MODEL_NAME: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Modelo de embeddings a usar")
    EMBEDDING_NORMALIZE: bool = Field(default=True, description="Normalizar embeddings (debe coincidir con creación de índice)")
    HF_CACHE_FOLDER: Optional[str] = Field(default="./huggingface_cache", description="Carpeta de caché para modelos Hugging Face")

    # --- Configuración FAISS (Independiente del LLM) ---
    FAISS_INDEX_FOLDER: str = Field(default="vector_store_index", description="Carpeta que contiene los archivos del índice FAISS")
    FAISS_INDEX_NAME: str = Field(default="data_index", description="Nombre base de los archivos del índice FAISS (sin extensión)")

    # --- Configuración Ejecución de Código ---
    CODE_EXECUTION_TIMEOUT: int = Field(default=15, description="Timeout en segundos para ejecución de código Python")
    CSV_FILE_PATH: str = Field(default="data/DataLimpia.csv", description="Ruta al archivo CSV principal con los datos")
    # --- Configuración del Modelo Pydantic ---
    model_config = SettingsConfigDict(
        env_file='.env',              # Nombre del archivo .env
        env_file_encoding='utf-8',    # Codificación del archivo .env
        extra='ignore',               # Ignorar variables extra en el entorno o .env
        case_sensitive=False          # Nombres de variables insensibles a mayúsculas/minúsculas
    )
    # En app/core/config.py, dentro de la clase Settings:
    PANDASAI_VERBOSE: bool = Field(default=True, description="Habilitar logs detallados de PandasAI")
    PANDASAI_ENABLE_CACHE: bool = Field(default=False, description="Habilitar caché de respuestas en PandasAI")
    PANDASAI_CHART_DIR_NAME: str = Field(default="pandasai_charts", description="Nombre del directorio donde PandasAI guarda los gráficos")
    PANDASAI_MAX_RETRIES: int = Field(default=3, description="Número máximo de reintentos de PandasAI para corregir código") # PandasAI puede reintentar

    # --- Validadores (Opcional pero recomendado) ---
    @validator('GEMINI_API_KEY', 'OPENAI_API_KEY', pre=True, always=True)
    def check_api_keys(cls, v, values):
        """Valida que la API key requerida esté presente según el proveedor."""
        provider = values.get('LLM_PROVIDER')
        if provider == 'google' and not v:
            logger.warning("LLM_PROVIDER='google' pero GEMINI_API_KEY no está configurada en .env o entorno.")
            # Podrías lanzar un ValueError si quieres que la app falle al inicio
            # raise ValueError("GEMINI_API_KEY es requerida cuando LLM_PROVIDER es 'google'")
        if provider == 'openai' and not v:
            logger.warning("LLM_PROVIDER='openai' pero OPENAI_API_KEY no está configurada en .env o entorno.")
            # raise ValueError("OPENAI_API_KEY es requerida cuando LLM_PROVIDER es 'openai'")
        return v

# Crear instancia única de la configuración
settings = Settings()

# Log inicial para confirmar el proveedor seleccionado
logger.info(f"Configuración cargada. Proveedor LLM seleccionado: {settings.LLM_PROVIDER}")
if settings.LLM_PROVIDER == "google":
    logger.info(f"  Usando modelo Gemini: {settings.GEMINI_MODEL_NAME}")
elif settings.LLM_PROVIDER == "openai":
     logger.info(f"  Usando modelo OpenAI: {settings.OPENAI_MODEL_NAME}")
elif settings.LLM_PROVIDER == "huggingface_local":
     logger.info(f"  Usando modelo local Hugging Face: {settings.HUGGINGFACE_MODEL_ID}")