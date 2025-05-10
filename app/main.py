from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import uvicorn
import os
from app.core.config import settings
from app.core.llm import get_llm
from app.core.embeddings import initialize_embeddings_model
from app.vector_store.faiss_store import load_faiss_index
from app.orchestration.graph_builder import get_compiled_graph
from app.api.endpoints import router as api_router
from app.core.dataframe_loader import load_and_preprocess_dataframe

# Configurar logging básico para la aplicación
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Lifespan Manager para Inicialización y Limpieza ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Iniciando Aplicación FastAPI ---")

    # 1. Inicializar el LLM (la función get_llm maneja la config interna)
    logger.info(f"Intentando inicializar LLM para proveedor: {settings.LLM_PROVIDER}...")
    llm_instance = get_llm() # Llama a la función genérica
    if not llm_instance:
         logger.error("Fallo crítico al inicializar el LLM. Revisa configuración y API Keys.")
         # Decide si fallar aquí o continuar con funcionalidad limitada
         raise RuntimeError(f"No se pudo inicializar el LLM del proveedor '{settings.LLM_PROVIDER}'.")
    else:
         logger.info("LLM inicializado correctamente.")


    # 2. Inicializar Modelo de Embeddings
    logger.info("Inicializando modelo de embeddings...")
    if not initialize_embeddings_model():
        logger.warning("Fallo al inicializar el modelo de embeddings.")


    # 3. Cargar Índice FAISS
    logger.info("Cargando índice FAISS...")
    faiss_db = load_faiss_index()
    if faiss_db is None:
        logger.warning("No se pudo cargar el índice FAISS.")

    # 4. Cargar el DataFrame de Pandas
    logger.info("Cargando y preprocesando DataFrame de Pandas...")
    dataframe = load_and_preprocess_dataframe()
    if dataframe is None:
         logger.error("Fallo crítico al cargar el DataFrame principal.")
         # Decide si la app puede funcionar sin él
         raise RuntimeError("No se pudo cargar el DataFrame de datos.")
    else:
         # Opcional: Almacenar en app.state si otros componentes lo necesitan directamente
         # app.state.dataframe = dataframe
         logger.info("DataFrame cargado en memoria.")

    # 5. Compilar el Grafo Langraph y Almacenarlo
    logger.info("Compilando grafo Langraph...")
    try:
        compiled_graph = get_compiled_graph()
        if compiled_graph is None:
             raise RuntimeError("get_compiled_graph() devolvió None.")
        app.state.graph = compiled_graph
        logger.info("Grafo Langraph compilado y almacenado en app.state.graph.")
    except Exception as e:
        logger.exception("Error crítico durante la compilación del grafo Langraph.")
        raise RuntimeError(f"Fallo crítico al compilar el grafo: {e}") from e

    logger.info("--- Aplicación lista para recibir peticiones ---")
    yield
    # Código de cierre
    logger.info("--- Cerrando aplicación FastAPI ---")
    app.state.graph = None
    # Podrías añadir limpieza para el cliente LLM si fuera necesario
    # global _llm_client (en llm.py)
    # _llm_client = None

# --- Crear la Instancia de la Aplicación FastAPI ---
app = FastAPI(
    title="Tesis - Sistema Multiagente BI",
    description="API para interactuar con un sistema multiagente basado en IA.",
    version="0.1.0",
    lifespan=lifespan
)

# --- Montar los Routers de la API ---
app.include_router(api_router, prefix="/api")

# --- Ruta Raíz Simple ---
@app.get("/", tags=["General"], summary="Endpoint Raíz")
async def read_root():
    return {"message": "Bienvenido al API del Sistema Multiagente de Tesis BI"}

# --- Bloque para ejecutar con Uvicorn directamente (sin cambios) ---
if __name__ == "__main__":
    # ... (código para ejecutar uvicorn como estaba) ...
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    uvicorn.run("app.main:app", host=host, port=port, reload=reload, log_level="info")