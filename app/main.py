from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import uvicorn # Para ejecutar si se llama directamente

# Importar configuración y funciones de inicialización/carga
from app.core.config import settings
from app.core.llm import configure_genai, get_gemini_llm # Importar config y getter
from app.core.embeddings import initialize_embeddings_model
from app.vector_store.faiss_store import load_faiss_index
from app.orchestration.graph_builder import get_compiled_graph # Importar el getter del grafo

# Importar el router de la API
from app.api.endpoints import router as api_router

# Configurar logging básico para la aplicación
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Lifespan Manager para Inicialización y Limpieza ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta ANTES de que la aplicación empiece a aceptar peticiones
    logger.info("--- Iniciando Aplicación FastAPI ---")

    # 1. Configurar API de Google GenAI (necesario antes de inicializar LLM)
    logger.info("Configurando Google GenAI...")
    if not configure_genai():
        logger.warning("Fallo al configurar Google GenAI. La funcionalidad LLM no estará disponible.")
        # Podrías decidir lanzar un error aquí si el LLM es absolutamente crítico
        # raise RuntimeError("Fallo crítico: No se pudo configurar la API de Gemini.")

    # 2. Inicializar Modelo de Embeddings
    logger.info("Inicializando modelo de embeddings...")
    if not initialize_embeddings_model():
        logger.warning("Fallo al inicializar el modelo de embeddings.")
        # Considera si esto es un error crítico

    # 3. Cargar Índice FAISS
    logger.info("Cargando índice FAISS...")
    faiss_db = load_faiss_index()
    if faiss_db is None:
        logger.warning("No se pudo cargar el índice FAISS. Funcionalidad de búsqueda vectorial limitada o nula.")
        # Decide si quieres que la app falle si el índice no carga
        # raise RuntimeError("Fallo crítico: No se pudo cargar la base de datos vectorial FAISS.")

    # 4. Compilar el Grafo Langraph y Almacenarlo en el Estado de la App
    logger.info("Compilando grafo Langraph...")
    try:
        compiled_graph = get_compiled_graph() # Llama al builder/getter
        if compiled_graph is None:
             raise RuntimeError("get_compiled_graph() devolvió None.")
        # Almacenar el grafo compilado en el estado de la aplicación FastAPI
        # para que esté disponible en las peticiones a través de request.app.state.graph
        app.state.graph = compiled_graph
        logger.info("Grafo Langraph compilado y almacenado en app.state.graph.")
    except Exception as e:
        logger.exception("Error crítico durante la compilación del grafo Langraph.")
        # Si el grafo no compila, la aplicación no puede funcionar.
        raise RuntimeError(f"Fallo crítico al compilar el grafo: {e}") from e

    logger.info("--- Aplicación lista para recibir peticiones ---")
    yield
    # Código que se ejecuta DESPUÉS de que la aplicación termine (limpieza)
    logger.info("--- Cerrando aplicación FastAPI ---")
    # Aquí podrías añadir código de limpieza si fuera necesario (cerrar conexiones, etc.)
    app.state.graph = None # Liberar referencia al grafo

# --- Crear la Instancia de la Aplicación FastAPI ---
app = FastAPI(
    title="Tesis - Sistema Multiagente BI",
    description="API para interactuar con un sistema multiagente basado en IA para consultas en lenguaje natural sobre datos marítimos.",
    version="0.1.0",
    lifespan=lifespan # Registrar el gestor de ciclo de vida
)

# --- Montar los Routers de la API ---
# Incluir el router que definimos en endpoints.py bajo el prefijo /api
app.include_router(api_router, prefix="/api")

# --- Ruta Raíz Simple ---
@app.get("/", tags=["General"], summary="Endpoint Raíz")
async def read_root():
    """Devuelve un mensaje de bienvenida."""
    return {"message": "Bienvenido al API del Sistema Multiagente de Tesis BI"}

# --- Bloque para ejecutar con Uvicorn directamente (opcional) ---
if __name__ == "__main__":
    logger.info("Ejecutando Uvicorn directamente desde main.py")
    # Leer host y port desde variables de entorno o usar defaults
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true" # Habilitar reload si RELOAD=true

    uvicorn.run("app.main:app", host=host, port=port, reload=reload, log_level="info")