from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.endpoints import router as api_router
from app.core.config import settings # Importar settings para usarlas si es necesario
# Importaremos funciones de carga más adelante
# from app.vector_store.faiss_store import load_vector_store
# from app.orchestration.graph_builder import get_compiled_graph

# --- Lifespan Management (para cargar modelos/grafos al inicio) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta ANTES de que la aplicación empiece a aceptar requests
    print("Application startup...")
    # 1. Cargar configuración (ya se hace al importar settings)
    print(f"Running with Embedding Model: {settings.embedding_model_name}")
    print(f"Running with Generative Model: {settings.generative_model_name}")

    # 2. Cargar Vector Store (opcional, faiss_store.py puede cargarlo on-demand)
    # try:
    #     load_vector_store()
    # except Exception as e:
    #     print(f"ERROR: Failed to load Vector Store during startup: {e}")
    #     # Podrías decidir detener la app aquí si el vector store es crítico

    # 3. Construir y compilar el grafo Langraph
    # graph = get_compiled_graph()
    # app.state.graph = graph # Almacenar el grafo en el estado de la app
    # print("Langraph compiled and stored in app state.")
    print("Placeholder: Graph compilation would happen here.")

    yield # La aplicación se ejecuta aquí

    # Código que se ejecuta DESPUÉS de que la aplicación se detiene
    print("Application shutdown...")
    # Limpiar recursos si es necesario

# --- Crear la aplicación FastAPI ---
app = FastAPI(
    title="Tesis Multiagente BI",
    description="API para consultar información y obtener respuestas textuales o visuales.",
    version="0.1.0",
    lifespan=lifespan # Usar el lifespan manager
)

# --- Incluir Routers ---
app.include_router(api_router, prefix="/api/v1") # Prefijo opcional para versionado

# --- Endpoint Raíz (Opcional) ---
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Bienvenido a la API del Sistema Multiagente BI"}

# --- Ejecución para pruebas locales (Opcional) ---
# Este bloque permite ejecutar con `python app/main.py`
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)