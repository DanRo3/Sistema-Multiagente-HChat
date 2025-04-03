from fastapi import APIRouter, HTTPException, Request, status
from app.api.schemas import QueryRequest, QueryResponse
from app.orchestration.graph_state import GraphState
import logging # Usar logging es mejor que prints para producción

# Configurar un logger básico para este módulo
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Procesar consulta en lenguaje natural",
    description="Recibe una consulta en lenguaje natural, la procesa a través del sistema multiagente y devuelve una respuesta textual, una imagen o un error.",
    tags=["Consulta Multiagente"] # Agrupa el endpoint en la documentación Swagger
)
async def process_query(request_data: QueryRequest, request: Request) -> QueryResponse:
    """
    Endpoint principal para procesar las consultas de los usuarios.
    """
    logger.info(f"Recibida nueva consulta: '{request_data.query}'")

    # --- Obtener el grafo compilado ---
    # El grafo se compila al inicio y se almacena en app.state (ver main.py)
    compiled_graph = request.app.state.graph
    if not compiled_graph:
        logger.error("Error crítico: El grafo Langraph no está disponible en app.state.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor: Sistema de procesamiento no inicializado.",
        )

    # --- Preparar la entrada para el grafo ---
    # El estado inicial solo necesita la consulta original
    initial_state = {"original_query": request_data.query}

    # --- Invocar el grafo Langraph ---
    final_state: GraphState = None # Inicializar para evitar errores si invoke falla
    try:
        logger.info("Invocando el grafo Langraph...")
        # .invoke() es síncrono por defecto en muchas implementaciones de Langgraph.
        # Si tu grafo tuviera pasos asíncronos, usarías await compiled_graph.ainvoke(initial_state)
        # Aquí asumimos que nuestros nodos (incluyendo llamadas LLM y ejecución de código)
        # son síncronos o manejados síncronamente por Langchain/Langgraph.
        # Si las llamadas LLM o la ejecución son muy largas, podrían bloquear el
        # event loop de FastAPI. Considera ejecutar invoke en un threadpool si es necesario:
        # from fastapi.concurrency import run_in_threadpool
        # final_state = await run_in_threadpool(compiled_graph.invoke, initial_state)

        final_state = compiled_graph.invoke(initial_state)

        logger.info("Invocación del grafo completada.")
        # logger.debug(f"Estado final del grafo: {final_state}") # Log detallado (cuidado con datos sensibles)

    except Exception as e:
        logger.exception(f"Error inesperado durante la invocación del grafo Langraph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al procesar la consulta: {e}",
        )

    # --- Extraer resultados del estado final ---
    if not final_state:
         logger.error("El estado final del grafo es nulo después de la invocación.")
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail="Error interno: No se pudo obtener el resultado del procesamiento.",
         )

    final_text = final_state.get("final_response_text")
    final_image = final_state.get("final_response_image")
    error_message = final_state.get("error_message")

    logger.info(f"Resultados finales extraídos: Texto={final_text is not None}, Imagen={final_image is not None}, Error={error_message is not None}")

    # --- Construir y devolver la respuesta API ---
    # Priorizar mensaje de error si existe
    if error_message:
        # Podríamos devolver un código de estado diferente si el error no es 500,
        # por ejemplo 400 si la consulta no se pudo procesar por ser inválida.
        # Pero por ahora, lo incluimos en la respuesta 200 OK con el campo error.
        return QueryResponse(error=error_message)
    else:
        # Si no hay error, devolver el texto y/o la imagen
        return QueryResponse(
            text_response=final_text,
            image_response=final_image
            # error es None por defecto
        )