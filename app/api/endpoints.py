# app/api/endpoints.py
from fastapi import APIRouter, HTTPException, Depends
from .schemas import QueryRequest, QueryResponse
# Importaremos el grafo Langraph más adelante
# from app.orchestration.graph_builder import get_compiled_graph

router = APIRouter()

# Placeholder para el grafo Langraph (lo obtendremos de app.state más tarde)
# async def get_graph():
#     # Esta función será reemplazada por la obtención del grafo desde el estado de FastAPI
#     raise NotImplementedError("Graph loading not implemented yet")

@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest): # Eliminamos graph=Depends(get_graph) por ahora
    """
    Recibe una consulta en lenguaje natural y devuelve una respuesta.
    (Actualmente es un placeholder)
    """
    print(f"Received query: {request.query}")

    # --- Aquí es donde invocaremos el grafo Langraph ---
    # try:
    #     # graph_input = {"original_query": request.query}
    #     # graph_output = await graph.ainvoke(graph_input) # Usar ainvoke para async
    #
    #     # Procesar graph_output para crear QueryResponse
    #     # text = graph_output.get("final_response_text")
    #     # image = graph_output.get("final_response_image")
    #     # error = graph_output.get("error_message")
    #
    #     # Dummy response por ahora:
    #     text = f"Procesando tu consulta: '{request.query}'..."
    #     image = None
    #     error = None
    #
    #     if error:
    #         # Podrías lanzar HTTPException o devolverlo en la respuesta
    #         return QueryResponse(error=error)
    #     else:
    #         return QueryResponse(text_response=text, image_response=image)
    #
    # except Exception as e:
    #     print(f"Error processing query: {e}")
    #     # Log the full exception traceback here
    #     raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    # Respuesta DUMMY temporal mientras construimos el grafo
    return QueryResponse(
        text_response=f"Recibido (placeholder): '{request.query}'. La lógica completa aún no está conectada.",
        debug_info={"status": "Placeholder endpoint reached"}
    )