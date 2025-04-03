# app/orchestration/graph_state.py
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    Define la estructura de datos compartida que fluye a través del grafo Langraph.
    Cada campo representa una pieza de información que un agente puede necesitar
    o producir.
    """
    # Entrada inicial
    original_query: str

    # Salida del Moderador
    intent: Optional[str]           # 'text', 'visual', 'code'
    filters: Optional[Dict[str, Any]] # Filtros para metadatos (o None)
    search_query: Optional[str]     # Consulta optimizada para búsqueda vectorial

    # Salida del Recuperador
    retrieved_docs: Optional[List[Document]] # Lista de documentos recuperados

    # Salida del Contextualizador
    summary: Optional[str]          # Resumen textual o explicación
    needs_visualization: bool = False # Flag para activar la rama de código/visualización
    data_for_python: Optional[Any]  # Datos preparados para el agente Python (str o dict/list)

    # Salida del Agente Python
    python_code: Optional[str]      # Código Python generado

    # Salida del Ejecutor de Código
    execution_output: Optional[str] # Resultado de la ejecución (stdout o base64 de imagen)
    execution_error: Optional[str]  # Mensaje de error si la ejecución falló

    # Salida del Validador (Respuesta final para el usuario)
    final_response_text: Optional[str]  # Texto final validado
    final_response_image: Optional[str] # Base64 de imagen final validada
    error_message: Optional[str]        # Mensaje de error final para el usuario