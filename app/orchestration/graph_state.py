# app/orchestration/graph_state.py
from typing import TypedDict, List, Optional, Dict, Any
# Ya no necesitamos Document

class GraphState(TypedDict):
    """
    Define la estructura de datos compartida para el flujo basado en PandasAI.
    """
    # --- Entrada Inicial ---
    original_query: str

    # --- Salida del Moderador ---
    intent: Optional[str]             # 'text', 'visual', 'code'
    pandasai_query: Optional[str]     # La consulta directa para PandasAI

    # --- Salida del Ejecutor PandasAI ---
    # Resultado principal si NO es un gráfico guardado en archivo
    pandasai_result: Optional[Any] = None
    # Tipo del resultado principal ('string', 'dataframe_list', 'number', etc.)
    pandasai_result_type: Optional[str] = None
    # Ruta al archivo PNG si PandasAI guardó un gráfico en disco
    pandasai_plot_path: Optional[str] = None
    # Error específico de la ejecución de PandasAI
    pandasai_error: Optional[str] = None

    # --- Salida del Contextualizador ---
    # Texto acompañante generado por el contextualizador
    summary: Optional[str] = None

    # --- Salida del Validador (Respuesta final) ---
    final_response_text: Optional[str] = None
    final_response_image: Optional[str] = None # Contendrá el string Base64 Data URI
    error_message: Optional[str] = None