# app/orchestration/graph_builder.py

from langgraph.graph import StateGraph, END
from app.orchestration.graph_state import GraphState
# Importar SOLO los nodos necesarios
from app.orchestration import agent_nodes # Contiene run_pandasai_executor ahora
import logging

logger = logging.getLogger(__name__)

_compiled_graph = None



# --- Función para Construir y Compilar el Grafo---
def build_graph() -> StateGraph:
    """
    Construye el StateGraph lineal basado únicamente en PandasAI.
    """
    logger.info("Construyendo el grafo Langraph (PandasAI-only)...")
    workflow = StateGraph(GraphState)

    # --- 1. Añadir Nodos ---
    logger.info("Añadiendo nodos al grafo...")
    workflow.add_node("moderator", agent_nodes.run_moderator)
    # --- CORRECCIÓN AQUÍ ---
    # Usar el nombre de la función que existe en agent_nodes.py
    workflow.add_node("pandasai_executor", agent_nodes.run_pandasai_executor)
    # Ya no usamos run_pandasai_data_executor
    # ----------------------
    workflow.add_node("contextualizer", agent_nodes.run_contextualizer)
    workflow.add_node("validator", agent_nodes.run_validator)

    # --- 2. Punto de Entrada ---
    logger.info("Estableciendo punto de entrada: 'moderator'")
    workflow.set_entry_point("moderator")

    # --- 3. Añadir Bordes Secuenciales ---
    logger.info("Añadiendo bordes secuenciales...")
    workflow.add_edge("moderator", "pandasai_executor") # El moderador va al único ejecutor
    workflow.add_edge("pandasai_executor", "contextualizer") # El ejecutor va al contextualizador
    workflow.add_edge("contextualizer", "validator")
    workflow.add_edge("validator", END)

    # --- 4. Compilar ---
    logger.info("Compilando el grafo (PandasAI-only)...")
    compiled_workflow = workflow.compile()
    logger.info("¡Grafo Langraph (PandasAI-only) compilado exitosamente!")
    return compiled_workflow


def get_compiled_graph() -> StateGraph:
    """Obtiene la instancia compilada del grafo Langraph (PandasAI-only)."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Grafo (PandasAI-only) no compilado. Compilando...")
        _compiled_graph = build_graph()
    return _compiled_graph