# app/orchestration/agent_nodes.py
from typing import Dict, Any, Optional
from app.orchestration.graph_state import GraphState
import logging

# Importar las funciones lógicas de cada agente
from app.agents import moderator_agent
from app.agents import contextualizer_agent
from app.agents import pandasai_agent # Agente PandasAI
from app.agents import validation_agent

logger_nodes = logging.getLogger(__name__)

# --- Funciones Nodo para Langraph---
# --- NODO EJECUTOR MODERADOR ---
def run_moderator(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta el agente moderador (versión PandasAI-only)."""
    logger_nodes.info("--- Ejecutando Nodo: Moderador ---")
    query = state['original_query']
    analysis_result = moderator_agent.analyze_query(query)
    logger_nodes.info(f"Resultado Moderador (PandasAI-only): {analysis_result}")
    return {
        "intent": analysis_result.get("intent"),
        "pandasai_query": analysis_result.get("pandasai_query")
    }

# --- NODO EJECUTOR PANDASAI ---
def run_pandasai_executor(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta la consulta usando PandasAI y devuelve el diccionario de resultados."""
    logger_nodes.info("--- Ejecutando Nodo: Ejecutor PandasAI ---")
    query_to_run = state.get('pandasai_query')

    if not query_to_run:
         logger_nodes.error("No se encontró consulta PandasAI para ejecutar.")
         # Devolver diccionario de error consistente con la salida de run_pandasai
         return {"pandasai_result": None, "pandasai_result_type": None, "pandasai_plot_path": None, "pandasai_error": "Consulta PandasAI vacía."}

    # Llama a la lógica del agente PandasAI, que devuelve un diccionario
    pandasai_output_dict = pandasai_agent.run_pandasai(query_to_run)
    logger_nodes.info(f"Resultado PandasAI Ejecutor: { {k: (type(v) if k=='pandasai_result' else v) for k, v in pandasai_output_dict.items()} }")

    # Devuelve el diccionario COMPLETO para actualizar el estado
    return pandasai_output_dict

# --- NODO EJECUTOR cONTEXTUALIZADOR ---
def run_contextualizer(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta el agente contextualizador (simplificado)."""
    logger_nodes.info("--- Ejecutando Nodo: Contextualizador ---")
    # Pasa el estado completo
    context_result = contextualizer_agent.contextualize(state)
    logger_nodes.info(f"Resultado Contextualizador: summary='{context_result.get('summary', '')[:50]}...'")
    # Devuelve solo los campos que modifica
    return {"summary": context_result.get("summary")}

# --- NODO EJECUTOR VALIDADOR ---
def run_validator(state: GraphState) -> Dict[str, Any]:
    logger_nodes.info("--- Ejecutando Nodo: Validador ---")
    original_query = state['original_query']
    summary = state.get('summary')
    plot_path = state.get('pandasai_plot_path') # Nombre correcto del estado
    pandasai_error = state.get('pandasai_error')

    # content_to_validate ya no es necesario pasarlo explícitamente aquí,
    # la lógica de validation_agent.validate usará 'summary' si no hay plot/error.

    final_text, final_image, error_msg = validation_agent.validate(
        original_query=original_query,
        summary_from_contextualizer=summary, # Pasa el summary
        pandasai_error=pandasai_error,         # Pasa el error de PandasAI
        plot_path_from_pandasai=plot_path    # Pasa la ruta del plot
    )

    logger_nodes.info(f"Resultado Validador: Texto={final_text is not None}, Imagen={final_image is not None}, Error={error_msg is not None}")
    return {
        "final_response_text": final_text,
        "final_response_image": final_image,
        "error_message": error_msg
    }