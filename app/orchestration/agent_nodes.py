# app/orchestration/agent_nodes.py

from typing import Dict, Any
from app.orchestration.graph_state import GraphState

# Importar las funciones lógicas de cada agente
from app.agents import moderator_agent
from app.agents import retrieval_agent
from app.agents import contextualizer_agent
from app.agents import python_agent
from app.utils import code_executor # El ejecutor está en utils
from app.agents import validation_agent

# --- Funciones Nodo para Langraph ---

def run_moderator(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta el agente moderador."""
    print("--- Ejecutando Nodo: Moderador ---")
    query = state['original_query']
    # Llama a la lógica del agente moderador
    analysis_result = moderator_agent.analyze_query(query)
    print(f"Resultado Moderador: {analysis_result}")
    # Devuelve solo los campos del estado que este agente modifica
    return {
        "intent": analysis_result.get("intent"),
        "filters": analysis_result.get("filters"),
        "search_query": analysis_result.get("search_query")
    }

def run_retriever(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta el agente de recuperación."""
    print("--- Ejecutando Nodo: Recuperador ---")
    search_query = state.get('search_query')
    filters = state.get('filters')
    # Llama a la lógica del agente recuperador
    retrieved_docs = retrieval_agent.retrieve(search_query, filters)
    print(f"Documentos recuperados: {len(retrieved_docs)}")
    # Devuelve solo los campos que modifica
    return {"retrieved_docs": retrieved_docs}

def run_contextualizer(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta el agente contextualizador."""
    print("--- Ejecutando Nodo: Contextualizador ---")
    original_query = state['original_query']
    intent = state.get('intent')
    retrieved_docs = state.get('retrieved_docs')
    # Llama a la lógica del agente contextualizador
    context_result = contextualizer_agent.contextualize(original_query, intent, retrieved_docs)
    print(f"Resultado Contextualizador: summary='{context_result.get('summary', '')[:50]}...', needs_viz={context_result.get('needs_visualization')}")
    # Devuelve los campos modificados
    return {
        "summary": context_result.get("summary"),
        "needs_visualization": context_result.get("needs_visualization", False),
        "data_for_python": context_result.get("data_for_python")
    }

def run_python_agent(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta el agente generador de código Python."""
    print("--- Ejecutando Nodo: Agente Python (Generador Código) ---")
    original_query = state['original_query']
    data_for_python = state.get('data_for_python')
    # Llama a la lógica del agente Python
    code_result = python_agent.generate_python_code(original_query, data_for_python)
    print(f"Resultado Agente Python: Código generado -> {'Sí' if code_result.get('python_code') else 'No'}")
    # Devuelve el campo modificado
    return {"python_code": code_result.get("python_code")}

def run_code_executor(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta el código Python generado."""
    print("--- Ejecutando Nodo: Ejecutor de Código ---")
    python_code = state.get('python_code')
    # Llama a la utilidad de ejecución segura
    output, error = code_executor.execute_python_safely(python_code)
    print(f"Resultado Ejecutor: Output -> {'Presente' if output else 'Ausente'}, Error -> {'Presente' if error else 'Ausente'}")
    # Devuelve los campos modificados
    return {"execution_output": output, "execution_error": error}

def run_validator(state: GraphState) -> Dict[str, Any]:
    """Nodo que ejecuta el agente de validación final."""
    print("--- Ejecutando Nodo: Validador ---")
    original_query = state['original_query']
    execution_error = state.get('execution_error')

    # Determinar qué contenido validar basado en el flujo
    if state.get('needs_visualization') and not execution_error:
        # Si se intentó visualizar y no hubo error de ejecución, validar la salida del ejecutor
        content_to_validate = state.get('execution_output')
        print("Validando salida del ejecutor de código...")
    elif execution_error:
         # Si hubo error de ejecución, pasarlo para que el validador lo formatee
         content_to_validate = None # No hay contenido válido que validar
         print(f"Pasando error de ejecución al validador: {execution_error}")
    else:
        # Si no se necesitaba visualización o falló antes, validar el resumen del contextualizador
        content_to_validate = state.get('summary')
        print("Validando resumen del contextualizador...")


    # Llama a la lógica del agente validador
    final_text, final_image, error_msg = validation_agent.validate(
        original_query,
        content_to_validate,
        execution_error # Pasa el error de ejecución explícitamente
    )
    print(f"Resultado Validador: Texto -> {'Sí' if final_text else 'No'}, Imagen -> {'Sí' if final_image else 'No'}, Error -> {'Sí' if error_msg else 'No'}")
    # Devuelve los campos finales que se usarán en la respuesta API
    return {
        "final_response_text": final_text,
        "final_response_image": final_image,
        "error_message": error_msg
    }