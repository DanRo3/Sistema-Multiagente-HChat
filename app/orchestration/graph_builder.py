from langgraph.graph import StateGraph, END
from app.orchestration.graph_state import GraphState # Importa la definición del estado
from app.orchestration import agent_nodes # Importa las funciones nodo que definimos

# Variable global para almacenar el grafo compilado (patrón singleton simple)
_compiled_graph = None

# --- Función de Decisión para la Bifurcación Condicional ---
def should_generate_code(state: GraphState) -> str:
    """
    Función de decisión para el borde condicional.
    Determina la siguiente ruta basada en si se necesita visualización
    y si hubo errores previos.

    Returns:
        El nombre del siguiente nodo a ejecutar ('to_python_agent' o 'to_validator').
    """
    print("--- Ejecutando Condición: ¿Necesita Visualización/Código? ---")
    needs_visualization = state.get('needs_visualization', False)
    python_code_exists = bool(state.get('python_code')) # Si reintentamos, podría ya existir
    execution_error = state.get('execution_error')

    if execution_error:
        # Si ya hubo un error en un intento previo de ejecución, no reintentar.
        # Ir directamente a validar para que el error sea formateado.
        print(f"Resultado Condición: Error de ejecución previo detectado -> Ir a Validador")
        return "to_validator"
    elif needs_visualization and not python_code_exists:
        # Si se necesita visualización y aún no se ha generado el código
        print(f"Resultado Condición: Necesita visualización y no hay código -> Ir a Agente Python")
        return "to_python_agent"
    else:
        # Si no se necesita visualización, o si ya se generó código (en un ciclo anterior?)
        # o si needs_visualization es False.
        print(f"Resultado Condición: No necesita visualización o código ya existe/no necesario -> Ir a Validador")
        return "to_validator"

# --- Función para Construir y Compilar el Grafo ---
def build_graph() -> StateGraph:
    """
    Construye el objeto StateGraph definiendo los nodos y las transiciones (edges).
    """
    print("Construyendo el grafo Langraph...")
    # Inicializar el grafo con el estado definido
    workflow = StateGraph(GraphState)

    # --- 1. Añadir Nodos ---
    # Asocia un nombre único a cada función nodo que importamos.
    print("Añadiendo nodos al grafo...")
    workflow.add_node("moderator", agent_nodes.run_moderator)
    workflow.add_node("retriever", agent_nodes.run_retriever)
    workflow.add_node("contextualizer", agent_nodes.run_contextualizer)
    workflow.add_node("python_agent", agent_nodes.run_python_agent) # Nodo que genera código
    workflow.add_node("code_executor", agent_nodes.run_code_executor) # Nodo que ejecuta código
    workflow.add_node("validator", agent_nodes.run_validator) # Nodo final de validación

    # --- 2. Definir el Punto de Entrada ---
    # Especifica por dónde comienza el flujo.
    print("Estableciendo punto de entrada: 'moderator'")
    workflow.set_entry_point("moderator")

    # --- 3. Añadir Bordes (Edges) Secuenciales ---
    # Define las transiciones directas entre nodos.
    print("Añadiendo bordes secuenciales...")
    workflow.add_edge("moderator", "retriever")
    workflow.add_edge("retriever", "contextualizer")
    # Después del contextualizador viene la decisión condicional.

    # --- 4. Añadir Borde Condicional ---
    # Desde 'contextualizer', el flujo depende del resultado de 'should_generate_code'.
    print("Añadiendo borde condicional desde 'contextualizer'...")
    workflow.add_conditional_edges(
        "contextualizer", # Nodo de origen de la condición
        should_generate_code, # Función que evalúa el estado y decide la ruta
        {
            # Mapeo: Si la función devuelve "to_python_agent", ir al nodo "python_agent".
            "to_python_agent": "python_agent",
            # Mapeo: Si la función devuelve "to_validator", ir al nodo "validator".
            "to_validator": "validator",
        }
    )

    # --- 5. Añadir Bordes después de la Rama de Código ---
    # Si se ejecutó la rama de código, conectar los nodos secuencialmente.
    print("Añadiendo bordes para la rama de generación/ejecución de código...")
    workflow.add_edge("python_agent", "code_executor")
    # IMPORTANTE: Después de ejecutar el código (incluso si falla), SIEMPRE
    # vamos al validador para que maneje el resultado o el error.
    workflow.add_edge("code_executor", "validator")

    # --- 6. Definir el Punto Final ---
    # El nodo 'validator' es el último paso lógico antes de terminar.
    # Conectamos 'validator' al nodo especial END.
    print("Añadiendo borde final desde 'validator' a END.")
    workflow.add_edge("validator", END)

    # --- 7. Compilar el Grafo ---
    # Crea el objeto ejecutable del grafo.
    print("Compilando el grafo...")
    compiled_workflow = workflow.compile()
    print("¡Grafo Langraph compilado exitosamente!")
    return compiled_workflow

# --- Función para Obtener el Grafo Compilado (Singleton) ---
def get_compiled_graph() -> StateGraph:
    """
    Obtiene la instancia compilada del grafo Langraph.
    Si no está compilado, lo compila la primera vez.
    """
    global _compiled_graph
    if _compiled_graph is None:
        print("Grafo no compilado. Iniciando compilación...")
        _compiled_graph = build_graph()
    # else:
    #     print("Devolviendo grafo compilado existente.") # Opcional: Log para ver si se reutiliza
    return _compiled_graph

# --- Bloque de prueba simple (opcional) ---
# if __name__ == '__main__':
#     # Esta prueba solo verifica que el grafo se compile sin errores.
#     # No lo ejecuta. La ejecución real se hará desde FastAPI.
#     print("--- Probando la Compilación del Grafo Langraph ---")
#     try:
#         test_graph = get_compiled_graph()
#         print("\nEstructura del Grafo (Nodos):", test_graph.nodes)
#         # Puedes intentar visualizarlo si tienes las dependencias opcionales
#         # try:
#         #     img_bytes = test_graph.get_graph().draw_mermaid_png()
#         #     with open("graph_structure.png", "wb") as f:
#         #         f.write(img_bytes)
#         #     print("Diagrama del grafo guardado como graph_structure.png")
#         # except Exception as draw_error:
#         #     print(f"No se pudo generar diagrama del grafo: {draw_error}")
#         #     print("Asegúrate de tener instaladas las dependencias opcionales: pip install pygraphviz matplotlib")

#     except Exception as e:
#         print(f"\nError durante la compilación o prueba del grafo: {e}")
#         import traceback
#         traceback.print_exc()