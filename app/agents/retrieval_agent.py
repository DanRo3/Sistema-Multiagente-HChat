# app/agents/retrieval_agent.py
from typing import List, Optional, Dict, Any
from app.vector_store.faiss_store import search_documents
from langchain_core.documents import Document

# --- Configuración Específica del Recuperador ---
INITIAL_RETRIEVAL_K = 50 # Número de documentos a recuperar inicialmente de FAISS

def retrieve(search_query: Optional[str], filters: Optional[Dict[str, Any]]) -> List[Document]:
    """
    Recupera documentos relevantes de la base de datos vectorial FAISS
    basándose en una consulta de búsqueda y filtros de metadatos opcionales.

    Args:
        search_query: La cadena de texto optimizada para la búsqueda por similitud,
                      proporcionada por el agente moderador.
        filters: Un diccionario con los filtros de metadatos a aplicar
                 (post-recuperación), proporcionado por el agente moderador.

    Returns:
        Una lista de objetos Document de LangChain que coinciden con la búsqueda
        y los filtros. Puede ser una lista vacía si no se encuentran resultados
        o si ocurre un error.
    """
    # Validación de entrada
    if not search_query:
        print("Advertencia (Recuperador): No se proporcionó 'search_query'. No se realizará la búsqueda.")
        return [] # No buscar si no hay consulta

    print(f"Recuperador: Iniciando búsqueda...")
    print(f"  Query        : '{search_query}'")
    print(f"  Filtros      : {filters}")
    print(f"  K (inicial)  : {INITIAL_RETRIEVAL_K}")

    try:
        # Llamar a la función de búsqueda centralizada que ya definimos.
        # Esta función maneja internamente la obtención del índice FAISS,
        # la ejecución de similarity_search_with_score, y el filtrado
        # post-recuperación basado en los 'filters'.
        retrieved_docs: List[Document] = search_documents(
            query=search_query,
            k=INITIAL_RETRIEVAL_K,
            filter_criteria=filters # Pasa los filtros directamente
        )

        print(f"Recuperador: Búsqueda completada. Documentos encontrados (después de filtro): {len(retrieved_docs)}")
        return retrieved_docs

    except Exception as e:
        # Capturar cualquier error inesperado durante la búsqueda
        print(f"Error Inesperado en el agente de recuperación durante la búsqueda: {e}")
        import traceback
        traceback.print_exc() # Útil para depuración
        return [] # Devolver lista vacía en caso de error

# --- Bloque de prueba simple ---
#if __name__ == '__main__':
    # Para probar esto, necesitas que FAISS y los embeddings carguen correctamente.
    # Además, necesitarías simular la salida del moderador.
    #print("--- Probando Agente de Recuperación ---")

    # Simular salida del moderador (Ejemplo 1: Búsqueda simple)
    # mock_search_query_1 = "fragata americana Charles Edwin capitán Litlejohn"
    # mock_filters_1 = {"master_name": "Litlejohn"}
    # print(f"\nPrueba 1: Buscando '{mock_search_query_1}' con filtro {mock_filters_1}")
    # results_1 = retrieve(mock_search_query_1, mock_filters_1)
    # print(f"Prueba 1: Encontrados {len(results_1)} documentos.")
    # if results_1:
    #     print("Primer documento encontrado:")
    #     print("  Texto:", results_1[0].page_content[:200] + "...")
    #     print("  Metadatos:", results_1[0].metadata)

    # Simular salida del moderador (Ejemplo 2: Búsqueda genérica sin filtro)
    # mock_search_query_2 = "viajes desde Nueva York"
    # mock_filters_2 = None
    # print(f"\nPrueba 2: Buscando '{mock_search_query_2}' con filtro {mock_filters_2}")
    # results_2 = retrieve(mock_search_query_2, mock_filters_2)
    # print(f"Prueba 2: Encontrados {len(results_2)} documentos.")
    # Imprimir algunos resultados si se encuentran
    # for i, doc in enumerate(results_2[:3]):
    #      print(f"\nDocumento {i+1}:")
    #      print("  Texto:", doc.page_content[:200] + "...")
    #      print("  Metadatos:", doc.metadata)