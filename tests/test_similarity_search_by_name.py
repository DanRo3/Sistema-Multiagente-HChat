# tests/test_faiss_query.py (O el nombre que estés usando)
import os
import pprint
import torch
from typing import List, Tuple, Optional, Dict, Any

# --- Dependencias Actualizadas ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuración ---
VECTOR_STORE_FOLDER = "vector_store_index"
VECTOR_STORE_INDEX_NAME = "data_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_CACHE_FOLDER = "./huggingface_cache"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NORMALIZE_EMBEDDINGS = True # ¡VERIFICA CONSISTENCIA CON CREACIÓN!

# --- Funciones (load_embeddings_model, load_vector_store, filter_results_by_metadata, test_query - SIN CAMBIOS) ---
# (Las funciones anteriores permanecen exactamente iguales que en la versión mejorada)
def load_embeddings_model() -> Optional[HuggingFaceEmbeddings]:
    """Carga el modelo de embeddings especificado usando langchain-huggingface."""
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME} en dispositivo {DEVICE}...")
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': NORMALIZE_EMBEDDINGS}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=HF_CACHE_FOLDER
        )
        print("Modelo de embeddings cargado exitosamente.")
        return embeddings
    except Exception as e:
        print(f"Error fatal al cargar el modelo de embeddings: {e}")
        return None

def load_vector_store(embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
    """Carga el índice FAISS local."""
    faiss_file_path = os.path.join(VECTOR_STORE_FOLDER, f"{VECTOR_STORE_INDEX_NAME}.faiss")
    pkl_file_path = os.path.join(VECTOR_STORE_FOLDER, f"{VECTOR_STORE_INDEX_NAME}.pkl")
    print(f"\nIntentando cargar índice FAISS desde: {VECTOR_STORE_FOLDER}/{VECTOR_STORE_INDEX_NAME}")
    if not os.path.exists(faiss_file_path) or not os.path.exists(pkl_file_path):
        print("--- ERROR ---")
        print("No se encontraron los archivos .faiss o .pkl en la ruta especificada.")
        return None
    try:
        print("Cargando índice local FAISS...")
        vector_store = FAISS.load_local(
            folder_path=VECTOR_STORE_FOLDER,
            embeddings=embeddings,
            index_name=VECTOR_STORE_INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        print("¡Índice FAISS cargado exitosamente!")
        return vector_store
    except ModuleNotFoundError as e:
        print(f"--- ERROR de Módulo al cargar FAISS ---: {e}")
        return None
    except Exception as e:
        print(f"--- ERROR inesperado al cargar el índice FAISS ---: {e}")
        return None

def filter_results_by_metadata(
    results_with_scores: List[Tuple[Any, float]],
    filter_criteria: Dict[str, Any]
) -> List[Tuple[Any, float]]:
    """Filtra una lista de resultados (documento, score) basada en metadatos (post-recuperación)."""
    if not filter_criteria:
        return results_with_scores
    filtered_results = []
    print(f"Filtrando resultados con: {filter_criteria}")
    for doc, score in results_with_scores:
        metadata = doc.metadata
        match = True
        for key, value in filter_criteria.items():
            if key not in metadata or str(metadata.get(key)) != str(value):
                match = False
                break
        if match:
            filtered_results.append((doc, score))
    print(f"Resultados después del filtrado: {len(filtered_results)}")
    return filtered_results

def test_query(
    vector_store: FAISS,
    query: str,
    k: int = 10,
    search_type: str = "similarity",
    filter_metadata: Optional[Dict[str, Any]] = None
):
    """Ejecuta una búsqueda en el vector store con opciones y muestra los resultados."""
    if not vector_store:
        print("El índice FAISS no está cargado. No se puede realizar la búsqueda.")
        return
    print(f"\n--- Ejecutando Búsqueda ---")
    print(f"Consulta: '{query}'")
    print(f"Método: {search_type}")
    print(f"Resultados iniciales a buscar (k): {k}")
    if filter_metadata:
        print(f"Filtro Post-Búsqueda: {filter_metadata}")
    results_with_scores = []
    try:
        if search_type == "similarity":
            results_with_scores = vector_store.similarity_search_with_score(query, k=k)
        elif search_type == "mmr":
            fetch_k = max(k * 5, 20)
            print(f"MMR fetch_k (candidatos iniciales): {fetch_k}")
            results_with_scores = vector_store.max_marginal_relevance_search_with_score(
                query, k=k, fetch_k=fetch_k
            )
        else:
            print(f"Error: Tipo de búsqueda '{search_type}' no soportado.")
            return
        print(f"\n--- Resultados Recuperados ({len(results_with_scores)}) ---")
        if filter_metadata:
            final_results = filter_results_by_metadata(results_with_scores, filter_metadata)
        else:
            final_results = results_with_scores
        print(f"\n--- Resultados Finales a Mostrar ({len(final_results)}) ---")
        if not final_results:
            print("No se encontraron resultados que coincidan.")
            return
        for i, (doc, score) in enumerate(final_results[:k]):
            print(f"\nResultado #{i+1} (Score: {score:.4f}):")
            print("-" * 20); print("Contenido (parsed_text):"); print(doc.page_content)
            print("\nMetadatos:"); pprint.pprint(doc.metadata, indent=2); print("-" * 20)
        if len(final_results) > k: print(f"... ({len(final_results) - k} más encontrados pero no mostrados)")
    except Exception as e:
        print(f"\n--- ERROR durante la búsqueda ---: {e}")

# --- Ejecución Principal ---
if __name__ == "__main__":
    print("--- Iniciando Test de Consulta FAISS (Mejorado) ---")
    embeddings_model = load_embeddings_model()
    if not embeddings_model:
        print("Fallo al cargar el modelo de embeddings. Abortando.")
    else:
        faiss_db = load_vector_store(embeddings_model)
        if not faiss_db:
            print("\nNo se pudo cargar el índice FAISS, la prueba no se ejecutará.")
        else:
            # --- Parámetros de la Prueba 1: Búsqueda Simple por Nombre ---
            test_query_text: str = "Litlejohn" # <-- SOLO EL NOMBRE
            number_of_results: int = 5        # <-- k reducido para esta prueba
            search_method: str = "similarity"
            metadata_filter: Optional[Dict[str, Any]] = None # <-- SIN FILTRO
            # -----------------------------------------------------------

            print("\n*** EJECUTANDO PRUEBA 1: BÚSQUEDA SIMPLE POR NOMBRE ***")
            test_query(
                vector_store=faiss_db,
                query=test_query_text,
                k=number_of_results,
                search_type=search_method,
                filter_metadata=metadata_filter
            )

    print("\n--- Fin del Test ---")