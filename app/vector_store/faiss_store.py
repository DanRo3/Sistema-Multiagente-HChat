# app/vector_store/faiss_store.py
import os
from langchain_community.vectorstores import FAISS
from app.core.config import settings
from app.core.embeddings import get_embeddings_model # Importa desde tu módulo
from typing import Optional, List, Tuple, Any
from langchain_core.documents import Document # Para type hinting

_vector_store: Optional[FAISS] = None

def load_faiss_index() -> Optional[FAISS]:
    """Carga el índice FAISS desde la carpeta configurada (singleton)."""
    global _vector_store
    if _vector_store is not None:
        # print("Índice FAISS ya está cargado.") # Opcional: Evitar logs repetidos
        return _vector_store

    index_folder = settings.FAISS_INDEX_FOLDER
    index_name = settings.FAISS_INDEX_NAME
    faiss_file_path = os.path.join(index_folder, f"{index_name}.faiss")
    pkl_file_path = os.path.join(index_folder, f"{index_name}.pkl")

    print(f"Intentando cargar índice FAISS desde: {index_folder}/{index_name}")

    if not os.path.exists(faiss_file_path) or not os.path.exists(pkl_file_path):
        print(f"--- ERROR ---")
        print(f"No se encontraron los archivos del índice FAISS: {faiss_file_path}, {pkl_file_path}")
        return None

    try:
        embeddings = get_embeddings_model()
        if not embeddings:
            raise ValueError("Modelo de embeddings no disponible para cargar FAISS.")

        print("Cargando índice local FAISS...")
        loaded_db = FAISS.load_local(
            folder_path=index_folder,
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True # ¡Necesario para PKL!
        )
        print("¡Índice FAISS cargado exitosamente!")
        _vector_store = loaded_db
        return _vector_store

    except Exception as e:
        print(f"--- ERROR al cargar el índice FAISS ---: {e}")
        _vector_store = None
        return None

def get_faiss_db() -> Optional[FAISS]:
    """Devuelve la instancia cargada de la base de datos FAISS."""
    if _vector_store is None:
        load_faiss_index() # Intenta cargar si no lo está
    return _vector_store

# Función de búsqueda mejorada que usará el retriever agent
def search_documents(query: str, k: int = 20, filter_criteria: Optional[dict] = None) -> List[Document]:
    """
    Realiza búsqueda por similitud y aplica filtrado post-recuperación.
    Devuelve solo los documentos.
    """
    vector_store = get_faiss_db()
    if not vector_store:
        print("Error: Base de datos vectorial no disponible para búsqueda.")
        return []

    try:
        print(f"Buscando k={k} documentos para query: '{query[:50]}...'") # Log corto
        results_with_scores: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(query, k=k)

        if not filter_criteria:
            print(f"Devolviendo {len(results_with_scores)} resultados sin filtro.")
            return [doc for doc, score in results_with_scores]

        # Filtrado Post-recuperación
        print(f"Aplicando filtro: {filter_criteria}")
        filtered_docs = []
        for doc, score in results_with_scores:
            metadata = doc.metadata
            match = True
            for key, value in filter_criteria.items():
                if key not in metadata or str(metadata.get(key)).lower() != str(value).lower(): # Comparación insensible a mayúsculas
                    match = False
                    break
            if match:
                filtered_docs.append(doc)

        print(f"Devolviendo {len(filtered_docs)} resultados después del filtro.")
        return filtered_docs

    except Exception as e:
        print(f"Error durante la búsqueda de documentos: {e}")
        return []