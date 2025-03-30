# app/vector_store/faiss_store.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from .config import settings
from .embeddings import get_query_embedding 

_index = None
_metadata = None

def load_vector_store():
    """Carga el índice FAISS y los metadatos desde los archivos."""
    global _index, _metadata
    if _index is None:
        index_path = settings.faiss_index_path
        metadata_path = settings.faiss_metadata_path
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        print(f"Loading FAISS index from: {index_path}")
        _index = faiss.read_index(str(index_path))
        print(f"Index loaded. Total vectors: {_index.ntotal}")

        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            _metadata = pickle.load(f)
        print(f"Metadata loaded for {_len(_metadata)} items.") # Use len() defensively

def search_similar_documents(query: str, k: int = 5) -> list[dict]:
    """
    Busca documentos similares a una consulta en el índice FAISS.

    Args:
        query: La consulta del usuario en lenguaje natural.
        k: El número de documentos similares a devolver.

    Returns:
        Una lista de diccionarios, cada uno representando un documento encontrado,
        incluyendo al menos 'text', 'score' (distancia), y 'original_id'.
        Devuelve una lista vacía si el índice no está cargado o no se encuentran resultados.
    """
    global _index, _metadata
    if _index is None or _metadata is None:
        try:
            load_vector_store()
        except FileNotFoundError as e:
            print(f"Error loading vector store: {e}")
            return [] # Devuelve vacío si no se puede cargar
        except Exception as e:
            print(f"Unexpected error loading vector store: {e}")
            return []


    if _index.ntotal == 0:
        print("Warning: FAISS index is empty.")
        return []

    print(f"Searching for {k} documents similar to: '{query}'")
    query_embedding = get_query_embedding(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')

    # Realizar la búsqueda
    try:
        distances, indices = _index.search(query_embedding_np, k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    results = []
    if indices.size == 0 or distances.size == 0:
         print("No results found from FAISS search.")
         return []

    # Asegurarse de que indices y distances tengan la forma esperada (shape (1, k))
    if indices.ndim == 1:
        indices = indices.reshape(1, -1)
    if distances.ndim == 1:
        distances = distances.reshape(1, -1)


    for i in range(indices.shape[1]): # Iterar sobre los k resultados
        faiss_id = indices[0, i]
        distance = distances[0, i]

        # FAISS puede devolver -1 si hay menos de k resultados
        if faiss_id == -1:
            continue

        # Recuperar la información del documento usando el metadata
        doc_info = _metadata.get(faiss_id)
        if doc_info:
             results.append({
                 "text": doc_info.get('text', 'N/A'),
                 "score": float(distance), # Convertir a float estándar
                 "original_id": doc_info.get('original_id', 'N/A'),
                 "faiss_id": int(faiss_id) # ID interno de FAISS
             })
        else:
             print(f"Warning: Metadata not found for FAISS ID {faiss_id}")


    print(f"Found {len(results)} relevant documents.")
    return results

# Carga inicial al importar el módulo (opcional, pero común)
# try:
#     load_vector_store()
# except FileNotFoundError:
#     print("Vector store not found on initial load. Will try again on first search.")
# except Exception as e:
#      print(f"Error during initial vector store load: {e}")