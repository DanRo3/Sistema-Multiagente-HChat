# app/core/embeddings.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import settings
import torch # O import tensorflow, según tu backend

# Determinar el dispositivo (CPU o GPU si está disponible)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Embeddings: Using device: {device}")

# Variable global para cachear el modelo (opcional, mejora rendimiento en llamadas sucesivas)
_embedding_model = None

def get_embedding_model():
    """
    Carga y devuelve el modelo de embeddings de Hugging Face.
    Cachea el modelo para evitar recargarlo en cada llamada.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {settings.embedding_model_name}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True} # Normalizar es común para FAISS L2/IP
        )
        print("Embedding model loaded.")
    return _embedding_model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos."""
    model = get_embedding_model()
    return model.embed_documents(texts)

def get_query_embedding(text: str) -> list[float]:
    """Genera embedding para un único texto (consulta)."""
    model = get_embedding_model()
    return model.embed_query(text)

# Ejemplo de uso (puedes quitar esto después)
# if __name__ == "__main__":
#     test_texts = ["Hola mundo", "Inteligencia artificial"]
#     embeddings = get_embeddings(test_texts)
#     print(f"Generated {len(embeddings)} embeddings.")
#     print(f"Dimension: {len(embeddings[0])}")
#     query_emb = get_query_embedding("¿Qué es AI?")
#     print(f"Query embedding dimension: {len(query_emb)}")