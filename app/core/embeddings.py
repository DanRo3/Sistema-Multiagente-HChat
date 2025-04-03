# app/core/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
import torch
from typing import Optional

_embeddings_model = None

def initialize_embeddings_model() -> Optional[HuggingFaceEmbeddings]:
    """Inicializa el modelo de embeddings (singleton)."""
    global _embeddings_model
    if _embeddings_model is None:
        print(f"Inicializando modelo de embeddings: {settings.EMBEDDING_MODEL_NAME}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando dispositivo para embeddings: {device}")

        model_kwargs = {'device': device}
        # --- ¡CRUCIAL: CONSISTENCIA! ---
        # Asegúrate que este valor coincida con la creación del índice.
        encode_kwargs = {'normalize_embeddings': True} # O False
        # --------------------------------

        try:
            _embeddings_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder="./huggingface_cache" # Opcional
            )
            print("Modelo de embeddings inicializado.")
        except Exception as e:
            print(f"--- ERROR ---")
            print(f"Error al inicializar el modelo de embeddings: {e}")
            _embeddings_model = None
    return _embeddings_model

def get_embeddings_model() -> Optional[HuggingFaceEmbeddings]:
    """Devuelve la instancia inicializada del modelo de embeddings."""
    if _embeddings_model is None:
        initialize_embeddings_model()
    return _embeddings_model