import os
import pprint # Para imprimir diccionarios de forma legible
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch


VECTOR_STORE_FOLDER = "vector_store_index"
VECTOR_STORE_INDEX_NAME = "data_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}



def load_embeddings_model():
    """Carga el modelo de embeddings especificado."""
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME} en dispositivo {device}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder="./huggingface_cache" # Opcional: Define una carpeta de caché
        )
        print("Modelo de embeddings cargado exitosamente.")
        return embeddings
    except Exception as e:
        print(f"Error fatal al cargar el modelo de embeddings: {e}")
        raise # Detener si el modelo no carga

def load_vector_store(embeddings):
    """Carga el índice FAISS local."""
    faiss_file_path = os.path.join(VECTOR_STORE_FOLDER, f"{VECTOR_STORE_INDEX_NAME}.faiss")
    pkl_file_path = os.path.join(VECTOR_STORE_FOLDER, f"{VECTOR_STORE_INDEX_NAME}.pkl")

    print(f"\nIntentando cargar índice FAISS desde: {VECTOR_STORE_FOLDER}/{VECTOR_STORE_INDEX_NAME}")

    if not os.path.exists(faiss_file_path) or not os.path.exists(pkl_file_path):
        print("--- ERROR ---")
        print("No se encontraron los archivos .faiss o .pkl en la ruta especificada.")
        print(f"Verifica que la carpeta '{VECTOR_STORE_FOLDER}' y los archivos")
        print(f"'{VECTOR_STORE_INDEX_NAME}.faiss' y '{VECTOR_STORE_INDEX_NAME}.pkl' existan.")
        return None

    try:
        print("Cargando índice local FAISS...")
        # Recordatorio: allow_dangerous_deserialization es necesario para archivos .pkl
        # generados por save_local. Es seguro porque tú generaste el índice.
        vector_store = FAISS.load_local(
            folder_path=VECTOR_STORE_FOLDER,
            embeddings=embeddings,
            index_name=VECTOR_STORE_INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        print("¡Índice FAISS cargado exitosamente!")
        return vector_store
    except ModuleNotFoundError as e:
        print(f"--- ERROR de Módulo al cargar FAISS ---")
        print(f"Parece que falta alguna dependencia necesaria para deserializar el índice: {e}")
        print("Asegúrate de tener instaladas las mismas librerías que usaste en Colab (pandas, numpy, etc.).")
        return None
    except Exception as e:
        print(f"--- ERROR inesperado al cargar el índice FAISS ---: {e}")
        return None

def test_query(vector_store, query: str, k: int = 5):
    """Ejecuta una búsqueda por similitud y muestra los resultados."""
    if not vector_store:
        print("El índice FAISS no está cargado. No se puede realizar la búsqueda.")
        return

    print(f"\n--- Ejecutando Búsqueda por Similitud ---")
    print(f"Consulta: '{query}'")
    print(f"Buscando los {k} documentos más similares...")

    try:
        # Realizar la búsqueda
        # results = vector_store.similarity_search(query, k=k)
        # O usar similarity_search_with_score para ver la puntuación de similitud
        results_with_scores = vector_store.similarity_search_with_score(query, k=k)

        print(f"\n--- Resultados Encontrados ({len(results_with_scores)}) ---")

        if not results_with_scores:
            print("No se encontraron resultados similares.")
            return

        for i, (doc, score) in enumerate(results_with_scores):
            print(f"\nResultado #{i+1} (Score: {score:.4f}):") # Formatear el score
            print("-" * 20)
            print("Contenido (parsed_text):")
            print(doc.page_content)
            print("\nMetadatos:")
            # Usar pprint para mostrar el diccionario de metadatos de forma ordenada
            pprint.pprint(doc.metadata, indent=2)
            print("-" * 20)

    except Exception as e:
        print(f"\n--- ERROR durante la búsqueda por similitud ---: {e}")

# --- Ejecución Principal ---
if __name__ == "__main__":
    print("--- Iniciando Test de Consulta FAISS ---")

    # 1. Cargar modelo de embeddings
    embeddings_model = load_embeddings_model()

    # 2. Cargar el índice FAISS
    # Solo proceder si el modelo de embeddings se cargó correctamente
    if embeddings_model:
        faiss_db = load_vector_store(embeddings_model)

        # 3. Realizar una consulta de prueba (si el índice cargó bien)
        if faiss_db:
            # --- ¡MODIFICA ESTA CONSULTA SEGÚN TUS DATOS! ---
            test_query_text = "1860"
            # O algo más específico si conoces tu dataset:
            # test_query_text = "Información sobre el cargamento del buque 'Titanic'" # Ejemplo ficticio
            # -------------------------------------------------

            number_of_results = 3 # Cuántos resultados quieres ver
            test_query(faiss_db, test_query_text, k=number_of_results)
        else:
            print("\nNo se pudo cargar el índice FAISS, la prueba de consulta no se ejecutará.")
    else:
        print("\nNo se pudo cargar el modelo de embeddings, la prueba no puede continuar.")

    print("\n--- Fin del Test ---")