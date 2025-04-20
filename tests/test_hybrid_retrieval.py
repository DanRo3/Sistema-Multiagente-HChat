# test_hybrid_retrieval.py
import pandas as pd
import os
import pprint
import logging
from dotenv import load_dotenv
from typing import Optional 
# --- Componentes FAISS ---
from langchain_community.vectorstores import FAISS
# Usa la importación actualizada para embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# --- Componentes PandasAI (v2) ---
from pandasai import SmartDataframe
# Importa el LLM wrapper específico de PandasAI o Langchain
# Ejemplo con OpenAI (requiere langchain-openai)
from pandasai.llm import OpenAI # O from langchain_openai import ChatOpenAI
# Ejemplo con Gemini (requiere langchain-google-genai)
# from pandasai.llm import GoogleGemini # O from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Paths y Nombres (AJUSTA SEGÚN TU PROYECTO)
CSV_FILE_PATH = "data/DataLimpia.csv" # Ruta a tu CSV original
VECTOR_STORE_FOLDER = "vector_store_index"
VECTOR_STORE_INDEX_NAME = "data_index"

# Configuración Embeddings (DEBE COINCIDIR CON CREACIÓN DE ÍNDICE)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NORMALIZE_EMBEDDINGS = True

# Configuración LLM para PandasAI (ELIGE UNO Y CONFIGURA)
# Opción 1: OpenAI
PANDASAI_LLM_PROVIDER = "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o-mini" 
# Opción 2: Google Gemini
# PANDASAI_LLM_PROVIDER = "google"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

# --- Fin Configuración ---


def load_dataframe(csv_path: str) -> Optional[pd.DataFrame]:
    """Carga los datos desde el archivo CSV a un DataFrame de Pandas."""
    logging.info(f"Cargando DataFrame desde: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        # Limpieza básica opcional (manejar NaNs, convertir tipos si es necesario)
        # df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
        # df = df.fillna('') # Llenar NaNs con strings vacíos podría ser necesario para PandasAI
        logging.info(f"DataFrame cargado exitosamente. Columnas: {df.columns.tolist()}")
        logging.info(f"Número de filas: {len(df)}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo CSV en {csv_path}")
        return None
    except Exception as e:
        logging.exception(f"Error inesperado al cargar el DataFrame: {e}")
        return None

# --- Cargar DataFrame ---
dataframe = load_dataframe(CSV_FILE_PATH)

# (Continuación de test_hybrid_retrieval.py)

def initialize_pandasai_llm():
    """Inicializa el LLM configurado para PandasAI."""
    logging.info(f"Inicializando LLM para PandasAI (Proveedor: {PANDASAI_LLM_PROVIDER})")
    if PANDASAI_LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            logging.error("OPENAI_API_KEY no encontrada en el entorno.")
            return None
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)
        except Exception as e:
            logging.exception(f"Error inicializando OpenAI para PandasAI: {e}")
            return None
    elif PANDASAI_LLM_PROVIDER == "google":
        if not GEMINI_API_KEY:
            logging.error("GEMINI_API_KEY no encontrada en el entorno.")
            return None
        try:
            # return GoogleGemini(api_key=GEMINI_API_KEY, model=GEMINI_MODEL_NAME)
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Nota: Es posible que necesites configurar genai primero si no lo hace automáticamente
            # import google.generativeai as genai
            # genai.configure(api_key=GEMINI_API_KEY)
            return ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0, convert_system_message_to_human=True)
        except Exception as e:
            logging.exception(f"Error inicializando Gemini para PandasAI: {e}")
            return None
    else:
        logging.error(f"Proveedor LLM no soportado para PandasAI: {PANDASAI_LLM_PROVIDER}")
        return None

# --- Inicializar LLM y SmartDataframe ---
smart_df = None
if dataframe is not None:
    pandasai_llm = initialize_pandasai_llm()
    if pandasai_llm:
        try:
            # Crear el SmartDataframe (Wrapper de PandasAI sobre tu DataFrame)
            # Puedes pasarle una configuración con descripción del DF para ayudar al LLM
            smart_df = SmartDataframe(
                dataframe,
                config={"llm": pandasai_llm, "verbose": True, "enable_cache": False}
                # Opcional: descripción para ayudar al LLM
                # description="Tabla con registros históricos de llegadas de barcos al puerto de La Habana."
            )
            logging.info("SmartDataframe de PandasAI inicializado correctamente.")
        except Exception as e:
            logging.exception(f"Error al crear SmartDataframe: {e}")
            smart_df = None # Asegurarse de que sea None si falla
    else:
        logging.warning("No se pudo inicializar el LLM para PandasAI. Las pruebas de PandasAI no se ejecutarán.")
else:
    logging.error("No se pudo cargar el DataFrame. Las pruebas no pueden continuar.")


def load_faiss_vector_store() -> Optional[FAISS]:
    """Carga el índice FAISS local."""
    logging.info("Cargando modelo de embeddings para FAISS...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': NORMALIZE_EMBEDDINGS}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            # cache_folder= # Opcional
        )
        logging.info("Modelo de embeddings cargado.")
    except Exception as e:
        logging.exception(f"Error fatal al cargar modelo de embeddings: {e}")
        return None

    logging.info(f"Intentando cargar índice FAISS desde: {VECTOR_STORE_FOLDER}/{VECTOR_STORE_INDEX_NAME}")
    faiss_file_path = os.path.join(VECTOR_STORE_FOLDER, f"{VECTOR_STORE_INDEX_NAME}.faiss")
    pkl_file_path = os.path.join(VECTOR_STORE_FOLDER, f"{VECTOR_STORE_INDEX_NAME}.pkl")

    if not os.path.exists(faiss_file_path) or not os.path.exists(pkl_file_path):
        logging.error("--- ERROR --- Archivos de índice FAISS no encontrados.")
        return None

    try:
        vector_store = FAISS.load_local(
            folder_path=VECTOR_STORE_FOLDER,
            embeddings=embeddings,
            index_name=VECTOR_STORE_INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        logging.info("¡Índice FAISS cargado exitosamente!")
        return vector_store
    except Exception as e:
        logging.exception(f"--- ERROR inesperado al cargar el índice FAISS ---: {e}")
        return None

# --- Cargar FAISS ---
faiss_vs = load_faiss_vector_store()


# --- Consultas de Prueba ---
# Consulta ideal para PandasAI (estructurada, agregación)
query_structured = "Cuenta cuántos barcos diferentes llegaron en 1860. Devuelve solo el número."
# query_structured = "What was the average travel_duration in days for ships arriving in March 1858? Extract the number of days first." # Necesita preprocesar 'travel_duration'
# query_structured = "List the ship_name and master_name for all 'vapor' type ships departing from 'New Orleans'."

# Consulta ideal para FAISS (semántica, sobre texto libre)
query_semantic = "Buscar registros que mencionen problemas o retrasos durante el viaje"
# query_semantic = "Find entries describing perishable cargo like fruits or livestock"

# Consulta potencialmente híbrida o ambigua
query_ambiguous = "¿Qué barcos de Barcelona llegaron en 1860 y qué carga traían?"

test_queries = {
    "Structured (PandasAI Target)": query_structured,
    "Semantic (FAISS Target)": query_semantic,
    "Ambiguous/Hybrid": query_ambiguous
}

import time # Para medir tiempo simple

# --- Ejecución de Pruebas ---
results_log = {} # Para almacenar resultados

for name, query in test_queries.items():
    print(f"\n{'='*20} INICIANDO PRUEBA: [{name}] {'='*20}")
    print(f"Consulta: {query}")
    results_log[name] = {"query": query}

    # --- Ejecutar con PandasAI ---
    if smart_df:
        print(f"\n--- Ejecutando con PandasAI ---")
        start_time = time.time()
        try:
            # Usar .chat() para enviar la consulta
            response = smart_df.chat(query)
            end_time = time.time()
            print(f"[PandasAI] Respuesta ({end_time - start_time:.2f}s):")
            # La respuesta puede ser un string, un número, un DataFrame de Pandas, etc.
            print(type(response))
            print(response)
            results_log[name]["pandasai_result"] = str(response) # Guardar como string
            results_log[name]["pandasai_error"] = None
            results_log[name]["pandasai_time"] = end_time - start_time
        except Exception as e:
            end_time = time.time()
            print(f"[PandasAI] ERROR ({end_time - start_time:.2f}s): {e}")
            results_log[name]["pandasai_result"] = None
            results_log[name]["pandasai_error"] = str(e)
            results_log[name]["pandasai_time"] = end_time - start_time
    else:
        print("\n--- PandasAI no inicializado, omitiendo ejecución ---")
        results_log[name]["pandasai_result"] = "SKIPPED"
        results_log[name]["pandasai_error"] = None
        results_log[name]["pandasai_time"] = 0

    # --- Ejecutar con FAISS ---
    if faiss_vs:
        print(f"\n--- Ejecutando con FAISS (Búsqueda Vectorial) ---")
        k = 5 # Buscar los 5 más similares
        start_time = time.time()
        try:
            # Buscar por similitud
            search_results = faiss_vs.similarity_search_with_score(query, k=k)
            end_time = time.time()
            print(f"[FAISS] Resultados ({end_time - start_time:.2f}s):")
            faiss_output = []
            if search_results:
                for i, (doc, score) in enumerate(search_results):
                    print(f"\n  Resultado FAISS #{i+1} (Score: {score:.4f}):")
                    print(f"  Texto: {doc.page_content[:200]}...")
                    print(f"  Metadatos: {pprint.pformat(doc.metadata, indent=2, width=100)}")
                    faiss_output.append({
                        "score": score,
                        "text": doc.page_content,
                        "metadata": doc.metadata
                    })
            else:
                print("  No se encontraron resultados similares.")
            results_log[name]["faiss_result"] = faiss_output
            results_log[name]["faiss_error"] = None
            results_log[name]["faiss_time"] = end_time - start_time
        except Exception as e:
            end_time = time.time()
            print(f"[FAISS] ERROR ({end_time - start_time:.2f}s): {e}")
            results_log[name]["faiss_result"] = None
            results_log[name]["faiss_error"] = str(e)
            results_log[name]["faiss_time"] = end_time - start_time
    else:
        print("\n--- FAISS no inicializado, omitiendo ejecución ---")
        results_log[name]["faiss_result"] = "SKIPPED"
        results_log[name]["faiss_error"] = None
        results_log[name]["faiss_time"] = 0

    print(f"\n{'='*20} FIN PRUEBA: [{name}] {'='*20}")


# --- Análisis Final (Manual) ---
print("\n\n===== ANÁLISIS DE RESULTADOS (Manual) =====")
# Imprimir el log de resultados para facilitar comparación
# pprint.pprint(results_log, indent=2)

for test_name, results in results_log.items():
    print(f"\n--- Análisis para: [{test_name}] ---")
    print(f"Consulta: {results['query']}")
    print("\nResultado PandasAI:")
    if results.get("pandasai_error"): print(f"  ERROR: {results['pandasai_error']}")
    elif results.get("pandasai_result") == "SKIPPED": print("  Omitido.")
    else: print(f"  {results.get('pandasai_result')}")
    print(f"  Tiempo: {results.get('pandasai_time'):.2f}s")

    print("\nResultado FAISS (Top 1 si existe):")
    if results.get("faiss_error"): print(f"  ERROR: {results['faiss_error']}")
    elif results.get("faiss_result") == "SKIPPED": print("  Omitido.")
    elif results.get("faiss_result"):
        top_res = results['faiss_result'][0]
        print(f"  Score: {top_res['score']:.4f}")
        print(f"  Texto: {top_res['text'][:200]}...")
        # print(f"  Metadatos: {top_res['metadata']}") # Puede ser muy largo
    else: print("  Sin resultados.")
    print(f"  Tiempo: {results.get('faiss_time'):.2f}s")

print("\n===== FIN DEL ANÁLISIS =====")
print("\nEvalúa manualmente:")
print("1. ¿Qué sistema fue más preciso para cada tipo de consulta (estructurada vs semántica)?")
print("2. ¿Cómo manejó cada sistema la consulta diseñada para el *otro* sistema?")
print("3. ¿Cómo manejaron la consulta ambigua?")
print("4. ¿Hubo diferencias notables de tiempo?")
print("5. ¿Qué tipo de errores ocurrieron?")