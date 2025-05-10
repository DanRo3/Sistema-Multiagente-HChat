# app/agents/pandasai_agent.py
import os
import pandas as pd
import logging
import time
import re # Aunque no se usa activamente, es bueno tenerlo si se debuggea código generado
from typing import Optional, Any, Tuple, Dict

# Importar SmartDataframe. El LLM específico de PandasAI (como pandasai.llm.OpenAI)
# se maneja a través del objeto LLM que pasamos en la configuración.
try:
    from pandasai import SmartDataframe
except ImportError:
    # Esto es crítico, así que registramos el error y relanzamos
    # para que la aplicación no continúe si PandasAI no está disponible.
    logging.getLogger(__name__).error("Falta la librería 'pandasai'. Ejecuta pip install pandasai")
    raise

# --- Importaciones de la Aplicación ---
from app.core.llm import get_llm # Para obtener el LLM configurado globalmente
from app.core.dataframe_loader import get_dataframe # Para cargar el DataFrame base
from app.core.config import settings # Para configuraciones (verbose, cache, save_charts, etc.)

logger = logging.getLogger(__name__)

# --- Singletons para el SmartDataframe y el LLM usado por PandasAI ---
# Esto evita reinicializar en cada llamada si la configuración no cambia.
_smart_df_instance: Optional[SmartDataframe] = None
_pandasai_llm_instance_cache = None # Cache para el LLM específico que usa PandasAI

def _initialize_pandasai_components() -> Optional[SmartDataframe]:
    global _smart_df_instance, _pandasai_llm_instance_cache
    if _smart_df_instance is not None:
        return _smart_df_instance
    logger.info("PandasAI Agent: Inicializando componentes (SmartDataframe)...")

    if _pandasai_llm_instance_cache is None:
        _pandasai_llm_instance_cache = get_llm()
        if not _pandasai_llm_instance_cache:
            logger.error("No se pudo obtener LLM para PandasAI.")
            return None
        logger.info("LLM obtenido/reutilizado para PandasAI.")

    base_df = get_dataframe()
    if base_df is None:
        logger.error("No se pudo obtener DataFrame base para PandasAI.")
        return None

    try:
        from pandasai import SmartDataframe

        chart_dir = settings.PANDASAI_CHART_DIR_NAME
        # ... (crear chart_dir) ...

        sdf_config = {
            "llm": _pandasai_llm_instance_cache,
            "verbose": settings.PANDASAI_VERBOSE,
            "enable_cache": settings.PANDASAI_ENABLE_CACHE,
            "save_charts": True,
            "save_charts_path": chart_dir,
            "max_retries": settings.PANDASAI_MAX_RETRIES,
        }

        _smart_df_instance = SmartDataframe(
            base_df.copy(), # <--- PASAR DataFrame DIRECTAMENTE
            config=sdf_config
        )
        logger.info("SmartDataframe inicializado (pasando DataFrame directamente).")
        return _smart_df_instance

    except ImportError:
        logger.error("Falta la librería 'pandasai'. Ejecuta pip install pandasai")
        raise
    except Exception as e:
        logger.exception(f"Error al inicializar SmartDataframe: {e}")
        _smart_df_instance = None
        return None

def run_pandasai(query: Optional[str]) -> Dict[str, Any]:
    """
    Ejecuta una consulta en lenguaje natural usando PandasAI y devuelve
    un diccionario estructurado con el resultado, tipo, ruta de plot y error.
    """
    # Estructura de salida por defecto
    output: Dict[str, Any] = {
        "pandasai_result": None,
        "pandasai_result_type": None,
        "pandasai_plot_path": None,
        "pandasai_error": None
    }

    # Validar la consulta de entrada
    if not query or not query.strip():
        logger.warning("PandasAI Agent: Consulta vacía proporcionada.")
        output["pandasai_error"] = "Consulta vacía."
        return output

    logger.info(f"PandasAI Agent: Ejecutando query: '{query}'")
    # Obtiene o inicializa la instancia de SmartDataframe
    smart_df = _initialize_pandasai_components()

    # Verificar si SmartDataframe se inicializó correctamente
    if smart_df is None:
        logger.error("PandasAI Agent: SmartDataframe no está disponible (falló la inicialización).")
        output["pandasai_error"] = "Error interno: Falla al inicializar el motor de PandasAI."
        return output

    start_time = time.time()
    try:
        # Ejecutar la consulta usando el método .chat() de SmartDataframe
        response = smart_df.chat(query)
        end_time = time.time()
        logger.info(f"PandasAI Agent: Respuesta recibida de smart_df.chat() en {end_time - start_time:.2f}s. Tipo de respuesta: {type(response)}")

        # --- Procesamiento de la Respuesta de PandasAI ---

        # Caso 1: PandasAI devuelve una ruta a un gráfico generado
        # (La condición verifica si el string 'response' contiene el nombre del dir de charts o termina en .png)
        if isinstance(response, str) and (settings.PANDASAI_CHART_DIR_NAME in response or response.endswith(".png")):
            output["pandasai_plot_path"] = response
            output["pandasai_result_type"] = "plot_path"
            logger.info(f"PandasAI devolvió una ruta de gráfico: {response}")

        # Caso 2: PandasAI devuelve un DataFrame o una Serie
        elif isinstance(response, (pd.DataFrame, pd.Series)):
            logger.info("Resultado es DataFrame/Series, convirtiendo a lista de diccionarios.")
            if isinstance(response, pd.Series):
                df_response = response.to_frame(name=response.name or 'resultado') # Asegurar nombre de columna
            else:
                df_response = response
            output["pandasai_result"] = df_response.to_dict(orient='records')
            output["pandasai_result_type"] = "dataframe_list"
            logger.info(f"  Convertido a lista de {len(output['pandasai_result'])} diccionarios.")

        # Caso 3: PandasAI devuelve tipos de datos Python estándar
        elif isinstance(response, (int, float, bool, list, dict, str)):
            output["pandasai_result"] = response
            output["pandasai_result_type"] = type(response).__name__.lower()
            logger.info(f"PandasAI devolvió un tipo estándar: {output['pandasai_result_type']}")

        # Caso 4: PandasAI devuelve None
        elif response is None:
             logger.warning("PandasAI devolvió None como respuesta.")
             output["pandasai_result"] = "PandasAI no pudo determinar una respuesta o el resultado fue nulo."
             output["pandasai_result_type"] = "string"

        # Caso 5: Tipo de respuesta inesperado
        else:
             logger.warning(f"PandasAI devolvió un tipo inesperado: {type(response)}. Convirtiendo a string.")
             output["pandasai_result"] = str(response)
             output["pandasai_result_type"] = "string"

    except Exception as e:
        end_time = time.time() # Registrar tiempo incluso en error
        logger.exception(f"PandasAI Agent: Error ({end_time - start_time:.2f}s) durante la ejecución de la consulta '{query}': {e}")
        # El traceback completo se loggea con logger.exception
        # Guardar un mensaje de error más conciso para el sistema
        output["pandasai_error"] = f"Error ejecutando la consulta con PandasAI: {str(e)[:300]}" # Limitar longitud

    # Loguear el resultado final del nodo de forma resumida
    log_output = {k: (type(v).__name__ if k == 'pandasai_result' and v is not None else v) for k, v in output.items()}
    logger.info(f"PandasAI Agent: Resultado final del nodo: {log_output}")
    return output