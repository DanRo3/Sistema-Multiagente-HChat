# app/agents/pandasai_agent.py
import os
import pandas as pd
import logging
import time
from typing import Optional, Any, Dict, List # Añadido List para type hinting

# Importar SmartDataframe
try:
    from pandasai import SmartDataframe
    from pandasai.connectors import PandasConnector # Necesario para pasar field_descriptions
except ImportError:
    logging.getLogger(__name__).error("Falta la librería 'pandasai' o 'PandasConnector'. Ejecuta pip install pandasai")
    raise

# --- Importaciones de la Aplicación ---
from app.core.llm import get_llm
from app.core.dataframe_loader import get_dataframe
from app.core.config import settings
from app.pandasai_utils.response_parsers import FullDataFrameResponseParser # Asegúrate que esta ruta sea correcta
from app.pandasai_utils.skills import plot_top_n_frequencies, get_tabular_data

logger = logging.getLogger(__name__)

# --- Definición de Field Descriptions ---
# Mueve esto a un archivo de configuración o un módulo dedicado si se vuelve muy extenso
FIELD_DESCRIPTIONS: Dict[str, str] = {
    'publication_date': 'Fecha de publicación de la noticia en el diario (formato YYYY-MM-DD). Ejemplo: 1851-12-04.',
    'news_section': 'Sección del periódico donde apareció la noticia (ej: "E" para Entradas, "S" para Salidas).',
    'travel_departure_date': 'Fecha de salida del barco del puerto de origen (formato YYYY-MM-DD).',
    'travel_duration': "Duración original del viaje como texto (ej. '6dine', '10 dias'). Usar 'travel_duration_days' para cálculos numéricos.",
    'travel_arrival_date': 'Fecha de llegada del barco al puerto de destino (formato YYYY-MM-DD).',
    'travel_departure_port': 'Puerto de salida del barco.',
    'travel_port_of_call_list': 'Lista de puertos intermedios visitados. Puede ser "nan" si no hay datos.',
    'travel_arrival_port': 'Puerto de llegada del barco. Generalmente "La Habana".',
    'ship_type': "Tipo de barco, usualmente abreviado (ej. 'berg. am.', 'frag. esp.'). Para análisis, usar estas abreviaturas. El Agente Contextualizador puede mapearlas a nombres completos para el usuario final.",
    'ship_name': 'Nombre del barco.',
    'cargo_list': 'Descripción textual de la carga transportada y otros detalles del manifiesto. Para buscar un tipo de carga específico (ej. "cacao", "azúcar"), buscar la palabra clave dentro de este texto.',
    'master_role': 'Rol del capitán o maestro del barco (ej. "(c)" para capitán).',
    'master_name': 'Nombre del capitán o maestro del barco.',
    'parsed_text': 'Texto completo original del registro marítimo. Es la fuente más detallada y puede usarse para búsquedas abiertas o cuando la información no se encuentra en campos específicos.',
    'travel_duration_days': 'Duración del viaje expresada únicamente en días (numérico). Preferir esta columna para cálculos de duración.'
}
logger.info(f"Cargadas {len(FIELD_DESCRIPTIONS)} descripciones de campos para PandasAI.")

# --- Singletons para el SmartDataframe y el LLM usado por PandasAI ---
_smart_df_instance: Optional[SmartDataframe] = None
_pandasai_llm_instance_cache = None

def _initialize_pandasai_components() -> Optional[SmartDataframe]:
    global _smart_df_instance, _pandasai_llm_instance_cache
    
    # 1. Comprobar si la instancia ya existe y las configs son las mismas
    if _smart_df_instance is not None and \
       _pandasai_llm_instance_cache is not None and \
       _pandasai_llm_instance_cache.temperature == settings.PANDASAI_TEMPERATURE and \
       (not hasattr(_pandasai_llm_instance_cache, 'model_kwargs') or \
        getattr(_pandasai_llm_instance_cache.model_kwargs, 'get', lambda k,d: d)('seed', None) == settings.PANDASAI_SEED):
        logger.debug("Reutilizando instancia existente de SmartDataframe y LLM para PandasAI.")
        # Asegurarse que los skills estén registrados incluso si se reutiliza la instancia
        # (Podría ser redundante si no cambian, pero es seguro)
        try:
            if hasattr(_smart_df_instance, 'add_skills'):
                # Podrías tener una bandera para no añadirlos múltiples veces si no es necesario
                # o si add_skills maneja duplicados internamente (revisar doc de PandasAI)
                _smart_df_instance.add_skills(get_tabular_data, plot_top_n_frequencies)
                logger.debug("Skills verificados/re-añadidos a instancia de SmartDataframe cacheada.")
            else:
                logger.warning("Instancia cacheada de SmartDataframe no tiene 'add_skills'.")
        except Exception as e_skill_cache:
            logger.error(f"Error añadiendo skills a instancia cacheada: {e_skill_cache}")
        return _smart_df_instance
    
    logger.info("PandasAI Agent: Inicializando componentes (SmartDataframe)...")

    # Obtener LLM configurado con temperatura y seed desde settings
    # Esto asegura que si las settings cambian, el LLM se reinstancia.
    current_llm = get_llm(
        temperature=settings.PANDASAI_TEMPERATURE,
        seed=settings.PANDASAI_SEED
    )
    if not current_llm:
        logger.error("No se pudo obtener LLM para PandasAI.")
        _smart_df_instance = None # Asegurar que se resetee si falla
        return None
    _pandasai_llm_instance_cache = current_llm # Actualizar caché del LLM
    logger.info(f"LLM obtenido/reutilizado para PandasAI con temp={settings.PANDASAI_TEMPERATURE}, seed={settings.PANDASAI_SEED}.")

    base_df = get_dataframe()
    if base_df is None:
        logger.error("No se pudo obtener DataFrame base para PandasAI.")
        _smart_df_instance = None
        return None
    logger.info(f"DataFrame base obtenido con {len(base_df)} filas.")

    try:
        chart_dir = settings.PANDASAI_CHART_DIR_NAME
        if not os.path.exists(chart_dir):
            os.makedirs(chart_dir)
            logger.info(f"Directorio de gráficos PandasAI creado: {chart_dir}")

        # Crear el PandasConnector con el DataFrame y field_descriptions
        # Usamos una copia del DataFrame para evitar modificaciones accidentales a la instancia global.
        connector = PandasConnector(
            {"original_df": base_df.copy()}, # PandasAI v2 espera un dict de DataFrames
            field_descriptions=FIELD_DESCRIPTIONS,
            name="HistoricoMaritimoConnector"
        )
        logger.info(f"PandasConnector inicializado con {len(FIELD_DESCRIPTIONS)} descripciones de campos.")

        sdf_config: Dict[str, Any] = {
            "llm": _pandasai_llm_instance_cache,
            "verbose": settings.PANDASAI_VERBOSE,
            "enable_cache": settings.PANDASAI_ENABLE_CACHE,
            "save_charts": True, # Permitir a PandasAI guardar gráficos si la query lo indica
            "save_charts_path": chart_dir,
            "max_retries": settings.PANDASAI_MAX_RETRIES,
            "response_parser": FullDataFrameResponseParser, # Usar nuestro parser personalizado
            # PandasAI v2 ya no usa 'language' directamente en la config general del SmartDataframe.
            # Se gestiona a través del LLM o de los prompts.
            # "custom_whitelisted_dependencies": [], # Añadir si se usan skills con dependencias no estándar
        }

        _smart_df_instance = SmartDataframe(
            connector, # Pasar el conector
            config=sdf_config
        )
        logger.info(f"SmartDataframe inicializado con PandasConnector y configuraciones: {sdf_config}")
        return _smart_df_instance

    except ImportError: # Ya se maneja al inicio del archivo, pero por si acaso.
        logger.error("Falta la librería 'pandasai'. Ejecuta pip install pandasai")
        raise
    except Exception as e:
        logger.exception(f"Error crítico al inicializar SmartDataframe: {e}")
        _smart_df_instance = None # Resetear en caso de fallo
        return None

def run_pandasai(query: Optional[str]) -> Dict[str, Any]:
    """
    Ejecuta una consulta en lenguaje natural usando PandasAI y devuelve
    un diccionario estructurado con el resultado, tipo, ruta de plot y error.
    """
    output: Dict[str, Any] = {
        "pandasai_result": None,
        "pandasai_result_type": None,
        "pandasai_plot_path": None,
        "pandasai_error": None
    }

    if not query or not query.strip():
        logger.warning("PandasAI Agent: Consulta vacía proporcionada.")
        output["pandasai_error"] = "Consulta vacía."
        return output

    # El Agente Moderador es responsable de la calidad y el formato de esta 'query'.
    # Si se necesita que PandasAI responda en un idioma específico, el Moderador
    # debe incluir esa instrucción en la 'query' que se pasa aquí.
    # Ejemplo: "Responde en español. {query_del_moderador_para_pandasai}"
    logger.info(f"PandasAI Agent: Ejecutando query (recibida del Moderador): '{query}'")
    
    smart_df = _initialize_pandasai_components()
    if smart_df is None:
        logger.error("PandasAI Agent: SmartDataframe no está disponible (falló la inicialización).")
        output["pandasai_error"] = "Error interno: Falla al inicializar el motor de PandasAI."
        return output

    start_time = time.time()
    try:
        # La respuesta ya vendrá procesada por FullDataFrameResponseParser
        response_data: Any = smart_df.chat(query)
        end_time = time.time()
        
        logger.info(f"PandasAI Agent: Respuesta recibida de smart_df.chat() (post-parser) en {end_time - start_time:.2f}s. Tipo: {type(response_data)}")

        # Procesamiento de la respuesta (ya formateada por el ResponseParser)
        if isinstance(response_data, str) and (settings.PANDASAI_CHART_DIR_NAME in response_data or response_data.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf"))):
            output["pandasai_plot_path"] = response_data
            output["pandasai_result_type"] = "plot_path"
            # El Contextualizador puede generar un mensaje más elaborado si lo desea.
            output["pandasai_result"] = f"Se generó un gráfico y se guardó en: {response_data}" 
            logger.info(f"PandasAI (post-parser) devolvió una ruta de gráfico: {response_data}")
        
        elif isinstance(response_data, list): # Asumimos lista de diccionarios del parser
            output["pandasai_result"] = response_data
            output["pandasai_result_type"] = "list_of_dicts"
            count = len(response_data)
            logger.info(f"PandasAI (post-parser) devolvió una lista con {count} elementos.")
            if count > 0 and not isinstance(response_data[0], dict):
                logger.warning("La lista devuelta no contiene diccionarios como se esperaba.")
        
        elif isinstance(response_data, (str, int, float, bool, dict)):
            output["pandasai_result"] = response_data
            output["pandasai_result_type"] = type(response_data).__name__.lower()
            logger.info(f"PandasAI (post-parser) devolvió un tipo estándar: {output['pandasai_result_type']}, valor: {str(response_data)[:200]}...")

        elif response_data is None:
             logger.warning("PandasAI (post-parser) devolvió None como respuesta.")
             output["pandasai_result"] = None # El Contextualizador decidirá cómo presentarlo
             output["pandasai_result_type"] = "none"
        
        else: 
             logger.warning(f"PandasAI (post-parser) devolvió un tipo inesperado: {type(response_data)}. Se intentará convertir a string.")
             try:
                 output["pandasai_result"] = str(response_data)
             except Exception as str_conv_error:
                 logger.error(f"No se pudo convertir el tipo inesperado {type(response_data)} a string: {str_conv_error}")
                 output["pandasai_result"] = f"Respuesta de tipo no manejable: {type(response_data)}"
             output["pandasai_result_type"] = "string_fallback"

    except Exception as e:
        end_time = time.time()
        logger.exception(f"PandasAI Agent: Error ({end_time - start_time:.2f}s) durante la ejecución de la consulta '{query}': {e}")
        output["pandasai_error"] = f"Error ejecutando la consulta con PandasAI: {str(e)[:300]}"

    # Loguear el resultado final del nodo de forma resumida
    log_summary: Dict[str, Any] = {}
    for k, v in output.items():
        if k == 'pandasai_result' and v is not None:
            if isinstance(v, list):
                log_summary[k] = f"list_of_dicts (len={len(v)})"
            else:
                log_summary[k] = f"{type(v).__name__} (value_snippet='{str(v)[:50]}...')"
        else:
            log_summary[k] = v
    logger.info(f"PandasAI Agent: Salida del nodo: {log_summary}")
    
    return output

# Función para limpiar la caché de PandasAI si es necesario (opcional)
def clear_pandasai_cache_if_enabled():
    if settings.PANDASAI_ENABLE_CACHE:
        try:
            # PandasAI puede tener su propia forma de limpiar la caché,
            # o podrías eliminar el directorio .pandasai-cache manualmente.
            # Por ahora, un placeholder. La forma más segura es a través de la API de PandasAI si existe.
            # from pandasai.helpers.cache import Cache  # Esto puede no ser público o cambiar
            # Cache().delete() # Ejemplo hipotético
            
            # Si PandasAI usa un archivo .db como antes:
            cache_path = ".pandasai-cache/cache.db" # O la ruta que use PandasAI v2
            if os.path.exists(cache_path):
                 os.remove(cache_path)
                 logger.info(f"Caché de PandasAI eliminada manualmente de: {cache_path}")
            elif os.path.exists(".pandasai-cache"): # Directorio pero no archivo db
                import shutil
                shutil.rmtree(".pandasai-cache")
                logger.info("Directorio .pandasai-cache eliminado.")
            else:
                 logger.info("No se encontró caché de PandasAI para eliminar.")

        except Exception as e:
            logger.error(f"No se pudo limpiar la caché de PandasAI: {e}")
    else:
        logger.info("La caché de PandasAI no está habilitada, no se requiere limpieza.")

# Podrías llamar a clear_pandasai_cache_if_enabled() al inicio de la app si quieres
# que siempre empiece con caché limpia durante el desarrollo, o bajo demanda.