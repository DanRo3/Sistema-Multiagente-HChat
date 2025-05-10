# app/agents/contextualizer_agent.py
import json
import logging
from typing import Dict, Any, Optional, List # Asegurar que List esté aquí

logger = logging.getLogger(__name__)

# Ya NO necesitamos LLM ni plantillas de prompt en este agente
# para el flujo simplificado de PandasAI.

def format_pandasai_data_for_summary(
    pandasai_result: Optional[Any],
    pandasai_result_type: Optional[str],
    original_query: str # Podríamos usarla para un encabezado
) -> Optional[str]:
    """
    Formatea el resultado de PandasAI (si no es un plot o error)
    en un string legible para el usuario final, sin usar un LLM.
    """
    if pandasai_result is None and pandasai_result_type is None:
        return "No se obtuvo un resultado específico del análisis de datos."

    if pandasai_result_type == "dataframe_list":
        if not isinstance(pandasai_result, list):
            logger.warning(f"Contextualizador: Se esperaba lista para dataframe_list, se obtuvo {type(pandasai_result)}")
            return "Se recibieron datos tabulares, pero en un formato inesperado."

        num_rows = len(pandasai_result)
        if num_rows == 0:
            return "No se encontraron registros que coincidan con tu consulta."

        # Extraer nombres de columna de la primera fila (si existe)
        # columns = list(pandasai_result[0].keys()) if num_rows > 0 and pandasai_result[0] else []

        # Caso especial: si la consulta pedía explícitamente una lista de algo (ej. nombres de barcos)
        # y el resultado es una lista de diccionarios con una sola clave.
        is_simple_list_output = False
        single_key = None
        if num_rows > 0 and isinstance(pandasai_result[0], dict) and len(pandasai_result[0]) == 1:
            single_key = list(pandasai_result[0].keys())[0]
            # Verificar si todos los dicts tienen solo esa clave
            if all(isinstance(item, dict) and len(item) == 1 and single_key in item for item in pandasai_result):
                is_simple_list_output = True

        if is_simple_list_output and single_key:
            items_list = [str(item.get(single_key, "N/A")) for item in pandasai_result]
            items_str = ", ".join(items_list[:20]) # Mostrar hasta 20 elementos directamente
            if num_rows == 1:
                return f"El resultado es: {items_str}."
            elif num_rows <= 20:
                return f"Los resultados son: {items_str}."
            else:
                return f"Se encontraron {num_rows} resultados. Los primeros son: {items_str} (...y {num_rows - 20} más)."
        else:
            # Para DataFrames más generales, solo indicar la cantidad y quizás las columnas
            # o un preview muy corto. El frontend podría renderizar la tabla completa si es necesario.
            # columns_str = ", ".join(columns)
            preview_items = pandasai_result[:3] # Muestra las primeras 3 filas
            try:
                preview_str = json.dumps(preview_items, indent=2, ensure_ascii=False, default=str)
            except Exception:
                 preview_str = str(preview_items)

            return f"Se encontraron {num_rows} registros. A continuación una muestra:\n{preview_str}" \
                   f"{f'\n(... y {num_rows - 3} filas más)' if num_rows > 3 else ''}"

    elif pandasai_result_type == "string":
        return f"{str(pandasai_result)}"
    elif pandasai_result_type in ["int", "float", "number", "bool"]: # 'number' podría ser un tipo de PandasAI
        return f"El resultado es: {str(pandasai_result)}."
    elif isinstance(pandasai_result, list): # Lista genérica, no de dataframes
        try:
            # Limitar tamaño de listas grandes
            res_str = json.dumps(pandasai_result[:20], ensure_ascii=False, default=str)
            if len(pandasai_result) > 20 : res_str += "\n(... más elementos no mostrados)"
            return f"Resultados: {res_str}"
        except Exception:
             return f"Resultados: {str(pandasai_result)[:1000]}"
    elif isinstance(pandasai_result, dict): # Diccionario genérico
        try:
            res_str = json.dumps(pandasai_result, indent=2, ensure_ascii=False, default=str)
            if len(res_str) > 1000: res_str = res_str[:1000] + "..."
            return f"Resultado: {res_str}"
        except Exception:
            return f"Resultado: {str(pandasai_result)[:1000]}"
    else:
        logger.warning(f"Contextualizador: Tipo de resultado PandasAI no manejado explícitamente para summary: {pandasai_result_type}")
        return f"Se obtuvo un resultado del análisis: {str(pandasai_result)[:500]}"


def contextualize(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepara el 'summary' para el validador y el usuario final, basándose
    directamente en los resultados de PandasAI (sin usar LLM aquí).
    Pasa la ruta del plot si existe.
    """
    original_query = state.get('original_query', "Consulta no especificada") # Fallback
    pandasai_result = state.get('pandasai_result')
    pandasai_result_type = state.get('pandasai_result_type')
    pandasai_plot_path = state.get('pandasai_plot_path')
    pandasai_error = state.get('pandasai_error')

    logger.info("Contextualizador (PandasAI-only - Flujo Simplificado): Iniciando...")
    logger.debug(f"  Recibido del estado: ResultType='{pandasai_result_type}', PlotPath='{pandasai_plot_path}', Error='{pandasai_error}'")

    output_summary: Optional[str] = None

    # Si hay un plot generado por PandasAI o un error, el validador los manejará.
    # Aquí solo generamos un summary textual si NO hay plot Y NO hay error.
    if not pandasai_plot_path and not pandasai_error:
        logger.info("Contextualizador: No hay plot ni error de PandasAI, formateando resultado para summary.")
        output_summary = format_pandasai_data_for_summary(
            pandasai_result,
            pandasai_result_type,
            original_query
        )
        logger.info(f"Contextualizador: Summary formateado (sin LLM): '{str(output_summary)[:150]}...'")
    elif pandasai_plot_path:
        logger.info("Contextualizador: Se detectó ruta de plot. El validador la manejará.")
        # Podemos poner un texto genérico que el validador usará si la imagen se procesa bien
        output_summary = "Se ha generado una visualización para tu consulta."
    elif pandasai_error:
        logger.info("Contextualizador: Se detectó error de PandasAI. El validador lo manejará.")
        # El summary puede reflejar el error para que el validador lo tenga
        output_summary = f"Error al procesar la consulta con PandasAI: {pandasai_error}"
    else:
        # Caso de fallback si no hay plot, ni error, ni resultado claro
        logger.warning("Contextualizador: No hay plot, ni error, ni resultado claro de PandasAI para formatear.")
        output_summary = "No se pudo obtener una respuesta clara del análisis de datos."


    # El estado devuelto solo necesita el summary.
    # El Validador leerá pandasai_plot_path y pandasai_error directamente del estado global.
    return {"summary": output_summary}