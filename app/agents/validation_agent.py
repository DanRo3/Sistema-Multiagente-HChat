# app/agents/validation_agent.py
import json
import logging
import base64
import os
import re # Si usas extract_json
from typing import Optional, Tuple, Dict, Any
from app.core.llm import get_llm # Todavía puede usarse para validar texto SI es necesario
from langchain_core.prompts import ChatPromptTemplate
# from .moderator_agent import extract_json # Asegúrate que esto esté definido/importado

logger = logging.getLogger(__name__)

# --- Plantilla Opcional para Validar Texto ---
# (Podemos decidir no usarla si confiamos en el summary del contextualizador)
# ... (SYSTEM_VALIDATION_INSTRUCTIONS y HUMAN_VALIDATION_TASK como antes) ...
# ... (Creación de prompt_template como antes) ...
# --- Función extract_json (si es necesaria) ---
# ... (código de extract_json) ...


def validate(
    original_query: str,
    summary_from_contextualizer: Optional[str], # Recibe el summary (puede ser None si hay plot/error)
    pandasai_error: Optional[str],          # Error directo de PandasAI
    plot_path_from_pandasai: Optional[str]  # Ruta del plot directo de PandasAI
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Valida la respuesta final. Prioriza errores, luego plots, luego texto.
    Ahora toma 4 argumentos posicionales.
    """
    logger.info("Validador (PandasAI-only): Iniciando validación final...")
    logger.debug(f"  Recibido: PlotPath={plot_path_from_pandasai}, Error={pandasai_error}, Summary={summary_from_contextualizer}")
    final_text: Optional[str] = None
    final_image: Optional[str] = None
    error_message: Optional[str] = None

    # 1. Manejar Error de PandasAI (prioritario)
    if pandasai_error:
        logger.error(f"Validador: Detectado error de PandasAI: {pandasai_error}")
        error_message = f"Lo siento, ocurrió un error al procesar tu consulta con el análisis de datos: {pandasai_error}"
        return None, None, error_message # Salir temprano

    # 2. Manejar Gráfico Generado por PandasAI
    if plot_path_from_pandasai:
        logger.info(f"Validador: Procesando ruta de gráfico: {plot_path_from_pandasai}")
        if os.path.exists(plot_path_from_pandasai):
            try:
                # ... (lógica de codificación Base64 como antes) ...
                with open(plot_path_from_pandasai, "rb") as img_file:
                    img_bytes = img_file.read()
                base64_encoded_string = base64.b64encode(img_bytes).decode('utf-8')
                final_image = f"data:image/png;base64,{base64_encoded_string}"
                final_text = summary_from_contextualizer or "Aquí tienes el gráfico solicitado:"
                logger.info("Validador: Imagen codificada exitosamente.")
                # ... (lógica de eliminación de archivo temporal como antes) ...
                try:
                     os.remove(plot_path_from_pandasai)
                     logger.info(f"Archivo de gráfico temporal eliminado: {plot_path_from_pandasai}")
                except OSError as e:
                     logger.warning(f"No se pudo eliminar el archivo de gráfico temporal {plot_path_from_pandasai}: {e}")
                return final_text, final_image, None
            except Exception as e:
                # ... (manejo de error de codificación como antes) ...
                error_message = "Error interno al procesar el gráfico generado."
                return None, None, error_message
        else:
            # ... (manejo de error si la ruta no existe como antes) ...
            error_message = "Se intentó generar un gráfico, pero ocurrió un problema al guardarlo o encontrarlo."
            return None, None, error_message

    # 3. Manejar Respuesta Textual (summary del contextualizador)
    if summary_from_contextualizer:
        logger.info("Validador: Procesando respuesta textual del contextualizador.")
        final_text = summary_from_contextualizer # Aceptar directamente por ahora
        error_message = None
        logger.info("Validador: Respuesta textual final aprobada.")
        return final_text, None, error_message

    # 4. Caso Fallback
    logger.error("Validador: No hubo error, ni plot, ni summary. Estado inesperado.")
    error_message = "Error inesperado: No se pudo determinar una respuesta final."
    return None, None, error_message