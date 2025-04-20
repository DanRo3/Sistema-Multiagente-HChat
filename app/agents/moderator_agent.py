import json
import re
import logging # Usar logging
from typing import Dict, Optional, Any
from app.core.llm import get_llm # Importar getter genérico
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

MODERATOR_PROMPT = """**Tarea:** Analiza la consulta del usuario y extrae información estructurada para guiar la búsqueda y respuesta en un sistema de consulta de registros marítimos históricos.

**Consulta del Usuario:**
"{query}"

**Contexto de Datos:** La base de datos contiene registros sobre viajes marítimos con las siguientes columnas de metadatos relevantes para filtrar: `publication_date`, `news_section`, `travel_departure_date`, `travel_duration`, `travel_arrival_date`, `travel_departure_port`, `travel_port_of_call_list`, `travel_arrival_port`, `ship_type`, `ship_name`, `cargo_list`, `master_role`, `master_name`. La columna `parsed_text` contiene el texto completo para búsqueda vectorial.

**Instrucciones Detalladas:**
1.  **Determina la Intención Principal (`intent`):** Clasifica como 'visual' si pide gráficos/tablas/visualizaciones, 'code' si pide código Python explícitamente, o 'text' en los demás casos.
2.  **Identifica Filtros de Metadatos (`filters`):** Busca valores específicos en la consulta que coincidan **exactamente** con las columnas de metadatos listadas (excepto `parsed_text`). Devuelve un diccionario Python con la estructura `{{columna: valor}}` donde `columna` es el nombre exacto del metadato y `valor` es el extraído. Si no se identifican filtros claros y específicos, devuelve `null` o un diccionario vacío `{{}}`. NO inventes filtros. Sé preciso (e.g., para 'capitán Smith', el filtro incluiría `master_name: "Smith"`).
3.  **Crea una Consulta de Búsqueda Vectorial (`search_query`):** Formula una consulta concisa con palabras clave y entidades (nombres, lugares, barcos, acciones, fechas relevantes) de la consulta original, optimizada para la búsqueda vectorial en `parsed_text`. Elimina palabras innecesarias.

**Formato de Salida Requerido:** Responde **estrictamente y únicamente** con un objeto JSON válido con las claves "intent" (string: 'text', 'visual' o 'code'), "filters" (dict o null), y "search_query" (string). No incluyas texto fuera del JSON.

**Ejemplo de Salida JSON:**
```json
{{  #
  "intent": "text",
  "filters": {{  # 
    "master_name": "Litlejohn"
  }}, # 
  "search_query": "fragata Charles Edwin capitán Litlejohn"
}}  # 


Tu Análisis (Solo JSON):
"""



try:
    prompt_template = ChatPromptTemplate.from_template(MODERATOR_PROMPT)
except Exception as e:
    logger.exception(f"Error CRÍTICO al crear ChatPromptTemplate para Moderador: {e}. El prompt puede tener errores de formato.")
    # No podemos continuar si la plantilla es inválida
    prompt_template = None # Indicar que falló

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analiza la consulta del usuario usando el LLM configurado para determinar
    la intención, extraer filtros y generar una consulta de búsqueda optimizada.
    """
    logger.info(f"Moderador: Iniciando análisis para query: '{query}'")

    # Verificar si la plantilla se creó correctamente
    if prompt_template is None:
        logger.error("La plantilla de prompt del Moderador es inválida. Usando fallback.")
        return {"intent": "text", "filters": None, "search_query": query}

    # Obtener la instancia del LLM configurado
    llm = get_llm()
    if not llm:
        logger.error("Error Crítico: LLM no disponible para el agente moderador.")
        return {"intent": "text", "filters": None, "search_query": query}

    # Crear la cadena (chain) LangChain
    chain = prompt_template | llm

    try:
        # Invocar la cadena con la consulta del usuario
        logger.debug("Invocando cadena del moderador...")
        response = chain.invoke({"query": query})
        content = response.content

        logger.info(f"Moderador: Respuesta cruda del LLM recibida (primeros 500 chars):\n---\n{content[:500]}\n---")

        # --- Extracción robusta del JSON ---
        json_str = extract_json(content)
        if json_str is None:
            logger.error("No se pudo extraer JSON de la respuesta del LLM moderador.")
            # Podríamos intentar un análisis más simple o directamente fallback
            return {"intent": "text", "filters": None, "search_query": query}

        logger.debug(f"Moderador: String EXACTA a parsear con json.loads:\n---\n{repr(json_str)}\n---")

        # Parsear el string JSON extraído
        parsed_response = json.loads(json_str)

        # --- Validación y Limpieza de la Respuesta Parseada ---
        validated_response = validate_parsed_response(parsed_response, query)
        logger.info(f"Moderador: Análisis finalizado y validado: {validated_response}")
        return validated_response

    except json.JSONDecodeError as e:
        logger.error(f"Error Crítico: Fallo al parsear JSON de la respuesta del LLM moderador: {e}")
        logger.error(f"Respuesta LLM que causó el error:\n{content}")
        return {"intent": "text", "filters": None, "search_query": query}
    except ValueError as e: # Capturar errores de validación nuestra
        logger.error(f"Error durante la validación de la respuesta parseada: {e}")
        return {"intent": "text", "filters": None, "search_query": query}
    except Exception as e:
        # Captura cualquier otro error inesperado (incluyendo el KeyError del prompt si aún ocurre)
        logger.exception(f"Error Inesperado en el agente moderador: {e}") # Usar logger.exception para incluir traceback
        return {"intent": "text", "filters": None, "search_query": query}


def extract_json(content: str) -> Optional[str]:
    """Extrae el primer bloque JSON ```json ... ``` o el primer objeto JSON { ... }."""
    # PRIORIDAD 1: Buscar bloque delimitado con ```json
    # Ajustado para buscar ```json, espacios opcionales, el JSON, espacios opcionales, y ```
    json_block_match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", content, re.DOTALL)
    if json_block_match:
        extracted = json_block_match.group(1).strip()
        logger.debug(f"Moderador: JSON extraído usando patrón ```json: {extracted[:100]}...") # Log para ver qué extrajo
        return extracted

    # PRIORIDAD 2: Buscar si la respuesta es SOLO un objeto JSON (con/sin espacios alrededor)
    # Usa strip() ANTES de aplicar el regex para limpiarlo
    stripped_content = content.strip()
    if stripped_content.startswith('{') and stripped_content.endswith('}'):
        logger.debug("Moderador: Respuesta parece ser solo un objeto JSON, intentando parsear directamente.")
        # Podríamos intentar parsear aquí mismo para validar, o simplemente devolverlo
        # Devolverlo directamente es más simple, json.loads lo validará después
        return stripped_content # Devolvemos el contenido limpio

    logger.warning("No se encontró un bloque JSON reconocible (ni ```json...``` ni objeto JSON directo) en la respuesta.")
    return None


def validate_parsed_response(parsed_response: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    """Valida y limpia la respuesta JSON parseada del LLM moderador."""
    required_keys = ["intent", "filters", "search_query"]
    if not all(k in parsed_response for k in required_keys):
        logger.error(f"Respuesta JSON del LLM incompleta. Faltan claves. Recibido: {parsed_response.keys()}")
        raise ValueError("Respuesta JSON del LLM no tiene la estructura esperada (faltan claves).")

    # Validar valor de 'intent'
    valid_intents = ["text", "visual", "code"]
    intent = parsed_response.get("intent")
    if intent not in valid_intents:
        logger.warning(f"Intención inválida '{intent}' recibida del LLM. Usando 'text' por defecto.")
        parsed_response["intent"] = "text"

    # Validar y limpiar 'filters'
    filters_value = parsed_response.get("filters")
    if filters_value is not None:
        if not isinstance(filters_value, dict):
            logger.warning(f"'filters' no es un diccionario (tipo: {type(filters_value)}). Se establecerá a None.")
            parsed_response["filters"] = None
        elif not filters_value: # Si es un diccionario vacío {}
            logger.debug("'filters' es un diccionario vacío, estableciendo a None.")
            parsed_response["filters"] = None
        else:
            # Opcional: Validar claves de filtro contra columnas conocidas
            known_columns = { 'publication_date', 'news_section', 'travel_departure_date',
                            'travel_duration', 'travel_arrival_date', 'travel_departure_port',
                            'travel_port_of_call_list', 'travel_arrival_port', 'ship_type',
                            'ship_name', 'cargo_list', 'master_role', 'master_name'}
            invalid_keys = [k for k in filters_value if k not in known_columns]
            if invalid_keys:
                logger.warning(f"Filtros contienen claves desconocidas: {invalid_keys}. Se mantendrán, pero podrían no funcionar.")
                # Podrías decidir eliminarlos:
                # parsed_response["filters"] = {k: v for k, v in filters_value.items() if k in known_columns}

    # Asegurar que search_query sea string y no esté vacío
    search_query = parsed_response.get("search_query")
    if not isinstance(search_query, str) or not search_query.strip():
        logger.warning(f"'search_query' inválido o vacío ('{search_query}'). Usando la query original como fallback.")
        parsed_response["search_query"] = original_query

    return parsed_response


