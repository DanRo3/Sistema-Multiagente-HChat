import json
import re
import logging
from typing import Dict, Optional, Any
from app.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# --- Prompt Mejorado para Enrutamiento ---

SYSTEM_INSTRUCTIONS = """**Tarea:** Eres un asistente experto en traducir consultas de usuarios sobre registros marítimos históricos a una `pandasai_query` clara y una `intent`. La `pandasai_query` será ejecutada por PandasAI, el cual tiene acceso a habilidades (`Skills`) personalizadas.

**Contexto de Datos:** El DataFrame (`df`) contiene: `publication_date` (datetime), `travel_departure_date` (datetime), `travel_duration` (texto, usar `travel_duration_days` para numérico), `travel_arrival_date` (datetime), `travel_departure_port`, `travel_port_of_call_list`, `travel_arrival_port`, `ship_type` (abreviaturas), `ship_name`, `cargo_list` (texto libre), `master_role`, `master_name`, `parsed_text`, `travel_duration_days` (numérico).

**Habilidades (Skills) Disponibles para PandasAI:**
1.  **`get_tabular_data(df, columns_to_select, filter_conditions, sort_by, limit, query_description)`**:
    *   Usa esta habilidad para recuperar, filtrar, seleccionar columnas, ordenar y limitar datos.
    *   `columns_to_select`: Lista de nombres de columnas (ej. `['ship_name', 'travel_arrival_port']`). Omitir para todas.
    *   `filter_conditions`: String con condiciones de Pandas `df.query()` (ej. `"ship_type == 'berg. am.' and publication_date.dt.year == 1851"`). **SIEMPRE usa `df.` para referenciar columnas dentro del string de `filter_conditions` si es una query de Pandas, ej. `df['columna']`. PERO para la habilidad, solo pasas el string de condición.**
    *   `sort_by`: Lista de dicts, ej. `[{{'column': 'publication_date', 'order': 'desc'}}]`. Omitir si no se ordena.
    *   `limit`: Número entero para limitar filas. Omitir si no hay límite.
    *   `query_description`: Breve descripción en texto de la operación.
2.  **`plot_top_n_frequencies(df, column_name, top_n, chart_title, normalize_ship_types, query_description)`**:
    *   Usa esta habilidad para generar gráficos de barras de frecuencia de los N elementos más comunes.
    *   `column_name`: La columna para la cual calcular frecuencias (ej. `'ship_type'`, `'travel_departure_port'`).
    *   `top_n`: Número de elementos a mostrar (ej. `15`).
    *   `chart_title`: Título para el gráfico (ej. `'Top 15 Puertos de Salida'`).
    *   `normalize_ship_types`: Booleano. Poner a `True` **solo** si `column_name` es `'ship_type'` y se desea ver nombres completos en el gráfico.
    *   `query_description`: Breve descripción en texto de la operación.

**Instrucciones Detalladas para Formular `pandasai_query`:**
1.  **Determina la Intención (`intent`):** 'visual' si se pide un gráfico, 'text' en los demás casos.
2.  **Si es `intent: 'text'` (obtener datos):**
    *   **SIEMPRE** usa la habilidad `get_tabular_data`.
    *   Analiza la consulta del usuario para extraer `columns_to_select`, `filter_conditions`, `sort_by` y `limit`.
    *   **Ejemplo:** Usuario "Lista los nombres de los barcos que llegaron a La Habana en 1851, ordenados por fecha de publicación."
        *   La `pandasai_query` que debes generar para PandasAI es un string como este: "Usa la habilidad `get_tabular_data` con el DataFrame `df`, `columns_to_select=['ship_name', 'publication_date']`, `filter_conditions=\"df['travel_arrival_port'] == 'La Habana' and df['publication_date'].dt.year == 1851\"`, `sort_by=[{{'column': 'publication_date', 'order': 'asc'}}]`, y `query_description='Nombres de barcos llegados a La Habana en 1851 ordenados.'`"
    *   **Ejemplo:** Usuario "Dame todos los datos del barco 'Perla'"
        *   La `pandasai_query` que debes generar para PandasAI es un string como este: "Usa la habilidad `get_tabular_data` con el DataFrame `df`, `filter_conditions=\"df['ship_name'] == 'Perla'\"`, y `query_description='Todos los datos del barco Perla.'`"
3.  **Si es `intent: 'visual'` (generar gráfico):**
    *   **SIEMPRE** usa la habilidad `plot_top_n_frequencies` si la consulta implica mostrar frecuencias de una columna categórica.
    *   Analiza la consulta para extraer `column_name`, `top_n` (usa 15 por defecto si no se especifica), `chart_title`.
    *   Establece `normalize_ship_types=True` si `column_name` es `'ship_type'`, de lo contrario `False`.
    *   **Ejemplo:** Usuario "Gráfico de los 10 tipos de barco más comunes."
        *   La `pandasai_query` que debes generar para PandasAI es un string como este: "Usa la habilidad `plot_top_n_frequencies` con el DataFrame `df`, `column_name='ship_type'`, `top_n=10`, `chart_title='Top 10 Tipos de Barco por Frecuencia'`, `normalize_ship_types=True`, y `query_description='Gráfico de los 10 tipos de barco más comunes.'`"
4.  **Si la consulta es muy general o no encaja en las habilidades (Fallback):**
    *   Puedes generar una `pandasai_query` directa para PandasAI (ej. "Calcula el promedio de df['travel_duration_days'] y devuelve solo el número."). Esto debería ser menos común.

**Formato de Salida Requerido:** Responde **ÚNICAMENTE** con un objeto JSON válido con las claves: "intent" (string: 'text' o 'visual') y "pandasai_query" (string).

**Ejemplo Salida 1 (Pide Datos con Skill) - ESTO ES LO QUE DEBES GENERAR:**
```json
{{
  "intent": "text",
  "pandasai_query": "Usa la habilidad `get_tabular_data` con el DataFrame `df`, `columns_to_select=['ship_name', 'publication_date']`, `filter_conditions=df['master_name'] == 'Smith' and df['travel_departure_port'] == 'Nueva York'`, `sort_by=[{{'column': 'publication_date', 'order': 'asc'}}]`, `query_description='Barcos del Cap. Smith desde Nueva York.'`"
}}

**Ejemplo Salida 2 (Pide Gráfico con Skill) - ESTO ES LO QUE DEBES GENERAR:**
```json
{{
  "intent": "visual",
  "pandasai_query": "Usa la habilidad `plot_top_n_frequencies` con el DataFrame `df`, `column_name='travel_departure_port'`, `top_n=5`, `chart_title='Top 5 Puertos de Salida Más Comunes'`, `normalize_ship_types=False`, `query_description='Gráfico de los 5 puertos de salida más comunes.'`"
}}

```

**Ejemplo Salida 3 (Pide lista completa con Skill) - ESTO ES LO QUE DEBES GENERAR:**
```json
{{
  "intent": "text",
  "pandasai_query": "Usa la habilidad `get_tabular_data` con el DataFrame `df`, `columns_to_select=['ship_type']`, `filter_conditions=\"df['travel_arrival_port'] == 'La Habana'\"`, `query_description='Lista de ship_type de barcos que entraron a La Habana.'`"
}}

```

**Ejemplo Salida 4 (Busca en Texto y pide columnas específicas - Fallback sin Skill) - ESTO ES LO QUE DEBES GENERAR:**
```json
{{
  "intent": "text",
  "pandasai_query": "Devuelve un DataFrame con las columnas 'ship_name' y 'parsed_text' de df donde la columna df['parsed_text'] contenga la palabra 'tormenta'."
}}

```
"""

HUMAN_TASK_PREFIX = """**Consulta del Usuario:**
"{query}"

**Tu Análisis (Solo JSON):**
"""

# Crear plantilla (usando from_messages para robustez)
prompt_template = None
try:
    base_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", HUMAN_TASK_PREFIX)
    ])
    base_template.input_variables = ["query"]
    prompt_template = base_template
    logger.info("Plantilla de prompt del Moderador creada.")
except Exception as e:
    logger.exception(f"Error CRÍTICO al crear ChatPromptTemplate para Moderador: {e}.")

def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analiza la consulta, determina la intención final, y genera
    la consulta optimizada para PandasAI.
    """
    logger.info(f"Moderador: Iniciando análisis para query: '{query}'")

    if prompt_template is None:
         logger.error("Plantilla de prompt del Moderador inválida. Usando fallback.")
         return {"intent": "text", "pandasai_query": query}

    llm = get_llm()
    if not llm:
        logger.error("Error Crítico: LLM no disponible para el agente moderador.")
        return {"intent": "text", "pandasai_query": query}

    chain = prompt_template | llm

    try:
        logger.debug("Invocando cadena del moderador...")
        response = chain.invoke({"query": query})
        content = response.content
        logger.info(f"Moderador: Respuesta cruda del LLM (primeros 500 chars):\n---\n{content[:500]}\n---")

        json_str = extract_json(content)
        if json_str is None:
            logger.error("No se pudo extraer JSON de la respuesta del LLM moderador.")
            return {"intent": "text", "pandasai_query": query}

        logger.debug(f"Moderador: String JSON a parsear:\n---\n{repr(json_str)}\n---")
        parsed_response = json.loads(json_str)
        validated_response = validate_parsed_pandasai_response(parsed_response, query)
        logger.info(f"Moderador: Análisis finalizado: {validated_response}")
        return validated_response

    except json.JSONDecodeError as e:
        logger.error(f"Error Crítico: Fallo al parsear JSON: {e}")
        logger.error(f"Respuesta LLM que causó el error:\n{content}")
        return {"intent": "text", "pandasai_query": query}
    except ValueError as e:
        logger.error(f"Error durante la validación de la respuesta parseada: {e}")
        return {"intent": "text", "pandasai_query": query}
    except Exception as e:
        logger.exception(f"Error Inesperado en el agente moderador: {e}")
        return {"intent": "text", "pandasai_query": query}

def extract_json(content: str) -> Optional[str]:
    """Extrae el primer bloque JSON ```json ... ``` o el primer objeto JSON { ... }."""
    json_block_match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", content, re.DOTALL)
    if json_block_match:
        extracted = json_block_match.group(1).strip()
        logger.debug(f"JSON extraído (```json): {extracted[:100]}...")
        return extracted
    stripped_content = content.strip()
    # Buscar un JSON que ocupe toda la línea o esté indentado
    json_object_match = re.search(r"^\s*(\{[\s\S]*?\})\s*$", stripped_content, re.MULTILINE)
    if json_object_match:
        logger.debug("Respuesta parece ser solo un objeto JSON.")
        return json_object_match.group(1).strip()
    logger.warning("No se encontró un bloque JSON reconocible en la respuesta.")
    return None

def validate_parsed_pandasai_response(parsed_response: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    """Valida la respuesta JSON parseada del LLM moderador (versión PandasAI)."""
    required_keys = ["intent", "pandasai_query"]
    if not all(k in parsed_response for k in required_keys):
        missing_keys = [k for k in required_keys if k not in parsed_response]
        logger.error(f"Respuesta JSON incompleta. Faltan: {missing_keys}. Recibido: {parsed_response.keys()}")
        raise ValueError("Respuesta JSON del moderador incompleta.")

    valid_intents = ["text", "visual", "code"] # Mantener 'code' por si acaso
    intent = parsed_response.get("intent")
    if intent not in valid_intents:
        logger.warning(f"Intención inválida '{intent}'. Usando 'text'.")
        parsed_response["intent"] = "text"

    pandasai_query = parsed_response.get("pandasai_query")
    if not isinstance(pandasai_query, str) or not pandasai_query.strip():
        logger.warning(f"'pandasai_query' inválido o vacío. Usando original.")
        parsed_response["pandasai_query"] = original_query

    # Añadir claves faltantes con None para consistencia del estado
    parsed_response.setdefault("filters", None) # Ya no lo generamos pero lo mantenemos None
    parsed_response.setdefault("search_query", None) # Ya no lo generamos

    return parsed_response