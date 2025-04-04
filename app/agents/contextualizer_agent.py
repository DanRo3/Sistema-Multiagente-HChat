import json
import re
from typing import Dict, List, Optional, Any
from app.core.llm import get_llm # Importa la función para obtener el LLM
from langchain_core.documents import Document # Para type hinting
from langchain_core.prompts import ChatPromptTemplate # Para las plantillas de prompt

# --- Plantilla para Resumen Textual ---
TEXT_SUMMARY_PROMPT = """
    **Tarea:** Basándote **únicamente** en los siguientes documentos recuperados de registros marítimos, genera una respuesta concisa y directa a la consulta original del usuario.

    **Consulta Original del Usuario:**
    "{query}"

    **Documentos Recuperados (Contenido y Metadatos Clave):**


    {documents_summary}

    **Instrucciones:**
    - Sintetiza la información clave de los documentos que responde **directamente** a la consulta.
    - Si varios documentos contienen información relevante, intenta combinarlos coherentemente.
    - Si los documentos recuperados **no contienen** información para responder la consulta, indica claramente que no se encontró información específica sobre ese tema en los registros consultados. Ejemplo: "Lo siento, no encontré información sobre [tema específico] en los documentos recuperados."
    - **No inventes información** que no esté explícitamente en los documentos proporcionados.
    - Sé breve y ve al punto.
    - Responde directamente con el resumen o la indicación de que no se encontró información. No añadas frases introductorias como "Basado en los documentos..." a menos que sea necesario para la claridad.

    **Resumen Conciso:**
"""

# --- Plantilla para Análisis de Visualización ---
VISUAL_ANALYSIS_PROMPT = """
    **Tarea:** Evalúa si los documentos recuperados contienen datos suficientes y apropiados para generar la visualización solicitada por el usuario. Si es viable, extrae y formatea los datos relevantes.

    **Consulta Original del Usuario (solicitando visualización):**
    "{query}"

    **Documentos Recuperados (Contenido y Metadatos Clave):**

    {documents_summary}

    **Instrucciones Detalladas:**
    1.  **Evalúa Viabilidad (`visualization_possible`):**
        *   Analiza la **consulta** del usuario: ¿Qué tipo de gráfico pide (barras, líneas, etc.)? ¿Qué variables necesita (puertos, duraciones, tipos de barco, fechas, cantidades)?
        *   Analiza los **documentos**: ¿Contienen las variables necesarias mencionadas en la consulta? ¿Hay suficientes datos (al menos 3-5 puntos de datos relevantes suelen ser necesarios para un gráfico simple)? ¿Los datos son consistentes y del tipo correcto (números para agregaciones, categorías para ejes)?
        *   Decide si la visualización es razonablemente posible (`true`) o no (`false`).
    2.  **Extrae Datos si es Viable (`data_for_python`):**
        *   Si `visualization_possible` es `true`, extrae los **datos exactos** necesarios de los documentos recuperados.
        *   Formatea estos datos como una **lista de diccionarios Python**. Cada diccionario representa una fila o punto de datos, y las claves deben ser nombres descriptivos y consistentes para las columnas que el agente Python usará (e.g., 'puerto_salida', 'frecuencia', 'tipo_barco', 'duracion_dias').
        *   Asegúrate de que los valores numéricos sean números (int/float) si es posible, y los textos sean strings.
        *   Devuelve esta lista de diccionarios **como un string** (usando `json.dumps` o similar internamente antes de ponerlo en el JSON de respuesta).
        *   Si `visualization_possible` es `false`, el valor de `data_for_python` debe ser `null`.
    3.  **Genera una Explicación (`explanation`):**
        *   Si `visualization_possible` es `true`, escribe una breve frase indicando que los datos están listos para generar la gráfica solicitada. Ejemplo: "Datos extraídos para generar gráfico de barras de frecuencia de puertos de salida."
        *   Si `visualization_possible` es `false`, explica **por qué** no es posible (datos insuficientes, variables faltantes, tipo de gráfico no adecuado para los datos, etc.). Ejemplo: "No se encontraron suficientes registros con información sobre la duración del viaje para el tipo de barco especificado para crear un gráfico de tendencias."

    **Formato de Salida Requerido:** Responde **estrictamente y únicamente** con un objeto JSON válido que contenga las claves: "visualization_possible" (boolean), "data_for_python" (string que representa una lista de dicts, o null), y "explanation" (string).

    Tu Análisis (Solo JSON):
"""


text_summary_template = ChatPromptTemplate.from_template(TEXT_SUMMARY_PROMPT)
visual_analysis_template = ChatPromptTemplate.from_template(VISUAL_ANALYSIS_PROMPT)

def format_docs_for_llm(docs: List[Document], max_docs: int = 20, max_len_per_doc: int = 500) -> str:
    """
    Formatea una lista de documentos de forma concisa para incluirla en el prompt del LLM.
    Extrae metadatos clave y limita la longitud del texto.
    """
    if not docs:
        return "No se recuperaron documentos relevantes."

    summary_parts = []
    # Priorizar metadatos potencialmente útiles
    relevant_metadata_keys = [
        'publication_date', 'travel_departure_date', 'travel_duration',
        'travel_arrival_date', 'travel_departure_port', 'travel_arrival_port',
        'ship_type', 'ship_name', 'cargo_list', 'master_name'
    ]

    for i, doc in enumerate(docs[:max_docs]):
        # Extraer metadatos relevantes
        meta_dict = {k: doc.metadata.get(k) for k in relevant_metadata_keys if doc.metadata.get(k)}
        meta_str = json.dumps(meta_dict, ensure_ascii=False) # JSON compacto

        # Limitar longitud del texto principal
        content_preview = doc.page_content[:max_len_per_doc]
        if len(doc.page_content) > max_len_per_doc:
            content_preview += "..."
            summary_parts.append(f"Doc {i+1} Metadata: {meta_str}\nDoc {i+1} Texto: {content_preview}\n---")

        if len(docs) > max_docs:
            summary_parts.append(f"(... y {len(docs) - max_docs} documentos más recuperados pero no mostrados para brevedad)")

        return "\n".join(summary_parts)


def contextualize(original_query: str, intent: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    """
    Contextualiza la respuesta basada en la intención y los documentos recuperados.

    Args:
        original_query: La consulta original del usuario.
        intent: La intención detectada ('text', 'visual', 'code').
        retrieved_docs: Lista de documentos recuperados por el agente anterior.

    Returns:
        Un diccionario con las claves:
        - 'summary' (str): El resumen textual o la explicación.
        - 'needs_visualization' (bool): True si se debe proceder a generar código.
        - 'data_for_python' (Optional[str]): String representando lista de dicts
                                            para el agente Python, o None.
    """
    print(f"Contextualizador: Iniciando. Intención='{intent}'. Documentos Recibidos={len(retrieved_docs)}")

    # Si no hay documentos, no se puede hacer nada más
    if not retrieved_docs:
        print("Contextualizador: No hay documentos para contextualizar.")
        return {"summary": "Lo siento, no encontré información relevante para tu consulta en los registros.",
                "needs_visualization": False,
                "data_for_python": None}

    # Obtener el LLM (usar temperatura baja para tareas estructuradas/resúmenes)
    llm = get_llm()
    if not llm:
        print("Error Crítico: LLM no disponible para el agente contextualizador.")
        return {"summary": "Error interno: No se pudo procesar la solicitud en este momento.",
                "needs_visualization": False,
                "data_for_python": None}

    # Formatear los documentos para pasarlos al LLM de forma eficiente
    docs_summary_for_prompt = format_docs_for_llm(retrieved_docs)

    # --- Lógica basada en la Intención ---

    if intent == 'text' or intent == 'code': # Tratar 'code' como 'text' por ahora, la validación se encargará
        print("Contextualizador: Intención es 'text' o 'code'. Generando resumen textual...")
        chain = text_summary_template | llm
        try:
            response = chain.invoke({
                "query": original_query,
                "documents_summary": docs_summary_for_prompt
            })
            summary = response.content.strip()
            print(f"Contextualizador: Resumen textual generado: '{summary[:150]}...'")
            # Para intención 'text' o 'code', no necesitamos visualización
            return {"summary": summary, "needs_visualization": False, "data_for_python": None}
        except Exception as e:
            print(f"Error inesperado generando resumen textual: {e}")
            return {"summary": "Error al intentar generar un resumen de la información encontrada.",
                    "needs_visualization": False,
                    "data_for_python": None}

    elif intent == 'visual':
        print("Contextualizador: Intención es 'visual'. Analizando viabilidad de visualización...")
        chain = visual_analysis_template | llm
        try:
            response = chain.invoke({
                "query": original_query,
                "documents_summary": docs_summary_for_prompt
            })
            content = response.content

            print(f"Contextualizador: Respuesta cruda del LLM (Análisis Visual):\n---\n{content}\n---")

            # Extraer JSON (similar al moderador)
            json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match_loose = re.search(r"(\{[\s\S]*?\})", content, re.DOTALL)
                json_str = json_match_loose.group(1).strip() if json_match_loose else content.strip()

            print(f"Contextualizador: JSON extraído (Análisis Visual):\n---\n{json_str}\n---")

            analysis_result = json.loads(json_str)

            # Validar respuesta del análisis
            if not all(k in analysis_result for k in ["visualization_possible", "data_for_python", "explanation"]):
                raise ValueError("Respuesta JSON del análisis de visualización incompleta.")

            is_possible = analysis_result.get("visualization_possible", False)
            data_output = analysis_result.get("data_for_python") # Puede ser string o null
            explanation = analysis_result.get("explanation", "Análisis de visualización incompleto.")

            if is_possible and data_output:
                print("Contextualizador: Visualización considerada POSIBLE.")
                # La explicación puede servir como texto acompañante inicial
                return {
                    "summary": explanation,
                    "needs_visualization": True,
                    "data_for_python": data_output # Pasa el string que representa la lista de dicts
                }
            else:
                print("Contextualizador: Visualización considerada NO POSIBLE.")
                # La explicación se convierte en la respuesta textual final
                return {
                    "summary": explanation,
                    "needs_visualization": False,
                    "data_for_python": None
                }

        except json.JSONDecodeError as e:
            print(f"Error Crítico: Fallo al parsear JSON del análisis de visualización: {e}")
            print(f"Respuesta LLM que causó el error:\n{content}")
            return {"summary": "Error al analizar la viabilidad de la visualización solicitada.",
                    "needs_visualization": False,
                    "data_for_python": None}
        except Exception as e:
            print(f"Error Inesperado durante el análisis de visualización: {e}")
            import traceback
            traceback.print_exc()
            return {"summary": "Error interno al procesar la solicitud de visualización.",
                    "needs_visualization": False,
                    "data_for_python": None}
    else:
        # Fallback si la intención no es reconocida (aunque el moderador debería validarla)
        print(f"Advertencia (Contextualizador): Intención desconocida '{intent}'. Procediendo como texto.")
        return contextualize(original_query, 'text', retrieved_docs)



# if __name__ == 'main':
#     print("--- Probando Agente Contextualizador ---")

#     # Simular documentos recuperados (ejemplo muy simple)
#     mock_docs = [
#         Document(page_content="Llegó fragata A de New York en 10 días. Capitan Smith.", metadata={"ship_type": "fragata", "travel_duration": "10dias", "master_name": "Smith"}),
#         Document(page_content="Vapor B llegó de La Habana en 5 días. Capitan Jones.", metadata={"ship_type": "vapor", "travel_duration": "5dias", "master_name": "Jones"}),
#         Document(page_content="Fragata C tardó 12 días desde Liverpool. Capitan Smith.", metadata={"ship_type": "fragata", "travel_duration": "12dias", "master_name": "Smith"})
#     ]

#     # Prueba 1: Intención Textual
#     # query1 = "¿Quién capitaneó la fragata C?"
#     # intent1 = "text"
#     # result1 = contextualize(query1, intent1, mock_docs)
#     # print("\nResultado Prueba 1 (Textual):", json.dumps(result1, indent=2, ensure_ascii=False))

#     # Prueba 2: Intención Visual (Posible)
#     # query2 = "Muéstrame la duración promedio por tipo de barco"
#     # intent2 = "visual"
#     # result2 = contextualize(query2, intent2, mock_docs)
#     # print("\nResultado Prueba 2 (Visual Posible):", json.dumps(result2, indent=2, ensure_ascii=False))

#     # Prueba 3: Intención Visual (No Posible)
#     query3 = "Grafica la carga de los barcos por capitán"
#     intent3 = "visual"
#     result3 = contextualize(query3, intent3, mock_docs)
#     print("\nResultado Prueba 3 (Visual No Posible):", json.dumps(result3, indent=2, ensure_ascii=False))
    