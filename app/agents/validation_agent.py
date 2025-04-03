import json
import re
from typing import Dict, Optional, Any, Tuple
from app.core.llm import get_gemini_llm # Importa la función para obtener el LLM
from langchain_core.prompts import ChatPromptTemplate # Para la plantilla de prompt

# --- Plantilla de Prompt para Validación ---
# Guía al LLM para evaluar la coherencia final.
VALIDATION_PROMPT = """
    **Tarea:** Evalúa si el resultado final proporcionado responde de manera coherente, relevante y útil a la consulta original del usuario. Considera si hubo errores previos.

    **Consulta Original del Usuario:**
    "{query}"

    **Resultado Final a Validar:**
    {result_description}


    {content_to_validate}

    {error_context}

    **Instrucciones:**
    1.  **Considera la Consulta Original:** ¿Qué buscaba o pedía específicamente el usuario?
    2.  **Considera el Resultado:**
        *   Si es texto: ¿El contenido responde directamente a la pregunta? ¿Es claro y preciso? ¿Se basa en la información que probablemente se encontró (contexto implícito)?
        *   Si es una imagen (descrita como generada): ¿Es plausible que una imagen generada sobre este tema responda a la consulta visual? (No puedes ver la imagen, evalúa conceptualmente).
        *   Si hubo un error previo: ¿El mensaje de error es informativo?
    3.  **Decide sobre la Validez:** ¿Es esta una respuesta final aceptable y relevante para entregar al usuario, considerando todo el contexto?
    4.  **Responde en Formato JSON:** Responde **estrictamente y únicamente** con un objeto JSON válido con las claves:
        *   `is_valid` (boolean: `true` si la respuesta (o el mensaje de error) es apropiada y relevante, `false` si el resultado es incoherente, irrelevante o inútil).
        *   `reasoning` (string: una breve explicación de tu decisión de validación).
        *   `alternative_suggestion` (string o null: Si `is_valid` es `false` y el contenido no es un error claro, sugiere brevemente por qué podría ser incorrecto o qué podría faltar. Si es válido o es un error claro, devuelve `null`).

    **Ejemplo de Salida JSON (Respuesta de Texto Válida):**

    ```json
    {{
    "is_valid": true,
    "reasoning": "El resumen textual proporciona directamente los detalles solicitados sobre el viaje del barco.",
    "alternative_suggestion": null
    }}


    Ejemplo de Salida JSON (Imagen Válida Conceptual):

    {{
    "is_valid": true,
    "reasoning": "Una imagen generada sobre la frecuencia de puertos parece una respuesta apropiada a la solicitud de visualización.",
    "alternative_suggestion": null
    }}


    Ejemplo de Salida JSON (Respuesta Inválida):

    {{
    "is_valid": false,
    "reasoning": "El texto proporcionado habla sobre tipos de carga, pero la consulta original era sobre la duración de los viajes.",
    "alternative_suggestion": "La respuesta parece no estar relacionada con la duración de los viajes solicitada."
    }}


    Ejemplo de Salida JSON (Error de Ejecución Válido como Mensaje):

    {{
    "is_valid": true,
    "reasoning": "Se informa correctamente al usuario sobre un error ocurrido durante la generación de la respuesta.",
    "alternative_suggestion": null
    }}

    Tu Análisis (Solo JSON):
"""


prompt_template = ChatPromptTemplate.from_template(VALIDATION_PROMPT)

def validate(original_query: str,final_content: Optional[str],execution_error: Optional[str] ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Valida la coherencia y relevancia de la respuesta final (texto o imagen)
    o formatea un mensaje de error adecuado.

    Args:
        original_query: La consulta inicial del usuario.
        final_content: El contenido generado (texto o base64 de imagen).
        execution_error: Mensaje de error si la ejecución del código falló.

    Returns:
        Una tupla (final_response_text, final_response_image, error_message):
        - final_response_text: El texto final validado o un texto acompañante para la imagen.
        - final_response_image: El string base64 de la imagen validada.
        - error_message: Un mensaje de error formateado para el usuario si aplica.
        (Si hay error_message, los otros dos suelen ser None).
    """
    print("Validador: Iniciando validación de la respuesta final...")
    final_text: Optional[str] = None
    final_image: Optional[str] = None
    error_message: Optional[str] = None

    # --- Paso 1: Manejar Errores Previos ---
    # Si hubo un error explícito en la ejecución del código, ese es el resultado principal.
    if execution_error:
        print(f"Validador: Detectado error de ejecución previo: {execution_error}")
        # Formatear un mensaje de error amigable para el usuario
        # Podríamos usar el LLM para mejorar este mensaje, pero por simplicidad lo hacemos directo.
        if "TimeoutExpired" in execution_error:
            error_message = "Lo siento, la generación de la visualización tardó demasiado y fue cancelada."
        elif "Error de ejecución de Python" in execution_error:
            # Podríamos intentar extraer detalles, pero puede ser riesgoso mostrar mucho
            error_message = "Lo siento, ocurrió un error técnico al generar la visualización solicitada."
        else:
            error_message = f"Lo siento, ocurrió un error inesperado al procesar tu solicitud: {execution_error}"
        print(f"Validador: Error formateado para el usuario: {error_message}")
        # En caso de error de ejecución, normalmente no mostramos contenido parcial.
        return None, None, error_message

    # --- Paso 2: Manejar Contenido Ausente ---
    # Si no hubo error de ejecución pero tampoco contenido, algo falló antes.
    if not final_content:
        print("Validador: No se proporcionó contenido final y no hubo error de ejecución explícito.")
        error_message = "Lo siento, no pude generar una respuesta para tu consulta con la información disponible o ocurrió un problema interno."
        return None, None, error_message

    # --- Paso 3: Preparar Contenido para Validación LLM ---
    is_image = isinstance(final_content, str) and final_content.startswith("data:image/png;base64,")
    content_type_description = "una imagen (generada correctamente)" if is_image else "el siguiente texto"
    # No pasar el base64 completo al LLM, solo una descripción o el texto.
    content_for_llm = "Se generó la imagen solicitada." if is_image else final_content
    error_context_for_llm = "" # Ya manejamos los errores explícitos

    print(f"Validador: Preparando para validar {content_type_description} con LLM.")
    print(f"Validador: Contenido para LLM (preview):\n---\n{content_for_llm[:200]}...\n---")

    # --- Paso 4: Llamar al LLM para Validación ---
    llm = get_gemini_llm(temperature=0.0) # Precisión para la validación
    if not llm:
        print("Advertencia (Validador): LLM no disponible. Aprobando la respuesta por defecto.")
        # Fallback: Si no hay LLM, asumimos que el contenido es válido.
        if is_image:
            final_image = final_content
            final_text = "Aquí tienes la visualización generada:" # Texto acompañante por defecto
        else:
            final_text = final_content
        return final_text, final_image, None # error_message es None aquí

    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "query": original_query,
            "result_description": f"Se produjo {content_type_description}:",
            "content_to_validate": content_for_llm,
            "error_context": error_context_for_llm
        })
        llm_content = response.content

        print(f"Validador: Respuesta cruda del LLM (Validación):\n---\n{llm_content}\n---")

        # Extraer JSON (similar a otros agentes)
        json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match_loose = re.search(r"(\{[\s\S]*?\})", llm_content, re.DOTALL)
            json_str = json_match_loose.group(1).strip() if json_match_loose else llm_content.strip()

        print(f"Validador: JSON extraído (Validación):\n---\n{json_str}\n---")

        validation_result = json.loads(json_str)

        # Verificar estructura del resultado de validación
        if not all(k in validation_result for k in ["is_valid", "reasoning", "alternative_suggestion"]):
            raise ValueError("Respuesta JSON de validación incompleta.")

        # --- Paso 5: Decidir Salida Final Basada en Validación ---
        if validation_result.get("is_valid"):
            print(f"Validador: Respuesta considerada VÁLIDA por LLM. Razonamiento: {validation_result.get('reasoning')}")
            if is_image:
                final_image = final_content
                # Podríamos pedir al LLM un mejor texto acompañante, pero usamos uno estándar por ahora
                final_text = "Aquí tienes la visualización generada de acuerdo a tu solicitud."
            else:
                final_text = final_content
            error_message = None # No hay error si es válido
        else:
            print(f"Validador: Respuesta considerada INVÁLIDA por LLM. Razonamiento: {validation_result.get('reasoning')}")
            # Si el LLM la invalida, generamos un mensaje de error para el usuario
            reason = validation_result.get('reasoning', 'La respuesta generada no parece adecuada.')
            suggestion = validation_result.get('alternative_suggestion')
            error_message = f"Lo siento, no pude generar una respuesta completamente relevante. {reason}"
            if suggestion:
                error_message += f" Sugerencia: {suggestion}"
            # No devolver contenido inválido
            final_text = None
            final_image = None

    except json.JSONDecodeError as e:
        print(f"Error Crítico (Validador): Fallo al parsear JSON de validación LLM: {e}. Aprobando respuesta por defecto.")
        print(f"Respuesta LLM que causó el error:\n{llm_content}")
        if is_image: final_image = final_content; final_text = "Visualización generada:"
        else: final_text = final_content
        error_message = None # Asumir válido si falla el parseo de validación
    except Exception as e:
        print(f"Error Inesperado durante la validación LLM: {e}. Aprobando respuesta por defecto.")
        import traceback
        traceback.print_exc()
        if is_image: final_image = final_content; final_text = "Visualización generada:"
        else: final_text = final_content
        error_message = None # Asumir válido si falla la validación misma

    # Asegurar un texto mínimo si hay imagen y no hubo error
    if final_image and not final_text and not error_message:
        final_text = "Visualización generada:"

    print(f"Validador: Finalizado. Texto: {'Presente' if final_text else 'Ausente'}, Imagen: {'Presente' if final_image else 'Ausente'}, Error: {'Presente' if error_message else 'Ausente'}")
    return final_text, final_image, error_message


# if __name__ == 'main':
#     print("--- Probando Agente de Validación ---")

#     # Prueba 1: Texto válido
#     query1 = "¿Cuál fue la duración del viaje de la fragata A?"
#     content1 = "La fragata A tardó 10 días en su viaje."
#     error1 = None
#     print("\n--- Prueba 1: Texto Válido ---")
#     txt1, img1, err1 = validate(query1, content1, error1)
#     print(f"Resultado: Texto='{txt1}', Imagen={img1 is not None}, Error='{err1}'")

#     # Prueba 2: Imagen válida (simulada)
#     query2 = "Muéstrame gráfico de barras de duración por tipo"
#     content2 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" # Base64 de 1x1 pixel png
#     error2 = None
#     print("\n--- Prueba 2: Imagen Válida ---")
#     txt2, img2, err2 = validate(query2, content2, error2)
#     print(f"Resultado: Texto='{txt2}', Imagen={img2 is not None}, Error='{err2}'")

#     # Prueba 3: Texto inválido
#     query3 = "¿Cuál fue la duración del viaje de la fragata A?"
#     content3 = "El capitán del barco B era Jones."
#     error3 = None
#     print("\n--- Prueba 3: Texto Inválido ---")
#     txt3, img3, err3 = validate(query3, content3, error3)
#     print(f"Resultado: Texto='{txt3}', Imagen={img3 is not None}, Error='{err3}'") # Esperamos error aquí

#     # Prueba 4: Error de ejecución previo
#     query4 = "Genera un gráfico complejo"
#     content4 = None # No se generó contenido
#     error4 = "Error de ejecución de Python (código 1). Detalles: KeyError 'columna_inexistente'"
#     print("\n--- Prueba 4: Error de Ejecución Previo ---")
#     txt4, img4, err4 = validate(query4, content4, error4)
#     print(f"Resultado: Texto='{txt4}', Imagen={img4 is not None}, Error='{err4}'") # Esperamos error aquí
