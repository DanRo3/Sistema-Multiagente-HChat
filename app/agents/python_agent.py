import re
from typing import Dict, Optional, Any
from app.core.llm import get_llm 
from langchain_core.prompts import ChatPromptTemplate

# --- Plantilla de Prompt para Generación de Código Python ---
PYTHON_GENERATION_PROMPT = """
    **Tarea:** Generar un script Python completo y ejecutable para crear una visualización de datos usando pandas y matplotlib/seaborn, basado en la consulta del usuario y los datos proporcionados.

    **Consulta Original del Usuario (solicitando visualización):**
    "{query}"

    **Datos Disponibles (Formato: String representando una lista Python de diccionarios):**
    ```python
    {data_str}


    Instrucciones y Restricciones MUY IMPORTANTES:

    Importaciones Necesarias: El script DEBE incluir import pandas as pd, import matplotlib.pyplot as plt, import seaborn as sns (si se usa), import ast (para cargar los datos), import io, import base64.

    Carga Segura de Datos: El script DEBE usar ast.literal_eval() para convertir el string data_str en una lista de diccionarios Python. NUNCA uses eval(). Luego, crea un DataFrame de Pandas: df = pd.DataFrame(ast.literal_eval(data_string)).

    Procesamiento y Gráfico: Usa pandas para cualquier manipulación o agregación necesaria (ej. value_counts(), groupby().mean(), sort_values()). Luego, usa matplotlib.pyplot o seaborn para generar el tipo de gráfico que mejor se ajuste a la consulta del usuario (barras, líneas, dispersión, etc.). Asegúrate de que las columnas usadas existan en los datos.

    Etiquetas y Título: El gráfico DEBE tener un título claro y relevante para la consulta original, y etiquetas descriptivas en los ejes X e Y. Si es un gráfico de barras o similar, considera rotar las etiquetas del eje X si son largas (plt.xticks(rotation=45, ha='right')). Usa plt.tight_layout() para ajustar el espaciado.

    Guardar Figura (¡OBLIGATORIO!): El script DEBE guardar la figura resultante usando exactamente plt.savefig('plot.png', bbox_inches='tight').

    Limpieza de Matplotlib: El script DEBE incluir plt.close() después de plt.savefig() para cerrar la figura y liberar memoria.

    NO USAR plt.show(): El script NO debe intentar mostrar el gráfico interactivamente.

    Manejo Básico de Errores: Envuelve la lógica de carga de datos, procesamiento y trazado dentro de un bloque try...except Exception as e:. En el bloque except, imprime un mensaje de error claro usando print(f"Error generating plot: {{e}}"). Esto es vital para la depuración si el código falla. Añade un print("Plot saved successfully.") al final del bloque try si todo va bien.

    Autocontenido: El script debe ser funcional por sí mismo sin dependencias externas no declaradas en las importaciones.

    Enfoque y Simplicidad: Céntrate estrictamente en generar el gráfico solicitado. Evita código complejo, funciones personalizadas innecesarias o análisis que no se pidieron explícitamente.

    Salida Final: Responde únicamente con el código Python completo, encerrado en un único bloque de markdown python .... No incluyas NADA antes o después de este bloque (ni explicaciones, ni saludos).

    Script Python Generado (Solo Código dentro de python ...):
"""


prompt_template = ChatPromptTemplate.from_template(PYTHON_GENERATION_PROMPT)

def generate_python_code(original_query: str, data_for_python: Optional[Any]) -> Dict[str, Optional[str]]:
    """
    Genera código Python para crear una visualización basada en la consulta
    y los datos proporcionados por el contextualizador.

    Args:
        original_query: La consulta original del usuario que pedía la visualización.
        data_for_python: Un string que representa una lista de diccionarios Python
                        con los datos extraídos y preparados por el contextualizador.

    Returns:
        Un diccionario con la clave 'python_code' que contiene el script generado
        como string, o None si ocurre un error o no hay datos.
    """
    print("Agente Python: Iniciando generación de código...")

    # Validar entrada
    if not data_for_python:
        print("Agente Python: No se proporcionaron datos ('data_for_python'). No se puede generar código.")
        return {"python_code": None}
    if not isinstance(data_for_python, str):
        print(f"Advertencia (Agente Python): 'data_for_python' no es un string (tipo: {type(data_for_python)}). Intentando convertir.")
        try:
            data_str = str(data_for_python)
        except Exception as e:
            print(f"Error: No se pudo convertir 'data_for_python' a string: {e}")
            return {"python_code": None}
    else:
        data_str = data_for_python

    print(f"Agente Python: Datos recibidos (primeros 200 chars): {data_str[:200]}...")

    # Obtener el LLM (quizás un poco más de temperatura para generar código variado pero siguiendo instrucciones)
    llm = get_llm()
    if not llm:
        print("Error Crítico: LLM no disponible para el agente Python.")
        return {"python_code": None}

    # Crear y ejecutar la cadena
    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "query": original_query,
            "data_str": data_str  # Pasamos el string formateado de datos
        })
        content = response.content

        print(f"Agente Python: Respuesta cruda del LLM recibida:\n---\n{content}\n---")

        # Extraer el bloque de código Python de forma robusta
        # Busca ```python seguido de cualquier cosa hasta el próximo ```
        code_match = re.search(r"```python\s*([\s\S]+?)\s*```", content, re.DOTALL)
        if code_match:
            python_code = code_match.group(1).strip()
            print(f"Agente Python: Código Python extraído:\n---\n{python_code[:500]}...\n---") # Log inicial del código
            # Validación básica simple (contiene imports clave y savefig)
            if "import pandas" in python_code and \
            "import matplotlib.pyplot" in python_code and \
            "ast.literal_eval" in python_code and \
            "plt.savefig('plot.png'" in python_code and \
            "plt.close()" in python_code:
                print("Agente Python: Validación básica del código superada.")
                return {"python_code": python_code}
            else:
                print("Error: El código generado no parece cumplir con las restricciones básicas (imports, savefig, close, literal_eval).")
                return {"python_code": None} # Devolver None si no cumple requisitos mínimos
        else:
            print("Error: No se encontró un bloque de código Python válido (```python ... ```) en la respuesta del LLM.")
            return {"python_code": None}

    except Exception as e:
        print(f"Error Inesperado durante la generación de código Python: {e}")
        import traceback
        traceback.print_exc()
        return {"python_code": None}


# if __name__ == 'main':
# print("--- Probando Agente Generador de Código Python ---")

# # Simular salida del contextualizador
# mock_query = "Muéstrame la duración promedio por tipo de barco en un gráfico de barras"
# # String representando una lista de diccionarios
# mock_data = '[{"ship_type": "fragata", "avg_duration_days": 15.5}, {"ship_type": "vapor", "avg_duration_days": 8.2}, {"ship_type": "goleta", "avg_duration_days": 20.1}]'

# result = generate_python_code(mock_query, mock_data)

# if result and result.get("python_code"):
#     print("\n--- Código Generado (Prueba) ---")
#     print(result["python_code"])
#     print("-------------------------------")
#     # Aquí podrías intentar ejecutar este código usando la utilidad del siguiente paso
#     # from app.agents.utils.code_executor import execute_python_safely
#     # output, error = execute_python_safely(result["python_code"])
#     # print(f"\nResultado Ejecución (Prueba): Output={output[:100] if output else None}, Error={error}")
# else:
#     print("\nNo se pudo generar código en la prueba.")
