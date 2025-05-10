import pandas as pd
import json
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import os

# Configuración del modelo LLM
# Asegúrate de que la variable de entorno OPENAI_API_KEY esté configurada
# o pasa el token directamente: OpenAI(api_token="sk-...")
llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))

def extraer_nombres_y_emails(excel_path: str) -> str:
    # Cargar el archivo Excel
    df = pd.read_excel(excel_path)

    # Envolver el DataFrame con PandasAI
    # Nota: "response_format" ahora está implícito a través del prompt
    # o puedes mantenerlo si quieres que PandasAI envuelva el DataFrame.
    smart_df = SmartDataframe(df, config={
        "llm": llm,
        "enforce_privacy": False, # Ten cuidado con datos sensibles
        "verbose": True,
        "use_error_correction_framework": True, # Puede ser útil
        "response_format": "json" # Esto hará que el resultado sea un dict {"type": "dataframe", "value": df_obj}
    })

    # Prompt para que PandasAI devuelva un DataFrame con las columnas renombradas
    prompt = """
    Del DataFrame proporcionado:
    1. Selecciona la columna 'Nombre' y renómbrala a 'nombre_trabajador'.
    2. Selecciona la columna 'Email' y renómbrala a 'email'.
    3. Devuelve un nuevo DataFrame que contenga únicamente estas dos columnas ('nombre_trabajador', 'email').
    Asegúrate de que el DataFrame resultante solo tenga estas dos columnas con esos nombres exactos.
    """

    # PandasAI intentará devolver un DataFrame.
    # Con response_format="json", esperaremos {"type": "dataframe", "value": <DataFrame_object>}
    resultado_pai = smart_df.chat(prompt)
    print(f"Resultado crudo de PandasAI: {resultado_pai}")


    # Validar que la salida tiene el formato esperado
    if (
        isinstance(resultado_pai, dict) and
        resultado_pai.get("type") == "dataframe" and
        isinstance(resultado_pai.get("value"), pd.DataFrame)
    ):
        df_resultado = resultado_pai["value"]
        # Convertir el DataFrame a lista de diccionarios
        lista_de_diccionarios = df_resultado.to_dict(orient='records')

        # Construir el objeto JSON final que el usuario quiere
        json_final_obj = {
            "type": "list",
            "value": lista_de_diccionarios
        }
        # Devolver el string JSON
        return json.dumps(json_final_obj, ensure_ascii=False, indent=2)
    elif isinstance(resultado_pai, pd.DataFrame): # Caso sin response_format="json"
        df_resultado = resultado_pai
        lista_de_diccionarios = df_resultado.to_dict(orient='records')
        json_final_obj = {
            "type": "list",
            "value": lista_de_diccionarios
        }
        return json.dumps(json_final_obj, ensure_ascii=False, indent=2)
    else:
        # Loguear el resultado inesperado para depuración
        error_msg = f"La respuesta del LLM no fue un DataFrame como se esperaba. Recibido: {type(resultado_pai)}"
        if isinstance(resultado_pai, dict):
            error_msg += f" | Contenido: {resultado_pai}"
        print(error_msg)
        raise ValueError("La respuesta del LLM no fue un DataFrame válido o no está en el formato esperado.")


# Ejemplo de uso
if __name__ == "__main__":
    # Crear un archivo Excel de prueba si no existe
    sample_data = {
        'ID': [106, 110, 108],
        'Nombre': ["Raúl Mendoza", "Daniela Luque", "Carmen Leyva"],
        'Email': ["-q1gjz@aol.com", "lnjvn-hicare@aol.com", "36l53wp@yahoo.com"],
        'Ciudad Base': ["León", "Mérida", "Tijuana"],
        'Años de Experiencia': [5, 8, 9],
        'Licencia': ["Camión B1", "Camión C1", "Camión C3"],
        'Vehículo Asignado': ["Scania R-450", "Kenworth W-900", "Peterbilt 389"]
    }
    sample_df = pd.DataFrame(sample_data)
    if not os.path.exists("tests"):
        os.makedirs("tests")
    ruta_excel_test = "tests/trabajadores.xlsx"
    sample_df.to_excel(ruta_excel_test, index=False)
    print(f"Archivo de prueba '{ruta_excel_test}' creado/actualizado.")

    try:
        print(f"Procesando archivo: {ruta_excel_test}")
        resultado_json = extraer_nombres_y_emails(ruta_excel_test)
        print("\nResultado JSON final:")
        print(resultado_json)
    except Exception as e:
        print("\nError al procesar el archivo:", e)