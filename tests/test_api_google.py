import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Configuración ---
# Cargar variables de entorno desde un archivo .env (si existe)
# Asegúrate de que tu archivo .env esté en el mismo directorio o en uno superior
# y contenga la línea: GEMINI_API_KEY="tu_clave_aqui"
load_dotenv()

# Obtener la API Key desde las variables de entorno
api_key = os.getenv("GEMINI_API_KEY")

model_name = "gemini-2.0-flash"

# Consulta de prueba simple
test_prompt = "Hola Gemini, ¿puedes contar hasta 3?"

# --- Fin Configuración ---

def test_api_key():
    """
    Intenta configurar la librería y hacer una llamada simple a la API de Gemini.
    """
    if not api_key:
        print("Error: La variable de entorno GEMINI_API_KEY no está configurada.")
        print("Asegúrate de tener un archivo .env con GEMINI_API_KEY=\"tu_clave_aqui\"")
        return

    print(f"Intentando configurar Google GenAI con la API Key...")
    try:
        # Configurar la librería con tu API key
        genai.configure(api_key=api_key)
        print("Configuración de GenAI exitosa.")

        # Inicializar el modelo generativo
        print(f"Inicializando el modelo: {model_name}...")
        # Opciones de generación (opcional)
        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # Cuántas respuestas generar
            # stop_sequences=['.'], # Dónde detener la generación
            # max_output_tokens=100, # Límite de tokens
            temperature=0.7 # Controla la aleatoriedad
        )
        # Configuraciones de seguridad (opcional, usa defaults si no se especifica)
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     # ... otras categorías
        # ]

        model = genai.GenerativeModel(
            model_name=model_name,
            # generation_config=generation_config,
            # safety_settings=safety_settings
        )
        print("Modelo inicializado correctamente.")

        # Hacer la llamada a la API para generar contenido
        print(f"\nEnviando prompt al modelo: '{test_prompt}'...")
        response = model.generate_content(test_prompt)

        # Imprimir la respuesta
        print("\n--- Respuesta del Modelo ---")
        # El texto generado está en response.text
        print(response.text)
        print("--------------------------")
        print("\n¡Prueba de API Key exitosa!")

    except Exception as e:
        print("\n--- ERROR DURANTE LA PRUEBA ---")
        print(f"Ocurrió un error: {e}")
        print("\nPosibles causas:")
        print("- API Key incorrecta o inválida.")
        print("- API 'Generative Language API' (o Vertex AI) no habilitada en tu proyecto de Google Cloud.")
        print("- Problemas de permisos asociados a la API Key o al proyecto.")
        print("- Restricciones de red o firewall.")
        print("- Nombre del modelo incorrecto o no accesible con tu clave.")
        print("- Problemas de facturación en Google Cloud (si aplica).")
        import traceback
        traceback.print_exc() # Imprime el traceback completo para más detalles

# --- Ejecutar la prueba ---
if __name__ == "__main__":
    test_api_key()