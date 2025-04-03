import subprocess
import base64
import os
import sys
import tempfile # Para crear entornos de ejecución aislados temporalmente
import time # Para medir tiempo si es necesario (aunque timeout lo maneja)
from typing import Tuple, Optional
from app.core.config import settings # Importa la configuración para el timeout

# --- ADVERTENCIA DE SEGURIDAD FUNDAMENTAL ---
# La ejecución de código generado por un LLM es INTRÍNSECAMENTE PELIGROSA.
# Esta implementación utiliza `subprocess` en un directorio temporal y un `timeout`,
# lo cual proporciona cierto nivel de aislamiento y control, pero NO es una
# solución de sandboxing completa y robusta contra código malicioso sofisticado.
#
# RIESGOS POTENCIALES:
# - Acceso al sistema de archivos (aunque limitado por `cwd=tmpdir`).
# - Consumo excesivo de CPU/Memoria (parcialmente mitigado por timeout).
# - Acceso a red (si no está bloqueado a nivel de sistema operativo/firewall).
# - Explotación de vulnerabilidades en librerías permitidas (pandas, matplotlib).
#
# ALTERNATIVAS MÁS SEGURAS (pero más complejas):
# 1. Contenedores Docker dedicados por ejecución (mejor aislamiento).
# 2. Librerías de Sandboxing (e.g., RestrictedPython, nsjail) - pueden limitar funcionalidad.
# 3. Servicios de ejecución de código en la nube (e.g., AWS Lambda, Google Cloud Functions con restricciones).
# 4. **Evitar la ejecución de código arbitrario:** Hacer que el LLM genere parámetros
#    para funciones de trazado predefinidas y seguras en tu backend.
#
# ¡NO USES ESTE CÓDIGO EN PRODUCCIÓN SIN UNA EVALUACIÓN DE RIESGOS EXHAUSTIVA
# Y POTENCIALMENTE MEDIDAS DE SEGURIDAD ADICIONALES!
# ---------------------------------------------------

# Nombre de archivo estándar que el código generado debe crear
PLOT_FILENAME = "plot.png"

def execute_python_safely(python_code: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Ejecuta un script de Python proporcionado como string de forma semi-segura
    utilizando un proceso separado en un directorio temporal y con un timeout.

    Intenta capturar una imagen 'plot.png' si se genera, o la salida estándar (stdout).

    Args:
        python_code: El string que contiene el código Python a ejecutar.

    Returns:
        Una tupla (execution_output, execution_error):
        - execution_output (Optional[str]):
            - Si 'plot.png' se genera: String base64 de la imagen con prefijo data URI.
            - Si no se genera imagen: El contenido de stdout del script.
            - None si hubo error grave o no hubo salida.
        - execution_error (Optional[str]):
            - Mensaje descriptivo si ocurre un error (timeout, error de Python, etc.).
            - None si la ejecución fue exitosa (aunque no produzca salida útil).
    """
    if not python_code or not python_code.strip():
        print("Executor: No se proporcionó código Python válido para ejecutar.")
        return None, "No se proporcionó código Python para ejecutar."

    # Validar imports básicos requeridos (medida de seguridad muy leve)
    required_imports = ["import pandas", "import matplotlib.pyplot", "import ast"]
    if not all(imp in python_code for imp in required_imports):
         missing = [imp for imp in required_imports if imp not in python_code]
         error_msg = f"El código parece no incluir importaciones requeridas: {', '.join(missing)}"
         print(f"Executor Error: {error_msg}")
         # Podríamos detener la ejecución aquí si queremos ser más estrictos
         # return None, error_msg
         # O continuar y dejar que falle en ejecución (como está ahora)


    print(f"Executor: Preparando ejecución segura (Timeout: {settings.CODE_EXECUTION_TIMEOUT}s)...")
    execution_output: Optional[str] = None
    execution_error: Optional[str] = None

    # Usar un directorio temporal asegura que los archivos creados (script, plot)
    # estén aislados y se limpien automáticamente al finalizar.
    with tempfile.TemporaryDirectory() as tmpdir:
        script_filename = "generated_script.py"
        script_path = os.path.join(tmpdir, script_filename)
        plot_path = os.path.join(tmpdir, PLOT_FILENAME)

        print(f"Executor: Directorio temporal creado: {tmpdir}")
        print(f"Executor: Escribiendo código en: {script_path}")

        # Escribir el código en el archivo temporal
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(python_code)
        except IOError as e:
            execution_error = f"Error de E/S al escribir el script temporal: {e}"
            print(f"Executor: {execution_error}")
            # No podemos continuar si no se puede escribir el script
            return None, execution_error

        start_time = time.time()
        try:
            # Ejecutar el script en un subproceso
            # - sys.executable: Usa el mismo intérprete de Python que ejecuta FastAPI.
            # - cwd=tmpdir: Establece el directorio de trabajo al temporal. Crucial
            #   para que `savefig('plot.png')` guarde el archivo en el lugar correcto
            #   y para limitar el acceso a archivos fuera de este directorio (parcialmente).
            # - capture_output=True: Captura stdout y stderr.
            # - text=True: Decodifica stdout/stderr como texto.
            # - timeout: Lanza TimeoutExpired si tarda demasiado.
            # - check=False: Evita que lance CalledProcessError si el script falla (returncode != 0).
            # - encoding/errors: Manejo de caracteres especiales en la salida.
            print(f"Executor: Ejecutando script '{script_filename}' en subprocess...")
            process = subprocess.run(
                [sys.executable, script_filename], # Usar script_filename relativo a cwd
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=settings.CODE_EXECUTION_TIMEOUT,
                check=False,
                encoding='utf-8',
                errors='replace'
            )

            end_time = time.time()
            duration = end_time - start_time
            print(f"Executor: Subprocess finalizado en {duration:.2f} segundos.")

            stdout = process.stdout.strip() if process.stdout else ""
            stderr = process.stderr.strip() if process.stderr else ""

            # Loggear siempre stdout y stderr para depuración
            if stdout:
                print(f"Executor: Salida Estándar (stdout):\n---\n{stdout}\n---")
            if stderr:
                print(f"Executor: Salida de Error (stderr):\n---\n{stderr}\n---")

            # Comprobar si el script falló (returncode no es 0)
            if process.returncode != 0:
                # Si hay stderr, es probablemente el error más relevante. Si no, usa stdout.
                error_detail = stderr if stderr else stdout
                execution_error = f"Error de ejecución de Python (código {process.returncode}). Detalles: {error_detail}"
                print(f"Executor: {execution_error}")
                # No intentar buscar la imagen si el script falló

            else:
                # El script terminó sin errores (returncode 0)
                print(f"Executor: Script ejecutado exitosamente (código 0). Verificando '{PLOT_FILENAME}'...")
                # Comprobar si se generó el archivo de imagen esperado
                if os.path.exists(plot_path):
                    print(f"Executor: Archivo '{PLOT_FILENAME}' encontrado en {tmpdir}.")
                    try:
                        # Leer los bytes de la imagen
                        with open(plot_path, "rb") as img_file:
                            img_bytes = img_file.read()
                        # Codificar en Base64
                        base64_encoded_img = base64.b64encode(img_bytes).decode('utf-8')
                        # Crear el Data URI para incrustar en HTML/JSON
                        execution_output = f"data:image/png;base64,{base64_encoded_img}"
                        print(f"Executor: Imagen codificada a Base64 exitosamente (longitud: {len(execution_output)}).")
                    except Exception as e:
                        execution_error = f"Error al leer o codificar el archivo de imagen '{PLOT_FILENAME}': {e}"
                        print(f"Executor: {execution_error}")
                        # Aunque falló la codificación, el script se ejecutó bien, así que no sobrescribimos la salida
                        # podríamos devolver un mensaje indicando el fallo de codificación
                        execution_output = None # O indicar error aquí?

                    # Limpieza explícita (aunque tempfile debería hacerlo) - Opcional
                    # try: os.remove(plot_path) except Exception: pass

                elif stdout:
                     # No se creó plot.png, pero hubo salida estándar. Usar stdout.
                     print(f"Executor: No se encontró '{PLOT_FILENAME}'. Usando stdout como salida.")
                     execution_output = stdout
                else:
                     # El script se ejecutó bien, no creó plot.png y no imprimió nada.
                     print(f"Executor: No se encontró '{PLOT_FILENAME}' y no hubo salida en stdout.")
                     execution_output = "El script se ejecutó correctamente pero no produjo salida de texto ni imagen."


        except subprocess.TimeoutExpired:
            execution_error = f"Error Crítico: La ejecución del código excedió el tiempo límite de {settings.CODE_EXECUTION_TIMEOUT} segundos."
            print(f"Executor: {execution_error}")
        except FileNotFoundError:
            # Esto ocurriría si sys.executable no es válido
            execution_error = "Error Crítico: No se pudo encontrar el intérprete de Python configurado."
            print(f"Executor: {execution_error}")
        except Exception as e:
            # Captura cualquier otro error inesperado durante el manejo del subprocess
            execution_error = f"Error Inesperado durante la ejecución del código: {e}"
            print(f"Executor: {execution_error}")
            import traceback
            traceback.print_exc()

    # Al salir del `with tempfile.TemporaryDirectory()`, el directorio y su contenido
    # (script.py, plot.png si existe) se eliminarán automáticamente.
    print(f"Executor: Limpiando directorio temporal {tmpdir}.")
    print(f"Executor: Finalizado. Output: {'Presente' if execution_output else 'Ausente'}, Error: {'Presente' if execution_error else 'Ausente'}")

    return execution_output, execution_error

# # --- Bloque de prueba simple ---
# if __name__ == '__main__':
#     print("--- Probando Ejecutor de Código Seguro ---")

#     # Prueba 1: Código que genera un plot correctamente
#     print("\n--- Prueba 1: Plot Correcto ---")
#     code1 = """
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import ast
# import io
# import base64
# import time

# print("Ejecutando script de prueba 1...")
# data_string = '[{"tipo": "A", "valor": 10}, {"tipo": "B", "valor": 25}, {"tipo": "A", "valor": 15}]'
# try:
#     data_list = ast.literal_eval(data_string)
#     df = pd.DataFrame(data_list)
#     print("DataFrame creado:", df.shape)
#     # Agrupar y trazar
#     df_agg = df.groupby('tipo')['valor'].mean().reset_index()
#     print("DataFrame Agregado:", df_agg.shape)
#     plt.figure(figsize=(5, 3))
#     sns.barplot(x='tipo', y='valor', data=df_agg)
#     plt.title('Valor Promedio por Tipo')
#     plt.xlabel('Tipo')
#     plt.ylabel('Valor Promedio')
#     plt.tight_layout()
#     plt.savefig('plot.png', bbox_inches='tight') # ¡Nombre de archivo correcto!
#     plt.close()
#     print("Plot saved successfully.")
# except Exception as e:
#     print(f"Error generating plot: {e}")
# """
#     output1, error1 = execute_python_safely(code1)
#     print(f"Resultado Prueba 1: Error={error1}")
#     if output1 and output1.startswith("data:image"):
#         print(f"Resultado Prueba 1: Imagen Base64 generada (primeros 100 chars): {output1[:100]}...")
#     elif output1:
#          print(f"Resultado Prueba 1: Salida de Texto: {output1}")


#     # Prueba 2: Código con error de Python
#     print("\n--- Prueba 2: Error de Python ---")
#     code2 = """
# import pandas as pd
# import ast
# print("Intentando acceder a columna inexistente...")
# data_string = '[{"colA": 1}]'
# try:
#     data_list = ast.literal_eval(data_string)
#     df = pd.DataFrame(data_list)
#     print(df['colB']) # Esto fallará
#     print("Esto no debería imprimirse")
# except Exception as e:
#     print(f"Error interno capturado: {e}") # El script captura, pero podría no hacerlo
# # print(1/0) # Error no capturado explícitamente
# """
#     output2, error2 = execute_python_safely(code2)
#     print(f"Resultado Prueba 2: Output={output2}, Error={error2}") # Esperamos un error aquí

#     # Prueba 3: Código que tarda demasiado (timeout)
#     print("\n--- Prueba 3: Timeout ---")
#     code3 = """
# import time
# print("Iniciando bucle largo...")
# time.sleep(15) # Asume que el timeout es menor que 15s
# print("Bucle terminado.")
# """
#     # Asegúrate que settings.CODE_EXECUTION_TIMEOUT sea menor que 15 (e.g., 10)
#     # Modifica temporalmente si es necesario para la prueba:
#     # from app.core import config
#     # original_timeout = config.settings.CODE_EXECUTION_TIMEOUT
#     # config.settings.CODE_EXECUTION_TIMEOUT = 5 # Timeout corto para probar
#     output3, error3 = execute_python_safely(code3)
#     # config.settings.CODE_EXECUTION_TIMEOUT = original_timeout # Restaurar timeout
#     print(f"Resultado Prueba 3: Output={output3}, Error={error3}") # Esperamos error de timeout

#     # Prueba 4: Código que solo imprime texto
#     print("\n--- Prueba 4: Salida de Texto ---")
#     code4 = """
# print("Hola desde el script.")
# print("Línea 2.")
# """
#     output4, error4 = execute_python_safely(code4)
#     print(f"Resultado Prueba 4: Error={error4}")
#     print(f"Resultado Prueba 4: Salida Texto:\n{output4}")
