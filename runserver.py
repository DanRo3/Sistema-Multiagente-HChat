import subprocess

def run_uvicorn():
    """Ejecuta el servidor uvicorn con la configuración especificada."""
    command = [
        "uvicorn",
        "app.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8008"
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al iniciar el servidor: {e}")

if __name__ == "__main__":
    run_uvicorn()
