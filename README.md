# 🤖📊 Sistema Multiagente IA para Consultas en Lenguaje Natural y Visualización de Datos

**Proyecto de Tesis: Construcción de un sistema multiagente basado en IA para la extracción y visualización de información desde bases de datos vectoriales mediante lenguaje natural.**

---

## 📝 Resumen del Proyecto

Este proyecto presenta un sistema avanzado basado en Inteligencia Artificial diseñado para interactuar con bases de datos vectoriales complejas utilizando consultas en lenguaje natural (español). El sistema, construido como un microservicio con FastAPI, no solo extrae información relevante mediante búsqueda por similitud semántica, sino que también es capaz de interpretar la intención del usuario para generar respuestas textuales concisas o visualizaciones de datos (gráficas) dinámicamente generadas.

El núcleo del sistema es una arquitectura multiagente orquestada con LangChain y Langraph, donde cada agente se especializa en una tarea específica (moderación, recuperación, contextualización, generación de código, validación), asegurando un procesamiento modular, robusto y escalable.

## ✨ Características Principales

**Procesamiento de Lenguaje Natural:** Interpreta consultas de usuario en español.
**Búsqueda Semántica:** Utiliza embeddings y FAISS para encontrar información relevante en una base de datos vectorial.
**Reconocimiento de Intención:** Determina si el usuario busca una respuesta textual o una visualización.
**Generación Contextual:** Enriquece la información recuperada con estadísticas o datos complementarios.
**Generación Dinámica de Código:** Crea scripts Python bajo demanda para análisis o visualización.
**Visualización Dinámica:** Ejecuta código Python para generar gráficos (imágenes PNG) como respuesta.
**Respuestas Flexibles:** Devuelve resultados en formato texto, código (como texto) o imagen (base64).
**Arquitectura Multiagente:** Flujo de trabajo orquestado con LangChain/Langraph para modularidad y control.
**API Robusta:** Expone la funcionalidad a través de un endpoint FastAPI.

## 🏗️ Arquitectura del Sistema

El sistema sigue una arquitectura de microservicio basada en FastAPI, con un flujo de trabajo multiagente gestionado por Langraph:

1. **Recepción (FastAPI):** El usuario envía una consulta vía POST al endpoint `/api/query`.
2. **Moderación (Moderator Agent):** Analiza la consulta, extrae palabras clave y determina la intención (texto/visualización).
3. **Recuperación (Retrieval Agent):** Genera embeddings de la consulta/keywords y busca documentos similares en la base de datos vectorial FAISS.
4. **Contextualización (Contextualizer Agent):** Sintetiza la información recuperada, añade contexto y prepara los datos para la respuesta o para el agente Python.
5. **Generación/Visualización (Python Agent & Executor):** Si se requiere visualización, genera código Python (e.g., usando Pandas, Matplotlib), lo ejecuta de forma segura y captura la imagen resultante.
6. **Validación (Validation Agent):** Revisa la coherencia y relevancia de la respuesta generada (texto o imagen) respecto a la consulta original.
7. **Entrega (Moderator Agent / FastAPI):** Ensambla la respuesta final y la devuelve al usuario en formato JSON, conteniendo texto y/o una imagen codificada en base64.

```console
graph LR
    A[Usuario] -- Consulta (NL) --> B(FastAPI Endpoint /api/query);
    B -- Query --> C{Moderator Agent (Proxy)};
    C -- Keywords/Intent --> D[Retrieval Agent];
    D -- Embeddings --> E[(FAISS Vector DB)];
    E -- Documentos Similares --> D;
    D -- Documentos --> C;
    C -- Documentos + Intent --> F[Contextualizer Agent];
    F -- Datos Procesados --> G{Conditional: Visualizar?};
    G -- No --> J[Validation Agent];
    G -- Sí --> H[Python Agent];
    H -- Código Python --> I[Code Executor & Visualizer];
    I -- Imagen/Resultado Ejecución --> J;
    F -- Resumen Textual --> J;
    J -- Respuesta Preliminar --> K[Moderator Agent (Ensamblador)];
    K -- Respuesta Final (Texto/Imagen) --> B;
    B -- JSON Response --> A;
```

🛠️ Tecnologías Utilizadas

Backend: FastAPI

Orquestación de Agentes: LangChain, Langraph

Base de Datos Vectorial: FAISS (Facebook AI Similarity Search)

Modelo de Embeddings: sentence-transformers/all-MiniLM-L6-v2 (vía Hugging Face)

Modelo Generativo (LLM): Google Gemini API (específicamente gemini-1.5-flash-latest o similar)

Procesamiento y Ejecución de Código: Python 3.x

Librerías Clave: langchain, langgraph, fastapi, uvicorn, faiss-cpu (o faiss-gpu), sentence-transformers, torch, pandas, matplotlib, seaborn, python-dotenv, google-generativeai

Contenerización (Opcional): Docker

📂 Estructura del Proyecto
/tesis-multiagente-bi/
├── app/                      # Código fuente de la aplicación FastAPI
│   ├── api/                  # Endpoints y Schemas Pydantic
│   ├── agents/               # Lógica de cada agente
│   ├── core/                 # Configuración, LLM, Embeddings
│   ├── orchestration/        # Definición y construcción del grafo Langraph
│   ├── vector_store/         # Lógica para cargar/interactuar con FAISS
│   └── utils/                # Utilidades generales y ejecución segura de código
│   └── main.py               # Entrypoint de FastAPI
├── data/                     # Datos fuente (e.g., CSV)
├── vector_store_index/       # Índice FAISS generado (.faiss, .pkl)
├── notebooks/                # Jupyter notebooks (e.g., para crear el índice)
├── tests/                    # Pruebas unitarias/integración
├── .env.example              # Ejemplo de variables de entorno
├── .gitignore
├── Dockerfile                # Opcional
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
🚀 Cómo Empezar
Prerrequisitos

Python 3.9+

pip (gestor de paquetes de Python)

Git

1. Clonar el Repositorio
git clone < url-del-repositorio >
cd tesis-multiagente-bi
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
2. Crear Entorno Virtual (Recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
3. Instalar Dependencias
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
4. Configurar Variables de Entorno

Copia el archivo de ejemplo y edítalo con tus credenciales:

cp .env.example .env
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Abre el archivo .env y añade tu clave API de Google Gemini:

# .env
GEMINI_API_KEY="TU_API_KEY_DE_GOOGLE_GEMINI"
# Otras configuraciones si las hubiera (normalmente se definen en app/core/config.py)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Dotenv
IGNORE_WHEN_COPYING_END
5. Construir la Base de Datos Vectorial FAISS

Asegúrate de tener tus datos fuente (e.g., datos.csv) en la carpeta data/.

Utiliza el notebook notebooks/1_build_vector_store.ipynb (o el script correspondiente) para procesar tus datos y generar el índice FAISS.

Este proceso leerá el CSV, generará embeddings usando sentence-transformers/all-MiniLM-L6-v2 y guardará los archivos my_data_index.faiss y my_data_index.pkl (o los nombres que hayas definido).

MUY IMPORTANTE: Copia los archivos .faiss y .pkl generados a la carpeta vector_store_index/ en la raíz del proyecto.

6.Ejecutar la Aplicación FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

--reload: Reinicia el servidor automáticamente al detectar cambios en el código (útil para desarrollo).

--host 0.0.0.0: Permite acceder al servidor desde otras máquinas en la red local.

--port 8000: Especifica el puerto (puedes cambiarlo si es necesario).

La API estará disponible en http://localhost:8000. Puedes ver la documentación interactiva (Swagger UI) en http://localhost:8000/docs.

⚙️ Uso (API)

Puedes interactuar con el sistema enviando peticiones POST al endpoint /api/query.

Endpoint: POST /api/query

Request Body (JSON):

{
  "query": "Muéstrame un gráfico de barras con la duración promedio de los viajes por tipo de barco."
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

o

{
  "query": "¿Cuál es el rol del capitán John Doe?"
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

Ejemplo con curl:

curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Genera una gráfica de los puertos de salida más comunes"}'
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Response Body (JSON):

La respuesta contendrá un campo text_response y/o image_response (imagen codificada en base64), dependiendo de la consulta y la intención detectada.

Respuesta Textual:

{
  "text_response": "El rol del capitán John Doe es supervisar la navegación y seguridad del buque.",
  "image_response": null,
  "error": null
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

Respuesta Visual:

{
  "text_response": "Aquí tienes una gráfica mostrando los puertos de salida más comunes.",
  "image_response": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAY...", // String largo en base64
  "error": null
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

Error:

{
  "text_response": null,
  "image_response": null,
  "error": "Lo siento, ocurrió un error al procesar tu solicitud: [Detalle del error]"
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
✅ Pruebas

Para verificar rápidamente la carga y búsqueda en el índice FAISS (antes de ejecutar toda la aplicación), puedes usar el script de prueba:

python test_faiss_query.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Este script cargará el índice desde vector_store_index/ y realizará una consulta de ejemplo, mostrando los resultados encontrados. Asegúrate de ajustar la consulta de prueba dentro del script para que sea relevante a tus datos.

Las pruebas unitarias y de integración más completas se encuentran en el directorio tests/.

🤝 Contribuciones

Este es un proyecto de tesis, pero las sugerencias y mejoras son bienvenidas. Por favor, abre un issue para discutir cambios importantes antes de realizar un pull request.

📄 Licencia

(Opcional: Especifica una licencia si aplica, e.g., MIT, Apache 2.0, o indica que es para fines académicos).

🙏 Agradecimientos

(Opcional: Menciona a tus tutores, fuentes de datos, librerías clave, etc.).

Desarrollado como parte de [Nombre de tu Tesis/Programa Académico] en [Tu Universidad/Institución].

**Puntos Clave y Mejoras en este README:**

**Título Claro y Emojis:** Llama la atención y resume el propósito.
**Resumen Conciso:** Explica el qué, el porqué y el cómo de alto nivel.
**Características Detalladas:** Lista las capacidades clave del sistema.
**Diagrama de Arquitectura:** Usa Mermaid para visualizar el flujo (se renderizará en plataformas como GitHub/GitLab).
**Tecnologías Organizadas:** Lista clara de todas las herramientas usadas.
**Estructura de Proyecto:** Facilita la navegación por el código fuente.
**Instrucciones Claras (`Getting Started`):** Pasos numerados y comandos exactos para la instalación y configuración.
**Énfasis en Pasos Críticos:** Resalta la importancia de configurar `.env` y construir/ubicar el índice FAISS.
**Ejemplos de Uso:** Muestra cómo interactuar con la API usando `curl` y los formatos esperados de request/response.
**Sección de Pruebas:** Indica cómo realizar verificaciones rápidas y dónde encontrar pruebas más formales.
**Secciones Opcionales:** Incluye placeholders para Licencia, Contribuciones, Agradecimientos.
**Formato Markdown:** Usa encabezados, listas, bloques de código y otros elementos para mejorar la legibilidad.