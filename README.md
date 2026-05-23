# Práctica U6 — Análisis de redes con NetworkX y LLM

Evolutivo de la **Unidad 4** (tweets sobre **Bitcoin**): grafo de menciones `@usuario`, métricas de red, prompt interpretativo y chat local con **Gemma**.

Pipeline (`execute_pipeline()`):

```
load_data → clean_data → build_interaction_graph → analyze_network
         → generate_prompt_from_network → chat_local_llm
```

La lógica vive en `DataExtractor` (`extractor/core.py`), organizada en mixins. `data_extractor.py` reexporta la clase por compatibilidad.

## Instalación

Requisitos: **Python 3.11** o **3.12** (en Windows: `py -3.11`).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
py -3.11 -m pip install -r requirements.txt
```

Opcional (funciones NLP de la U4):

```powershell
py -3.11 -m spacy download en_core_web_sm
```

El modelo `google/gemma-4-E2B-it` se descarga solo la primera vez (~10 GB). Suele bastar con instalar dependencias; si Hugging Face pide acceso, acepta la licencia en [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) y ejecuta `huggingface-cli login`.

## Configuración

Copia `.env.example` a `.env` para RapidAPI:

```env
RAPIDAPI_KEY=tu_clave
RAPIDAPI_HOST=twitter-api45.p.rapidapi.com
```

Clave en [Twitter API45 (RapidAPI)](https://rapidapi.com/alexanderxbx/api/twitter-api45). Si la API falla, `main.py` usa `datasets/Bitcoin_tweets_dataset_2.csv`.

## Ejecución

```powershell
py -3.11 main.py
```

1. Carga tweets (`bitcoin` vía API o CSV local).
2. Ejecuta los 6 pasos del pipeline.
3. Abre chat con Gemma (escribe `salir` para terminar).

Prueba rápida sin LLM:

```python
from data_extractor import DataExtractor

extractor = DataExtractor(source_file="datasets/tweets_from_api.csv")
extractor.load_data()
extractor.execute_pipeline(output_dir="outputs", run_llm=False)
```

## Análisis de red

- **Grafo:** `nx.DiGraph`; nodos = `user_name`, aristas = menciones en `text` (autor → mencionado, sin duplicados).
- **Métricas:** centralidad de grado y betweenness, densidad, componentes débiles, comunidades Louvain.
- **Figura:** `outputs/network_graph.png` (color = comunidad, tamaño = centralidad de grado).

## LLM

- **Prompt** (`generate_prompt_from_network`): top 3 por centralidad de grado, hashtag dominante, densidad, comunidades y tareas de interpretación → `outputs/network_prompt.txt`.
- **Chat** (`chat_local_llm`): carga Gemma en CUDA/CPU, responde al prompt y permite seguir conversando. Guarda `outputs/llm_chat.txt`.
- Durante la generación se muestra una barra de progreso por tokens y la respuesta aparece en streaming.
- Por defecto `max_new_tokens=256` (respuestas más cortas; útil con ~8 GB VRAM).

El modelo recibe el resumen de la red, no el CSV completo.

## Salidas

| Archivo | Contenido |
|---------|-----------|
| `outputs/network_graph.png` | Grafo de menciones |
| `outputs/network_prompt.txt` | Prompt para el LLM |
| `outputs/llm_chat.txt` | Conversación del chat |

![Grafo de interacciones](outputs/network_graph.png)

## Estructura

```
UD6/
├── main.py
├── data_extractor.py
├── rapidapi_client.py
├── extractor/          # mixins: io, text_analysis, nlp, export, network, llm
├── datasets/
├── outputs/
└── requirements.txt
```

## Hardware

Grafo y prompt corren en CPU. Para Gemma conviene **GPU NVIDIA con 6–8 GB VRAM** y **16 GB RAM**; en CPU la inferencia es muy lenta pero funcional.

## Unidad 4

LDA, sentimiento, spaCy, wordcloud y exportación siguen en los mixins; no se ejecutan desde `main.py` actual.
