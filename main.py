import os
import sys

from data_extractor import DataExtractor

USE_RAPIDAPI = True
QUERY = "bitcoin"
SEARCH_TYPE = "Top"
MAX_RESULTS = 100

LOCAL_DATASET = os.path.join("datasets", "Bitcoin_tweets_dataset_2.csv")
API_OUTPUT = os.path.join("datasets", "tweets_from_api.csv")
OUTPUT_DIR = "outputs"


def load_dataset(extractor: DataExtractor) -> DataExtractor:
    """Carga tweets desde RapidAPI o, si falla, desde el CSV local."""
    print("=== Carga de datos (RapidAPI / respaldo local) ===")

    if USE_RAPIDAPI:
        try:
            extractor.load_data_api(
                query=QUERY,
                max_results=MAX_RESULTS,
                output_file=API_OUTPUT,
                search_type=SEARCH_TYPE,
            )
            print("Datos cargados desde RapidAPI.")
            return extractor
        except Exception as exc:
            print(f"No se pudo usar RapidAPI: {exc}")
            print("Se usará el dataset local de Bitcoin.")

    fallback = DataExtractor(source_file=LOCAL_DATASET)
    fallback.load_data()
    print("Datos cargados desde dataset local.")
    return fallback


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extractor = DataExtractor(source_file=API_OUTPUT if USE_RAPIDAPI else LOCAL_DATASET)
    extractor = load_dataset(extractor)

    try:
        extractor.execute_pipeline(output_dir=OUTPUT_DIR, run_llm=True)
    except Exception as exc:
        print(f"\nEl pipeline se detuvo: {exc}", file=sys.stderr)
        sys.exit(1)
