import os

from data_extractor import DataExtractor


USE_RAPIDAPI = True
QUERY = "bitcoin"
MAX_RESULTS = 100

LOCAL_DATASET = os.path.join("datasets", "Bitcoin_tweets_dataset_2.csv")
API_OUTPUT = os.path.join("datasets", "tweets_from_api.csv")
OUTPUT_DIR = "outputs"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    extractor = DataExtractor(source_file=LOCAL_DATASET)

    print("=== 1. CARGA DE DATOS ===")

    if USE_RAPIDAPI:
        try:
            df = extractor.load_data_api(
                query=QUERY,
                max_results=MAX_RESULTS,
                output_file=API_OUTPUT
            )
            print("Datos cargados desde RapidAPI correctamente.")
        except Exception as exc:
            print("No se pudo cargar desde RapidAPI.")
            print(f"Motivo: {exc}")
            print("Se usará el dataset local de Bitcoin como respaldo.")
            extractor = DataExtractor(source_file=LOCAL_DATASET)
            df = extractor.load_data()
    else:
        df = extractor.load_data()
        print("Datos cargados desde dataset local.")

    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print("Columnas disponibles:")
    print(df.columns.tolist())

    print("\n=== 2. ANÁLISIS DE HASHTAGS ===")
    hashtag_results = extractor.analytics_hashtags_extended()

    print("\nTop 10 hashtags globales:")
    print(hashtag_results["overall"].head(10))

    print("\nTop 10 hashtags por usuario:")
    print(hashtag_results["by_user"].head(10))

    print("\nTop 10 hashtags por fecha:")
    print(hashtag_results["by_date"].head(10))

    extractor.generate_hashtag_wordcloud(
        overall_df=hashtag_results["overall"],
        save_path=os.path.join(OUTPUT_DIR, "wordcloud_hashtags.png")
    )

    extractor.plot_top_hashtags(
        overall_df=hashtag_results["overall"],
        top_n=10,
        save_path=os.path.join(OUTPUT_DIR, "top_hashtags.png")
    )

    try:
        extractor.plot_hashtag_trend(
            hashtag="#bitcoin",
            by_date_df=hashtag_results["by_date"],
            save_path=os.path.join(OUTPUT_DIR, "trend_bitcoin.png")
        )
    except ValueError as exc:
        print(f"Aviso en tendencia de hashtag: {exc}")

    print("\n=== 3. GUARDADO DE RESULTADOS BÁSICOS ===")
    basic_paths = extractor.save_results(output_dir=OUTPUT_DIR)

    print("\nArchivos básicos guardados:")
    for name, path in basic_paths.items():
        print(f"- {name}: {path}")

    print("\n=== 4. MODELADO DE TÓPICOS LDA ===")
    topics = extractor.model_topics(num_topics=5, passes=3)

    for topic in topics:
        print(f"Tópico {topic['topic_id']}: {topic['words_joined']}")

    print("\n=== 5. ANÁLISIS DE SENTIMIENTO ===")
    sentiment_df = extractor.analyze_sentiment(method="textblob")

    print("\nDistribución de sentimiento:")
    print(sentiment_df["sentiment_label"].value_counts())

    print("\nEjemplos de sentimiento:")
    print(
        sentiment_df[
            ["clean_text", "sentiment_polarity", "sentiment_subjectivity", "sentiment_label"]
        ].head(10)
    )

    extractor.plot_sentiment_distribution(
        save_path=os.path.join(OUTPUT_DIR, "sentiment_distribution.png")
    )

    print("\n=== 6. PARSING CON SPACY ===")
    parsing_df = extractor.parse_sample_texts(sample_size=3)

    print("\nEjemplo de parsing:")
    print(parsing_df.head(20))

    print("\n=== 7. RESUMEN EXTRACTIVO ===")
    summary = extractor.parse_and_summarize(summary_ratio=0.1)

    print("\nResumen generado:")
    print(summary[:2000])

    print("\n=== 8. GUARDADO DE RESULTADOS AVANZADOS ===")
    advanced_paths = extractor.save_advanced_results(
        topics=topics,
        parsing_df=parsing_df,
        summary=summary,
        output_dir=OUTPUT_DIR
    )

    print("\nArchivos avanzados guardados:")
    for name, path in advanced_paths.items():
        print(f"- {name}: {path}")

    print("\nProceso completado correctamente.")