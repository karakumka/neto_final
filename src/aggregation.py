import pandas as pd


def add_confidence(data: pd.DataFrame) -> pd.DataFrame:

    result = data.copy()
    result["confidence"] = result[["proba_negative", "proba_neutral", "proba_positive"]].max(axis=1)

    return result


def get_sentiment_summary(data: pd.DataFrame) -> dict:
    total_articles = len(data)

    counts = (
        data["predicted_sentiment"]
        .value_counts()
        .reindex(["negative", "neutral", "positive"], fill_value=0)
    )

    shares = (counts / total_articles).round(3) if total_articles > 0 else counts

    return {
        "total_articles": total_articles,
        "negative_count": int(counts["negative"]),
        "neutral_count": int(counts["neutral"]),
        "positive_count": int(counts["positive"]),
        "negative_share": float(shares["negative"]),
        "neutral_share": float(shares["neutral"]),
        "positive_share": float(shares["positive"]),
    }

def prepare_articles_for_llm(data: pd.DataFrame) -> list[dict]:

    columns = ["published_utc", "ticker", "company_name", "title", "description", "predicted_sentiment", "confidence"]
    existing_columns = [col for col in columns if col in data.columns]

    return data[existing_columns].to_dict(orient="records")