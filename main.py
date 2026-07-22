from src.api_client import fetch_articles_for_ticker
from src.preprocessing import preprocess_articles
from src.inference import predict_sentiment
from src.aggregation import add_confidence, get_sentiment_summary

raw_df = fetch_articles_for_ticker("AAPL", period=7)
preprocessed_df = preprocess_articles(raw_df)
result_df = predict_sentiment(preprocessed_df)
result_df = add_confidence(result_df)
summary = get_sentiment_summary(result_df)


print(result_df[[
    "ticker",
    "company_name",
    "predicted_sentiment",
    "confidence",
    "proba_negative",
    "proba_neutral",
    "proba_positive",
]].head())

print(summary)