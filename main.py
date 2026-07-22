from src.request_parser import parse_brand_health_request
from src.api_client import fetch_articles_for_ticker
from src.preprocessing import preprocess_articles
from src.inference import predict_sentiment
from src.aggregation import add_confidence, get_sentiment_summary

user_query = "Проанализируй здоровье бренда нвидиа за последние три дня"

parsed_request = parse_brand_health_request(user_query)

raw_df = fetch_articles_for_ticker(ticker=parsed_request["ticker"], period=parsed_request["period_days"])
preprocessed_df = preprocess_articles(raw_df)
result_df = predict_sentiment(preprocessed_df)
result_df = add_confidence(result_df)
summary = get_sentiment_summary(result_df)

print(result_df.head())

print(summary)