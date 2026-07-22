# src/pipeline.py

from src.request_parser import parse_brand_health_request
from src.api_client import fetch_articles_for_ticker
from src.preprocessing import preprocess_articles
from src.inference import predict_sentiment
from src.aggregation import add_confidence, get_sentiment_summary, prepare_articles_for_llm
from src.report import generate_brand_health_report


def prepare_articles_dataframe(result_df):

    article_columns = ["published_utc", "publisher", "ticker", "company_name", 
                   "title", "description", "predicted_sentiment", "confidence"]
    existing_columns = [col for col in article_columns if col in result_df.columns]

    return result_df[existing_columns].copy()


def analyze_brand_health(user_query: str) -> dict:
    parsed_request = parse_brand_health_request(user_query)

    ticker = parsed_request["ticker"]
    company_name = parsed_request["company_name"]
    period_days = parsed_request["period_days"]

    raw_df = fetch_articles_for_ticker(ticker=ticker, period=period_days)
    preprocessed_df = preprocess_articles(raw_df)

    result_df = predict_sentiment(preprocessed_df)
    result_df = add_confidence(result_df)

    summary = get_sentiment_summary(result_df)
    articles = prepare_articles_for_llm(result_df)
    articles_df = prepare_articles_dataframe(result_df)

    report = generate_brand_health_report(company_name=company_name, ticker=ticker, period_days=period_days,
                                          summary=summary, articles=articles)

    return {
        "parsed_request": parsed_request,
        "summary": summary,
        "articles_df": articles_df,
        "result_df": result_df,
        "report": report,
    }