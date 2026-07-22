import os
import requests
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

BASE_URL = "https://api.massive.com/v2/reference/news"


def fetch_articles_for_ticker(ticker: str, period: int = 7, limit: int = 1000, 
                                  api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Функция выгружает новости по тикеру из Massive API.

    Parameters
    ----------
    ticker:
        Тикер, например "AAPL", "MSFT".
    period:
        Период, за который выгружаем новости. Высчитывается от текущей даты.
    limit:
        Максимальное число новостей на один тикер.
    api_key:
        API key. Если не передан, берётся из переменной окружения MASSIVE_API_KEY.

    Returns
    -------
    pd.DataFrame
        Таблица с колонками: 
        published_utc, title, description, publisher, ticker

    Raises
    -------
    ValueError: 
        Если не задан MASSIVE_API_KEY. Если запрашиваемый период меньше 1.
    RuntimeError:
        Если не удалось получить данные или разобрать данные из Massive API.
    """

    api_key = os.getenv("MASSIVE_API_KEY")

    if api_key is None:
        api_key = os.getenv("MASSIVE_API_KEY")

    if not api_key:
        raise ValueError("MASSIVE_API_KEY environment variable is not set.")

    all_rows = []

    if period < 1:
        raise ValueError("Period must be a positive integer.")

    start_date = datetime.now(timezone.utc) - timedelta(days=period)
    start_date_iso = start_date.date().isoformat()

    params = {"ticker": ticker, "limit": limit, "apiKey": api_key, 
              "sort": "published_utc", "order": "desc", "published_utc.gte": start_date_iso}

    try:
        response = requests.get(url=BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        articles = data.get("results", [])

    except Exception as e:
        raise RuntimeError(f"Failed to fetch articles for ticker {ticker}: {e}.")

    for article in articles:
        published_utc = article.get("published_utc")
        title = article.get("title")
        description = article.get("description")

        publisher = article.get("publisher") or {}
        publisher_name = publisher.get("name")

        insights = article.get("insights") or []

        for insight in insights:
            insight_ticker = insight.get("ticker")

            if insight_ticker == ticker:
                all_rows.append(
                    {
                        "published_utc": published_utc,
                        "title": title,
                        "description": description,
                        "publisher": publisher_name,
                        "ticker": insight_ticker
                    }
                )

    return pd.DataFrame(all_rows)