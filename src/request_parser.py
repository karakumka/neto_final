import re
import json
from typing import Any, Dict
from src.llm_client import ask_ollama


TICKER_TO_COMPANY = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "META": "Meta", "AMZN": "Amazon", "NVDA": "Nvidia", "NFLX": "Netflix", 
                 "ADBE": "Adobe", "INTC": "Intel", "AMD": "AMD", "IBM": "IBM", "CSCO": "Cisco"}
COMPANY_TO_TICKER = {company.lower(): ticker for ticker, company in TICKER_TO_COMPANY.items()}

def resolve_ticker(company_name: str) -> str:
    key = company_name.strip().lower()

    if key not in COMPANY_TO_TICKER:
        supported = ", ".join(TICKER_TO_COMPANY.values())
        raise ValueError(f"Unsupported company: {company_name}. "
                         f"Supported companies: {supported}.")

    return COMPANY_TO_TICKER[key]


def resolve_company_name(ticker: str) -> str:
    if not ticker:
        raise ValueError("Ticker is empty.")

    ticker = ticker.strip().upper()

    if ticker not in TICKER_TO_COMPANY:
        raise ValueError(f"Unsupported ticker: {ticker}.")

    return TICKER_TO_COMPANY[ticker]


def extract_json_from_text(text: str) -> dict:
    """
    Функция извлекает JSON-объект из ответа LLM.
    """

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text}")

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse JSON from LLM response: {json_text}") from error


def parse_brand_health_request(user_query: str) -> Dict[str, Any]:
    """
    Функция парсит запрос пользователя на естественном языке и превращает его в структурированные параметры.

    LLM извлекает:
    - company_name
    - period_days
    """

    prompt = f"""
You are a request parser for a brand health analytics system.

Your task is to extract the company name and analysis period from the user's request.

You must normalize the company name to exactly one of the supported company names:
Apple, Microsoft, Google, Meta, Amazon, Nvidia, Netflix, Adobe, Intel, AMD, IBM, Cisco.

Examples:
- "эппл", "Apple Inc", "AAPL" -> Apple
- "майкрософт", "MSFT" -> Microsoft
- "гугл", "Alphabet", "GOOGL" -> Google
- "нвидиа", "NVDA" -> Nvidia

If the company cannot be mapped to one of the supported companies, return null.

Return ONLY valid JSON. Do not add explanations, markdown, or comments.

Example of JSON schema:
{{
  "company_name": "Apple",
  "period_days": 7
}}

Rules:
- Extract the company mentioned by the user.
- company_name must be one of the supported companies.
- If the user writes a ticker, map it to the corresponding company name.
- If the user says "today", or "за сегодня", use 1.
- If the user says "last week", "past week", or "за последнюю неделю", use 7.
- If the user says "two weeks" or "две недели", use 14.
- If the user says "last month", "past month", or "за месяц", use 30.
- If the user mentions a specific number of days, use that number.
- If no period is mentioned, use 7.
- If the company is not supported, return company_name as null.
- period_days must be an integer.

User request:
{user_query}
"""

    raw_response = ask_ollama(prompt=prompt, temperature=0.0)
    parsed = extract_json_from_text(raw_response)

    company_name = parsed.get("company_name")
    period_days = parsed.get("period_days", 7)

    if company_name is None:
        raise ValueError("Could not identify a supported company in the request.")

    try:
        period_days = int(period_days)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid period_days value: {period_days}") from error

    if period_days < 1:
        raise ValueError("period_days must be greater than or equal to 1.")

    ticker = resolve_ticker(company_name)
    normalized_company_name = resolve_company_name(ticker)

    return {
        "task": "brand_health_analysis",
        "company_name": normalized_company_name,
        "ticker": ticker,
        "period_days": period_days,
        "original_query": user_query,
    }
