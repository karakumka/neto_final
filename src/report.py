import json
from typing import Any, Dict, List
from src.llm_client import ask_ollama


def generate_brand_health_report(company_name: str, ticker: str, period_days: int, summary: Dict[str, Any], articles: List[Dict[str, Any]]) -> str:
    """
    Функция генерирует аналитический отчет о здоровье бренда на основании публикаций в медиа, используя LLM.
    """

    prompt = f"""
You are a brand health analyst.

Your task is to write a concise analytical report about the brand's media health based on news sentiment classification results.
You must write a report ONLY based on the provided articles.
Do not use external knowledge.
Do not invent articles, publishers, dates, facts, or events.
If an article is not present in the provided JSON list, you must not mention it.

Company: {company_name}
Ticker: {ticker}
Analysis period: last {period_days} days

Sentiment summary:
{json.dumps(summary, ensure_ascii=False, indent=2)}

Articles:
{json.dumps(articles, ensure_ascii=False, indent=2)}

Strict rules:
- Write the report in Russian.
- Do not invent facts that are not present in the articles.
- Keep the report structured and readable.
- Base your conclusions only on the provided sentiment summary and article list.
- Mention the overall sentiment background.
- Mention positive signals if they are present.
- Mention negative or risky signals if they are present.
- If there are few articles, explicitly say that conclusions are preliminary.
- The report must be about {company_name} only.
- Mention article titles only when they are important for explaining the conclusion.
- Do not include raw confidence values with many decimals.
- Do not list articles as database records.
- Do not output fields like "Название статьи", "Описание", "Прогнозируемое суждение", "Доверие".
- Do not repeat the article JSON structure.
- Use articles only as evidence for a concise analytical summary.
- Do not provide investment advice.
- Do not recommend buying, selling, or holding stocks.
- Do not make stock price forecasts.
- Do not discuss whether the user should invest.
- The report must focus only on brand health and media sentiment.
- If financial or investment implications are mentioned, state that they are outside the scope of this report.

Language rules:
- Use grammatically correct Russian.
- If period_days = 1, write "за последний день".
- If period_days is greater than 1, write "за последние {period_days} дней".
- Use "одна статья", "две статьи", "три статьи", "пять статей" correctly.
- Avoid awkward phrases like "в течение последних 1 дня" or "два статьи".

Content requirements:
- Always describe neutral articles if neutral_count > 0.
- Always describe positive articles if positive_count > 0.
- Always describe negative articles if negative_count > 0.
- If a class has zero articles, write briefly that such signals were not found.
- In the final conclusion, summarize the brand media background, not investment attractiveness.
- Add a final disclaimer: "Данный анализ отражает только информационный фон вокруг бренда и не является инвестиционной рекомендацией."

Relevance rules:
- The report is about the target company only.
- If an article does not explicitly mention the target company in title or description, do not present it as a direct brand signal.
- Articles with article_relevance = "industry_context_or_unclear" may only be described as broader industry context or unclear relevance.
- Do not claim that such articles directly affect the brand unless the connection is explicit in the provided text.
- If most articles are indirect or unclear, say that the data does not provide enough direct evidence about the brand's media health.

Report structure:
1. Общая оценка
2. Что формирует фон
3. Риски и ограничения
4. Вывод

"""

    return ask_ollama(prompt=prompt, temperature=0.0)