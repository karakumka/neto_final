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

Company: {company_name}
Ticker: {ticker}
Analysis period: last {period_days} days

Sentiment summary:
{json.dumps(summary, ensure_ascii=False, indent=2)}

Articles:
{json.dumps(articles, ensure_ascii=False, indent=2)}

Instructions:
- Write the report in Russian.
- Do not invent facts that are not present in the articles.
- Keep the report structured and readable.
- Base your conclusions only on the provided sentiment summary and article list.
- Mention the overall sentiment background.
- Mention positive signals if they are present.
- Mention negative or risky signals if they are present.
- If there are few articles, explicitly say that conclusions are preliminary.
- If many predictions have low confidence, say that interpretation should be cautious.

Publisher/source analysis:
- Pay attention to the publisher field for each article.
- If possible, assess whether the publisher appears to be:
  1) a large professional/business media outlet,
  2) a national or widely recognized media outlet,
  3) a smaller or niche source,
  4) an unclear/unknown source.
- Do not invent facts about publishers.
- If the importance of a publisher cannot be determined from its name, say that the source importance is unclear.
- Give more analytical weight to articles from large, professional, or widely recognized sources.
- Treat articles from unclear or small sources more cautiously.
- If a negative article comes from a large or professional source, mention it as a more important reputational risk.
- If a positive article comes from a large or professional source, mention it as a stronger positive signal.

Report structure:
1. Общая оценка информационного фона
2. Характер источников
3. Позитивные сигналы
4. Негативные сигналы и риски
5. Итоговый вывод
"""

    return ask_ollama(prompt=prompt, temperature=0.1)