from src.pipeline import analyze_brand_health

result = analyze_brand_health("Проанализируй здоровье бренда Apple за последнюю неделю")

print(result["parsed_request"])
print(result["summary"])
print(result["articles_df"])
print(result["report"])