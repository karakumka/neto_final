from src.api_client import fetch_articles_for_ticker
from src.preprocessing import preprocess_articles

raw_df = fetch_articles_for_ticker("AAPL", period=7)
preprocessed_df = preprocess_articles(raw_df)

print("Raw shape:", raw_df.shape)
print(raw_df.head())

print("Prepared shape:", preprocessed_df.shape)
print(preprocessed_df.head())