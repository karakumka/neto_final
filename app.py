import pandas as pd
import streamlit as st
import plotly.express as px
from src.pipeline import analyze_brand_health


st.set_page_config(page_title="Brand Health Monitor", layout="wide")

st.title("BRAND HEALTH MONITOR")

st.markdown(
    """
    Это приложение анализирует здоровье бренда на основании новостных статей.\n
    Для демонстрации решения введите запрос на русском языке.
    """
)


user_query = st.text_input("ВВЕДИТЕ ЗАПРОС:", placeholder="Пример: 'Проанализируй бренд Nvidia за последние три дня'")

run_button = st.button("АНАЛИЗ")


if run_button:
    if not user_query.strip():
        st.warning("Введите запрос для анализа.")
    else:
        try:
            with st.spinner("Запускаю анализ здоровья бренда..."):
                result = analyze_brand_health(user_query)

            parsed_request = result["parsed_request"]
            summary = result["summary"]
            articles_df = result["articles_df"]
            report = result["report"]

            st.subheader("1. Распознанные параметры запроса:")

            col1, col2, col3 = st.columns(3)

            col1.metric("Company", parsed_request["company_name"])
            col2.metric("Ticker", parsed_request["ticker"])
            col3.metric("Period (days)", parsed_request["period_days"])

            st.divider()

            st.subheader("2. Визуализация новостного фона:")

            total_articles = summary.get("total_articles", 0)
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total articles", total_articles)
            col2.metric("Negative", summary.get("negative_count", 0))
            col3.metric("Neutral", summary.get("neutral_count", 0))
            col4.metric("Positive", summary.get("positive_count", 0))

            SENTIMENT_COLORS = {"positive": "aquamarine", "neutral": "slategrey", "negative": "firebrick"}

            sentiment_chart_df = pd.DataFrame(
                {"sentiment": ["negative", "neutral", "positive"],
                    "count": [
                        summary.get("negative_count", 0),
                        summary.get("neutral_count", 0),
                        summary.get("positive_count", 0),
                    ]}
            )

            sentiment_chart_df = sentiment_chart_df[sentiment_chart_df["count"] > 0]

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("Распределение тональности")

                if sentiment_chart_df.empty:
                    st.info("Нет данных для построения графика тональности")
                else:
                    fig_pie = px.pie(sentiment_chart_df, names="sentiment", values="count", hole=0.25,
                                     color="sentiment", color_discrete_map=SENTIMENT_COLORS)
                    fig_pie.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent:.1%}")
                    fig_pie.update_layout(showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)


            with chart_col2:
                st.markdown("Количество новостей по дням")

                if articles_df.empty or "published_utc" not in articles_df.columns:
                    st.info("Нет данных о датах публикаций для построения графика")
                else:
                    daily_news_df = articles_df.copy()

                    daily_news_df["published_date"] = pd.to_datetime(daily_news_df["published_utc"], errors="coerce", utc=True).dt.date

                    daily_news_df = (
                        daily_news_df
                        .dropna(subset=["published_date"])
                        .groupby(["published_date", "predicted_sentiment"])
                        .size()
                        .reset_index(name="count")
                        .sort_values("published_date")
                    )

                    fig_bar = px.bar(daily_news_df, x="published_date", y="count", 
                                     color="predicted_sentiment", color_discrete_map=SENTIMENT_COLORS,
                                     labels={
                                        "published_date": "Дата публикации",
                                        "count": "Количество статей",
                                        "predicted_sentiment": "Тональность"
                                    })

                    fig_bar.update_layout(margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Дата", yaxis_title="Количество статей")

                    st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()

            st.subheader("3. Статьи, использованные для анализа:")

            if articles_df.empty:
                st.info("По текущему запросу не найдено ни одной статьи")
            else:
                st.dataframe(articles_df, use_container_width=True, hide_index=True)

            st.divider()

            st.subheader("4. Отчет:")

            st.write(report)

        except Exception as error:
            st.error("Ошибка анализа")
            st.exception(error)