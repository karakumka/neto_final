import pandas as pd

COMPANY_NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "META": "Meta", "AMZN": "Amazon", "NVDA": "Nvidia", "NFLX": "Netflix", 
                 "ADBE": "Adobe", "INTC": "Intel", "AMD": "AMD", "IBM": "IBM", "CSCO": "Cisco"}


def preprocess_articles(data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция приводит загруженные новости к формату, который ожидает классификатор.

    Parameters
    ----------
    data : pd.DataFrame
        Сырые данные из Massive API.

    Returns
    -------
    pd.DataFrame
        Данные с добавленными колонками company_name и text.

    Raises
    -------
    ValueError:
        Если в датафрейме отсутствуют необходимые колонки.
    """

    required_columns = ["ticker", "title", "description"]

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data_preprocessed = data.copy()

    data_preprocessed["company_name"] = data_preprocessed["ticker"].map(COMPANY_NAMES)

    data_preprocessed["title"] = data_preprocessed["title"].fillna("")
    data_preprocessed["description"] = data_preprocessed["description"].fillna("")
    data_preprocessed = data_preprocessed[
        (data_preprocessed["title"] != "") | (data_preprocessed["description"] != "")
    ]

    data_preprocessed["text"] = (data_preprocessed["title"] + ". " + data_preprocessed["description"])
    data_preprocessed = data_preprocessed.dropna(subset=["company_name", "text"])

    data_preprocessed = data_preprocessed.drop_duplicates(
        subset=["published_utc", "title", "description", "ticker"],
        keep="first"
    )

    return data_preprocessed.reset_index(drop=True)