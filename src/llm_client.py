import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:14b"


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """
    Функция посылает запрос к модели Qwen/Qwen3-14B, установленной локально через Ollama.
    """

    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"Failed to get response from Ollama: {error}")

    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response format: {data}")

    return data["response"].strip()