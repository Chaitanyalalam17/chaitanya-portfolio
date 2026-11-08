
from ollama import chat


MODEL_NAME = "llama3.2:3b"


def ask_ollama(question: str) -> str:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are FinSight AI, a financial data "
                    "analytics assistant. Explain financial "
                    "concepts clearly. Do not invent transaction "
                    "data or claim that anomalies prove fraud."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.message.content or ""


if __name__ == "__main__":
    question = "Explain financial anomaly detection in simple terms."

    print("FinSight AI Assistant")
    print("-" * 40)
    print(ask_ollama(question))
