
import streamlit as st

from src.genai.rag_assistant import ask_finsight


st.set_page_config(
    page_title="FinSight AI Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 FinSight AI Financial Assistant")

st.caption(
    "Ask questions about FinSight AI, financial analytics, "
    "anomaly detection, and explainable machine learning."
)

st.info(
    "This assistant uses a local Llama 3.2 model and "
    "retrieves information from FinSight AI documentation. "
    "It does not provide financial advice."
)

if "finsight_messages" not in st.session_state:
    st.session_state.finsight_messages = []

for message in st.session_state.finsight_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            st.caption(
                "Sources: " + ", ".join(message["sources"])
            )

question = st.chat_input(
    "Ask FinSight AI a question..."
)

if question:
    st.session_state.finsight_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                result = ask_finsight(question)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    st.caption(
                        "Sources: " + ", ".join(sources)
                    )

            except Exception as error:
                answer = (
                    "Unable to generate a response. "
                    "Please check that Ollama is running "
                    "and the knowledge base is indexed."
                )
                sources = []

                st.error(answer)
                st.caption(f"Technical details: {error}")

    st.session_state.finsight_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
