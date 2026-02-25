import streamlit as st

from core_engine import RAGEngine

import pandas as pd



st.set_page_config(page_title="Enterprise RAG", layout="wide")



st.title("🏢 Enterprise Document Intelligence")



engine = RAGEngine()



uploaded_file = st.file_uploader("Upload document", type=["txt", "csv"])



if uploaded_file:



    if uploaded_file.type == "text/plain":

        text = uploaded_file.read().decode("utf-8")



    else:

        df = pd.read_csv(uploaded_file)

        text = "\n".join(df.astype(str).apply(" ".join, axis=1))



    if st.button("Build Knowledge Base"):

        engine.build_from_text(text)

        st.success("Knowledge Base Ready")



question = st.text_input("Ask a business question")



if st.button("Analyze"):

    answer = engine.ask(question)

    st.markdown("### Insight")

    st.write(answer)

