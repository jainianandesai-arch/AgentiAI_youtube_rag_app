import streamlit as st

import pandas as pd

from core_engine import RAGEngine



st.set_page_config(page_title="Enterprise Document Intelligence", layout="wide")



st.title("🏢 Enterprise Document Intelligence")



# -----------------------------

# SESSION STATE ENGINE

# -----------------------------

if "engine" not in st.session_state:

    st.session_state.engine = RAGEngine()



engine = st.session_state.engine





# -----------------------------

# Upload Section

# -----------------------------

uploaded_file = st.file_uploader(

    "Upload document",

    type=["txt", "csv"]

)



if uploaded_file is not None:



    if uploaded_file.type == "text/plain":

        text = uploaded_file.read().decode("utf-8")



    else:

        df = pd.read_csv(uploaded_file)

        text = "\n".join(df.astype(str).apply(" ".join, axis=1))



    if st.button("Build Knowledge Base"):

        with st.spinner("Building knowledge base..."):

            engine.build_from_text(text)

        st.success("Knowledge Base Ready")





# -----------------------------

# Question Section

# -----------------------------

st.subheader("Ask a business question")



question = st.text_input("Enter your question")



if st.button("Analyze"):



    if engine.embeddings is None:

        st.error("Please build the knowledge base first.")

    else:

        with st.spinner("Analyzing..."):

            answer = engine.ask(question)

        st.success("Answer:")

        st.write(answer)

