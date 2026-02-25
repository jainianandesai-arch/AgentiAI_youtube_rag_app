import streamlit as st

from core_engine import RAGEngine



st.set_page_config(page_title="YouTube RAG", layout="wide")

st.title("🎥 YouTube RAG Q&A")



# Initialize engine only once

if "engine" not in st.session_state:

    st.session_state.engine = RAGEngine()

    st.session_state.index_built = False



video_input = st.text_input("Enter YouTube Video ID")



if st.button("Build Index"):

    if video_input:

        with st.spinner("Building index..."):

            st.session_state.engine.build_from_video(video_input)



        st.session_state.index_built = True

        st.success("Index built successfully!")



question = st.text_input("Ask a question")



if st.button("Ask"):

    if not st.session_state.index_built:

        st.error("Please build the index first.")

    else:

        with st.spinner("Thinking..."):

            answer = st.session_state.engine.ask(question)

        st.markdown("### Answer")

        st.write(answer)

