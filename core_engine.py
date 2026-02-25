
#version2
import os

import pandas as pd

import numpy as np

import faiss

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

import tiktoken

from openai import OpenAI



nltk.download("punkt", quiet=True)



EMBED_MODEL = "text-embedding-3-small"

CHAT_MODEL = "gpt-4o-mini"



class RAGEngine:



    def __init__(self):

        self.client = OpenAI()

        self.index = None

        self.chunk_df = None

        self.encoder = tiktoken.get_encoding("cl100k_base")



    def token_count(self, text):

        return len(self.encoder.encode(text))



    def build_from_text(self, text, max_tokens_per_chunk=700):



        sentences = nltk.sent_tokenize(text)



        chunks = []

        current_chunk = ""



        for sentence in sentences:

            if self.token_count(current_chunk + sentence) < max_tokens_per_chunk:

                current_chunk += " " + sentence

            else:

                chunks.append(current_chunk.strip())

                current_chunk = sentence



        if current_chunk:

            chunks.append(current_chunk.strip())



        self.chunk_df = pd.DataFrame({"text": chunks})



        embeddings = []

        for chunk in chunks:

            response = self.client.embeddings.create(

                model=EMBED_MODEL,

                input=chunk

            )

            embeddings.append(response.data[0].embedding)



        embedding_array = np.array(embeddings).astype("float32")



        dimension = embedding_array.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embedding_array)



    def ask(self, question, k=3):



        if self.index is None:

            raise ValueError("Index not built yet.")



        query_embedding = self.client.embeddings.create(

            model=EMBED_MODEL,

            input=question

        ).data[0].embedding



        query_vector = np.array([query_embedding]).astype("float32")



        distances, indices = self.index.search(query_vector, k)



        context = "\n\n".join(

            self.chunk_df.iloc[i]["text"] for i in indices[0]

        )



        prompt = f"""

        Use the context below to answer the question.



        Context:

        {context}



        Question:

        {question}

        """



        response = self.client.chat.completions.create(

            model=CHAT_MODEL,

            messages=[{"role": "user", "content": prompt}],

            temperature=0

        )



        return response.choices[0].message.content



