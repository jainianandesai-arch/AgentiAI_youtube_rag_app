import os

import numpy as np

import pandas as pd

from openai import OpenAI

from sklearn.metrics.pairwise import cosine_similarity





class RAGEngine:

    def __init__(self):

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        self.embeddings = None

        self.text_chunks = None



    # -----------------------------

    # Build index from raw text

    # -----------------------------

    def build_from_text(self, text, chunk_size=500):



        # Simple chunking (NO NLTK – avoids deployment errors)

        chunks = [

            text[i : i + chunk_size]

            for i in range(0, len(text), chunk_size)

        ]



        self.text_chunks = chunks



        embeddings = []



        for chunk in chunks:

            response = self.client.embeddings.create(

                model="text-embedding-3-small",

                input=chunk

            )

            embeddings.append(response.data[0].embedding)



        self.embeddings = np.array(embeddings)



    # -----------------------------

    # Ask question

    # -----------------------------

    def ask(self, question):



        if self.embeddings is None:

            raise ValueError("Index not built yet.")



        # Embed question

        response = self.client.embeddings.create(

            model="text-embedding-3-small",

            input=question

        )



        question_embedding = np.array(response.data[0].embedding).reshape(1, -1)



        # Similarity search

        similarities = cosine_similarity(question_embedding, self.embeddings)

        best_index = np.argmax(similarities)



        context = self.text_chunks[best_index]



        # Generate answer

        completion = self.client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[

                {"role": "system", "content": "Answer based only on provided context."},

                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}

            ]

        )



        return completion.choices[0].message.content

