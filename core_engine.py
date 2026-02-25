
import numpy as np
import pandas as pd
import faiss
import tiktoken
import nltk
from openai import OpenAI
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import os



nltk.download("punkt", quiet=True)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

class RAGEngine:

    def __init__(self, api_key=None):
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Relies on env var being set now
            self.client = OpenAI()

        self.index = None
        self.chunk_df = None
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def token_count(self, text):
        return len(self.encoder.encode(text))

    def build_from_video(self, video_id, max_tokens_per_chunk=700):
        # Adaptive transcript retrieval based on available methods
        transcript = None
        try:
            # 1. Try modern list_transcripts (preferred)
            if hasattr(YouTubeTranscriptApi, 'list_transcripts'):
                print("Using list_transcripts...")
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(['en']).fetch()

            # 2. Try standard get_transcript
            elif hasattr(YouTubeTranscriptApi, 'get_transcript'):
                 print("Using get_transcript...")
                 transcript = YouTubeTranscriptApi.get_transcript(video_id)

            # 3. Fallback: Check for instance-based methods (fixes 'missing 1 positional argument' error)
            else:
                 print("Using instance-based fallback...")
                 # Instantiate the class
                 yt_instance = YouTubeTranscriptApi()

                 if hasattr(yt_instance, 'list'):
                     print("Using instance.list()...")
                     raw_result = yt_instance.list(video_id)
                     # Check if result is a list or an object
                     if hasattr(raw_result, 'find_transcript'):
                         transcript = raw_result.find_transcript(['en']).fetch()
                     else:
                         transcript = raw_result
                 elif hasattr(yt_instance, 'fetch'):
                     print("Using instance.fetch()...")
                     transcript = yt_instance.fetch(video_id)
                 else:
                     # Last ditch: try get_transcript on instance
                     transcript = yt_instance.get_transcript(video_id)

        except Exception as e:
            print(f"Error during transcript retrieval: {e}")
            raise e

        if not transcript:
            raise ValueError("Could not retrieve transcript.")

        sentences = []
        current = ""
        start_time = None
        end_time = None

        for item in transcript:
            # Transcript items are usually dicts: {'text': ..., 'start': ..., 'duration': ...}
            # But handle object access just in case
            if isinstance(item, dict):
                text = item.get('text', '').strip()
                start = item.get('start', 0.0)
                duration = item.get('duration', 0.0)
            else:
                # Fallback if it returns objects with attributes
                text = getattr(item, 'text', '').strip()
                start = getattr(item, 'start', 0.0)
                duration = getattr(item, 'duration', 0.0)

            end = start + duration

            if start_time is None:
                start_time = start

            current += " " + text
            end_time = end

            if text.endswith((".", "!", "?")):
                sentences.append({
                    "text": current.strip(),
                    "start_time": start_time,
                    "end_time": end_time
                })
                current = ""
                start_time = None

        # Add remaining text if any
        if current:
             sentences.append({
                    "text": current.strip(),
                    "start_time": start_time,
                    "end_time": end_time
             })

        sentence_df = pd.DataFrame(sentences)
        sentence_df["token_count"] = sentence_df["text"].apply(self.token_count)

        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_start = None
        chunk_end = None

        for _, row in sentence_df.iterrows():
            text = row["text"]
            tokens = row["token_count"]

            if current_chunk and (current_tokens + tokens > max_tokens_per_chunk):
                chunks.append({
                    "chunk_text": current_chunk.strip(),
                    "start_time": chunk_start,
                    "end_time": chunk_end
                })
                current_chunk = ""
                current_tokens = 0
                chunk_start = None
                chunk_end = None

            if chunk_start is None:
                chunk_start = row["start_time"]

            chunk_end = row["end_time"]
            current_chunk += " " + text
            current_tokens += tokens

        if current_chunk:
            chunks.append({
                "chunk_text": current_chunk.strip(),
                "start_time": chunk_start,
                "end_time": chunk_end
            })

        self.chunk_df = pd.DataFrame(chunks)

        response = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=self.chunk_df["chunk_text"].tolist()
        )

        embeddings = np.array([r.embedding for r in response.data]).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def ask(self, query, top_k=5):
        if self.index is None:
            raise ValueError("Index not built. Run build_from_video() first.")

        query_embedding = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=[query]
        ).data[0].embedding

        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = self.chunk_df.iloc[indices[0]]
        context = "\n\n".join(results["chunk_text"].tolist())

        prompt = f"""
Use ONLY the context below to answer.

Context:
{context}

Question:
{query}
"""

        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content