"""Retriever interface + a zero-dependency TF-IDF implementation.

TfidfRetriever uses scikit-learn (already a project dependency) and is fully
offline — no model download, no service. To upgrade to semantic retrieval later,
add an EmbeddingRetriever with the same `.retrieve()` signature; nothing else
in the pipeline changes (same swappable pattern as the LLM providers).
"""
from abc import ABC, abstractmethod


class Retriever(ABC):
    name = "base"

    @abstractmethod
    def retrieve(self, query, k=4):
        """Return up to k [{text, source, score}] chunks most relevant to query."""
        raise NotImplementedError


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        if chunks:
            self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])
        else:
            self.matrix = None

    def retrieve(self, query, k=4):
        if not self.chunks:
            return []
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = np.argsort(sims)[::-1][:k]
        return [
            {**self.chunks[i], "score": round(float(sims[i]), 3)}
            for i in top
            if sims[i] > 0
        ]