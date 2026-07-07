"""Build a retriever from the local knowledge corpus, and derive a retrieval query
from a structured EEG assessment."""
from src.rag.documents import load_corpus, KNOWLEDGE_DIR
from src.rag.retriever import TfidfRetriever


def build_retriever(path=KNOWLEDGE_DIR):
    """Load + chunk the corpus and return a ready-to-query retriever."""
    return TfidfRetriever(load_corpus(path))


def query_from_assessment(assessment):
    """Turn the structured findings into a natural-language retrieval query."""
    a = assessment
    return (
        f"EEG seizure detection clinical report. Epilepsy risk level {a['risk_level']}, "
        f"{a['n_episodes']} seizure episode(s), {a['n_abnormal_windows']} abnormal epochs, "
        f"peak detection confidence {a['peak_confidence']}. "
        f"EEG interpretation, reporting structure, and recommendation guidance."
    )


if __name__ == "__main__":
    r = build_retriever()
    print(f"Corpus chunks: {len(r.chunks)}")
    demo_query = "high risk seizure episode confidence reporting recommendation"
    for hit in r.retrieve(demo_query, k=3):
        print(f"\n[{hit['source']}  score={hit['score']}]\n{hit['text'][:160]}...")