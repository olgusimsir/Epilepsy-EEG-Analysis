"""Local, offline retrieval-augmented generation (RAG) for EEG report grounding.

Loads a local corpus of epilepsy/EEG reference documents, retrieves the passages
most relevant to a set of findings, and supplies them as context to the report LLM.
Fully offline. The retriever is swappable (TF-IDF today, embeddings later).
"""