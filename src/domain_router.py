from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class DomainRouter:
    """Embedding-based routing: map user query → domain key."""

    def __init__(self, domains: dict, embedding_model: SentenceTransformer) -> None:
        self.domains = domains
        self.model = embedding_model  # Dùng chung model với RAG, không load thêm

        # Pre-encode tất cả domain descriptions
        keys = list(domains.keys())
        descs = [domains[k]["desc"] for k in keys]

        self.domain_keys = keys
        self.domain_embeddings = self.model.encode(
            descs,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )  # shape: (n_domains, dim)

    def route(self, query: str, top_k: int = 2) -> List[str]:
        """
        Trả về list domain_key được rank theo relevance.
        top_k=2 để fallback sang domain liên quan thứ 2 nếu cần.
        """
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )  # shape: (1, dim)

        # Cosine similarity (đã normalize nên chỉ cần dot product)
        scores = (query_emb @ self.domain_embeddings.T).squeeze(0)  # shape: (n_domains,)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [self.domain_keys[i] for i in ranked_indices]