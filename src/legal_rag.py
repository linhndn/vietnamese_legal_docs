import preprocess
from domain_router import DomainRouter

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
import json
import chromadb
from groq import Groq
import re
import os
import datetime

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp luật Việt Nam. Cung cấp thông tin dựa 
trên dữ liệu được truy xuất — KHÔNG thay thế tư vấn từ luật sư có giấy phép.
Trả lời bằng tiếng Việt, chuyên nghiệp nhưng dễ hiểu với người không chuyên.

NGUYÊN TẮC DỮ LIỆU:
1. Chỉ dùng thông tin từ ngữ cảnh truy xuất. Không suy diễn số điều, mức phạt, thời hạn, thủ tục nếu context không nêu rõ.
2. Ngữ cảnh ĐỦ: Trả lời trực tiếp + trích dẫn [Tên văn bản, số hiệu, Điều/Khoản/Điểm].
3. Ngữ cảnh MỘT PHẦN: Trả lời phần có căn cứ, ghi rõ phần nào còn thiếu dữ liệu.
4. Ngữ cảnh KHÔNG CÓ: "Chưa đủ căn cứ trong dữ liệu đã nạp để trả lời câu hỏi này."
5. Nhiều nguồn: Ưu tiên: Hiến pháp > Luật > Nghị định > Thông tư; văn bản mới hơn > cũ hơn. Nêu rõ nếu có mâu thuẫn.
6. Context có dấu hiệu hết hiệu lực (bị thay thế/sửa đổi): cảnh báo người dùng.
7. Câu hỏi mơ hồ: hỏi lại 1-2 điểm cần làm rõ trước khi trả lời.

ĐỊNH DẠNG:
- Câu đơn giản:
  Căn cứ: [Tên văn bản, số hiệu - Điều X Khoản Y]
  → [Trả lời ngắn gọn]

- Câu phức tạp:
  Căn cứ pháp lý:
  • [Văn bản 1 - Điều X]: ...
  • [Văn bản 2 - Điều Y]: ...
  Trả lời: [Giải thích theo từng khía cạnh]
  Lưu ý: [Giới hạn dữ liệu hoặc cảnh báo hiệu lực — nếu có]

TÌNH HUỐNG NHẠY CẢM (hình sự, tranh chấp, đất đai, khẩn cấp):
Kết thúc bằng: "📌 Thông tin trên chỉ mang tính tham khảo. Vui lòng tham vấn 
luật sư hoặc cơ quan có thẩm quyền để được tư vấn cho trường hợp cụ thể."
"""

@dataclass
class RetrievedChunk:
    text: str
    meta: Dict[str, str]
    dense_rank: int
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class LegalRAGBot:
    def __init__(
        self,
        domains_path: str = "../domains.json",           # <-- THAY db_path
        collection_name_map: Optional[dict] = None,     # mặc định dùng domain key
        embedding_model: str = "AITeamVN/Vietnamese_Embedding_v2",
        reranker_model: str = "AITeamVN/Vietnamese_Reranker",
        groq_model: str = "openai/gpt-oss-120b",
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load embedding model (dùng chung cho cả routing + retrieval)
        self.emb_model = SentenceTransformer(embedding_model, device=self.device)
        self.emb_model.max_seq_length = 1024

        # Load reranker
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model
        ).to(self.device)
        self.reranker_model.eval()

        # Load domains config
        with open(domains_path, "r", encoding="utf-8") as f:
            self.domains = json.load(f)

        # Load tất cả ChromaDB collections (lazy-connect, không load data vào RAM)
        self.collections: Dict[str, chromadb.Collection] = {}
        for key, cfg in self.domains.items():
            try:
                client = chromadb.PersistentClient(path=cfg["chroma_path"])
                col_name = (collection_name_map or {}).get(key, key)
                self.collections[key] = client.get_collection(col_name)
                print(f"[OK] Loaded collection: {key} ({cfg['chroma_path']})")
            except Exception as e:
                print(f"[WARN] Không load được collection '{key}': {e}")

        # Khởi tạo router (dùng chung emb_model)
        self.router = DomainRouter(self.domains, self.emb_model)

        # Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GROQ_API_KEY")
        self.groq_client = Groq(api_key=api_key)
        self.groq_model = groq_model

    def _extract_constraints(self, query: str) -> Dict[str, Optional[str]]:
        query_norm = preprocess.normalize_text(query)

        document_number = None
        match = re.search(r"\b\d{1,4}/\d{4}/[a-zđ\-]+", query_norm, flags=re.IGNORECASE)
        if match:
            document_number = match.group(0).upper()

        article = re.search(r"(điều\s+\d+)", query_norm, flags=re.IGNORECASE)
        clause = re.search(r"(khoản\s+\d+)", query_norm, flags=re.IGNORECASE)
        point = re.search(r"(điểm\s+[a-zđ])", query_norm, flags=re.IGNORECASE)

        return {
            "document_number": document_number,
            "article": article.group(1) if article else None,
            "clause": clause.group(1) if clause else None,
            "point": point.group(1) if point else None,
        }

    def _expand_queries(self, query: str) -> List[str]:
        variants = [query]
        norm = preprocess.normalize_text(query)

        if "quy định" not in norm and "điều" not in norm:
            variants.append(f"quy định pháp luật về {query}")
        elif "mức phạt" in norm or "phạt" in norm:
            variants.append(f"mức xử phạt đối với {query}")
            
        return variants[:2]  # ⬅️ Giới hạn cứng 2

    def _lexical_score(self, query: str, meta: Dict[str, str], text: str) -> float:
        query_tokens = set(preprocess.simple_tokens(query))
        if not query_tokens:
            return 0.0

        target = " ".join(
            [
                text,
                meta.get("title", ""),
                meta.get("document_number", ""),
                meta.get("legal_type", ""),
                meta.get("legal_sectors", ""),
                meta.get("article", ""),
                meta.get("clause", ""),
                meta.get("point", ""),
            ]
        )
        target_tokens = set(preprocess.simple_tokens(target))
        return len(query_tokens & target_tokens) / max(len(query_tokens), 1)

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 30,
        top_k_rerank: int = 24,
        top_k_final: int = 4,
        router_top_k: int = 2,       # <-- số domain được route tới
    ) -> List[RetrievedChunk]:
        constraints = self._extract_constraints(query)
        all_candidates: List[RetrievedChunk] = []

        # --- ROUTING ---
        routed_domains = self.router.route(query, top_k=router_top_k)
        active_collections = {
            k: self.collections[k]
            for k in routed_domains
            if k in self.collections
        }

        if not active_collections:
            return []

        # --- DENSE RETRIEVAL trên các collection được route ---
        for q_index, query_variant in enumerate(self._expand_queries(query)):
            query_embedding = self.emb_model.encode(
                [query_variant],
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            for domain_key, collection in active_collections.items():
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=top_k_dense,
                )

                for dense_rank, (doc, meta) in enumerate(
                    zip(results["documents"][0], results["metadatas"][0]), start=1
                ):
                    meta = meta or {}
                    lexical_score = self._lexical_score(query, meta, doc)
                    score = (1.0 / (dense_rank + q_index)) + (1.4 * lexical_score)
                    score += preprocess._calculate_recency_boost(meta, query) * 0.35

                    if constraints["document_number"] and constraints["document_number"] == preprocess.normalize_text(meta.get("document_number", "")).upper():
                        score += 2.5
                    if constraints["article"] and constraints["article"] in preprocess.normalize_text(meta.get("article", "")):
                        score += 0.7
                    if constraints["clause"] and constraints["clause"] in preprocess.normalize_text(meta.get("clause", "")):
                        score += 0.7
                    if constraints["point"] and constraints["point"] in preprocess.normalize_text(meta.get("point", "")):
                        score += 0.7

                    all_candidates.append(
                        RetrievedChunk(
                            text=doc,
                            meta=meta,
                            dense_rank=dense_rank,
                            lexical_score=lexical_score,
                            final_score=score,
                        )
                    )

        # all_candidates.sort(key=lambda item: item.final_score, reverse=True)

        # Sắp xếp chính theo score, phụ theo ngày ban hành (mới hơn lên trước)
        def sort_key(item: RetrievedChunk) -> Tuple[float, datetime.datetime]:
            doc_date = datetime.datetime.strptime(item.meta.get("issuance_date", "01/01/1900"), "%d/%m/%Y")
            return (item.final_score, doc_date)
        all_candidates.sort(key=sort_key, reverse=True)

        deduped: List[RetrievedChunk] = []
        seen = set()
        for item in all_candidates:
            key = (
                item.meta.get("doc_id", ""),
                item.meta.get("article", ""),
                item.meta.get("clause", ""),
                item.meta.get("point", ""),
                item.text[:160],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= top_k_rerank:
                break

        pairs = [[query, item.text] for item in deduped]
        inputs = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            rerank_scores = self.reranker_model(**inputs).logits.view(-1).float().cpu().tolist()

            for item, rerank_score in zip(deduped, rerank_scores):
                item.rerank_score = rerank_score
                item.final_score = (0.65 * rerank_score) + (0.35 * item.final_score)

        deduped.sort(key=lambda item: item.final_score, reverse=True)
        return deduped[:top_k_final]

    def _build_context(self, retrieved: Sequence[RetrievedChunk]) -> str:
        blocks: List[str] = []
        for idx, item in enumerate(retrieved, start=1):
            citation = item.meta.get("citation") or preprocess.build_citation(item.meta)

            text = item.text.strip()
            # Cắt ở 500 ký tự, giữ nguyên ranh giới từ
            if len(text) > 500:
                text = text[:500].rsplit(" ", 1)[0] + "..."

            blocks.append(
                f"[Căn cứ {idx}]\n"
                f"Nguồn: {citation}\n"
                f"Ngày ban hành: {item.meta.get('issuance_date', '')}\n"
                f"Nội dung:\n{item.text}"
            )
        return "\n\n".join(blocks)

    def _should_fallback(self, retrieved: Sequence[RetrievedChunk]) -> bool:
        if not retrieved:
            return True

        top = retrieved[0]
        strong_signal = (
            top.rerank_score >= 0.02
            or top.lexical_score >= 0.15
            or top.final_score >= 0.25
        )

        if strong_signal:
            return False

        second_ok = len(retrieved) >= 2 and (
            retrieved[1].rerank_score >= 0.03
            or retrieved[1].lexical_score >= 0.18
        )

        return not second_ok

    def answer(self, query: str, history: Optional[Sequence[Tuple[str, str]]] = None) -> Tuple[str, List[Dict[str, str]]]:
        retrieved = self.retrieve(query)
        if self._should_fallback(retrieved):
            return (
                "Chưa tìm thấy ngữ cảnh pháp lý đủ gần với câu hỏi trong dữ liệu đã nạp. "
                "Bạn nên nêu rõ hơn chủ đề, số hiệu văn bản hoặc điều/khoản cần hỏi.",
                [],
            )

        context = self._build_context(retrieved)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for user_text, assistant_text in (history or [])[-3:]:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

        messages.append(
            {
                "role": "user",
                "content": (
                    f"NGỮ CẢNH PHÁP LÝ:\n{context}\n\n"
                    f"CÂU HỎI: {query}\n\n"
                    "Hãy trả lời đúng theo ngữ cảnh. Nếu dữ liệu không đủ để khẳng định năm áp dụng hoặc hiệu lực hiện hành,"
                " phải nói rõ giới hạn đó, nhưng vẫn tóm tắt được phần căn cứ đang truy xuất."
                ),
            }
        )

        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=1300,
            top_p=1,
        )
        answer = response.choices[0].message.content.strip()

        citations = [
            {
                "citation": item.meta.get("citation") or preprocess.build_citation(item.meta),
                "source_brief": preprocess.build_source_brief(item.meta),
                "title": item.meta.get("title", ""),
                "document_number": item.meta.get("document_number", ""),
                "legal_type": item.meta.get("legal_type", ""),
                "issuance_date": item.meta.get("issuance_date", ""),
                "article_clause": " | ".join(
                    [part for part in [item.meta.get("article", ""), item.meta.get("clause", ""), item.meta.get("point", "")] if part]
                ),
                "lexical_score": round(item.lexical_score, 4),
                "rerank_score": round(item.rerank_score, 4),
                "final_score": round(item.final_score, 4),
            }
            for item in retrieved
        ]
        return answer, citations

    def stream_answer(self, query: str, history: Optional[Sequence[Tuple[str, str]]] = None):
        retrieved = self.retrieve(query)
        if self._should_fallback(retrieved):
            fallback = (
                "Chưa đủ căn cứ trong dữ liệu đã nạp để trả lời chắc chắn câu hỏi này. "
                "Bạn nên nêu rõ tên văn bản, số hiệu hoặc điều/khoản cần hỏi."
            )
            yield fallback, []
            return

        context = self._build_context(retrieved)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for user_text, assistant_text in (history or [])[-3:]:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

        messages.append(
            {
                "role": "user",
                "content": (
                    f"NGỮ CẢNH PHÁP LÝ:\n{context}\n\n"
                    f"CÂU HỎI: {query}\n\n"
                    "Hãy trả lời đúng theo ngữ cảnh. Nếu dữ liệu không đủ để khẳng định năm áp dụng hoặc hiệu lực hiện hành,"
                    " phải nói rõ giới hạn đó, nhưng vẫn tóm tắt được phần căn cứ đang truy xuất."
                ),
            }
        )

        citations = [
            {
                "citation": item.meta.get("citation") or preprocess.build_citation(item.meta),
                "source_brief": preprocess.build_source_brief(item.meta),
                "title": item.meta.get("title", ""),
                "document_number": item.meta.get("document_number", ""),
                "legal_type": item.meta.get("legal_type", ""),
                "issuance_date": item.meta.get("issuance_date", ""),
                "article_clause": " | ".join(
                    [part for part in [item.meta.get("article", ""), item.meta.get("clause", ""), item.meta.get("point", "")] if part]
                ),
                "score": round(item.final_score, 4),
                "rerank_score": round(item.rerank_score, 4),
                "lexical_score": round(item.lexical_score, 4),
            }
            for item in retrieved
        ]

        stream = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=1300,
            top_p=1,
            stream=True,
        )

        collected = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            collected += delta
            yield collected, citations