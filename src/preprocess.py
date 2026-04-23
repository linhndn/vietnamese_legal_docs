import re
from typing import Dict, List
import datetime


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[0-9a-zA-ZÀ-ỹ]+", normalize_text(text))

def build_citation(meta: Dict[str, str]) -> str:
    title = meta.get("title", "").strip()
    doc_number = meta.get("document_number", "").strip()
    legal_type = meta.get("legal_type", "").strip()
    location = " | ".join(
        [part for part in [meta.get("article", ""), meta.get("clause", ""), meta.get("point", "")] if part]
    )
    left = f"{title} ({doc_number})" if doc_number else title
    if legal_type:
        left = f"{left} - {legal_type}"
    return f"{left} | {location}".strip(" |")

def build_source_brief(meta: Dict[str, str]) -> str:
    doc_number = meta.get("document_number", "").strip()
    title = meta.get("title", "").strip()
    left = f"{doc_number} - {title}" if doc_number else title
    return left

def _calculate_recency_boost(meta: Dict[str, str], query: str) -> float:
    """Tính điểm ưu tiên văn bản mới. Max ~1.2, decay theo thời gian, tăng khi query có từ khóa thời gian."""
    date_str = meta.get("issuance_date", "").strip()
    if not date_str:
        return 0.0
    
    try:
        doc_date = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        days_ago = (datetime.datetime.now() - doc_date).days
        if days_ago < 0:
            return 0.0  # Bỏ qua ngày tương lai
            
        # Decay curve: 1.0 cho <1 năm, ~0.6 cho ~2 năm, ~0.3 cho ~3.5 năm, ~0 cho >5 năm
        base_score = max(0.0, 1.0 - (days_ago / 1825.0))
        
        # Tăng trọng số nếu query yêu cầu thông tin hiện hành
        query_norm = normalize_text(query)
        temporal_keywords = ["hiện nay", "mới nhất", "cập nhật", "hiện hành", "2024", "2025", "2026", "năm nay", "mới ban hành"]
        is_temporal_query = any(kw in query_norm for kw in temporal_keywords)
        
        return base_score * (1.5 if is_temporal_query else 0.8)
    except ValueError:
        return 0.0