import gc
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import chromadb
import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ARTICLE_RE = re.compile(r"(?im)^(Điều\s+\d+[^\n]*)")
CLAUSE_RE = re.compile(r"(?im)^((?:Khoản\s+\d+|\d+\.)[^\n]*)")
POINT_RE = re.compile(r"(?im)^([a-zđ]\)[^\n]*)")

@dataclass
class ChunkConfig:
    max_chunk_chars: int = 1200
    overlap_chars: int = 150

@dataclass
class RuntimeConfig:
    device: str
    encode_batch_size: int
    upsert_batch_size: int
    use_fp16: bool
    max_seq_length: int

def safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.DOTALL)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_long_text(text: str, limit: int = 1200, overlap: int = 150) -> List[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(paragraph) <= limit:
            current = paragraph
            continue

        sentences = [s.strip() for s in re.split(r"(?<=[\.;:!?])\s+", paragraph) if s.strip()]
        temp = ""
        for sentence in sentences:
            candidate = f"{temp} {sentence}".strip()
            if len(candidate) <= limit:
                temp = candidate
            else:
                if temp:
                    chunks.append(temp)
                    temp = temp[-overlap:] + " " + sentence if overlap > 0 else sentence
                else:
                    chunks.append(sentence[:limit])
                    temp = sentence[max(0, limit - overlap):]
        current = temp.strip()

    if current:
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def split_sections(text: str, pattern: re.Pattern) -> List[tuple[str, str]]:
    parts = pattern.split(text)
    if len(parts) <= 1:
        return [("", text.strip())]

    sections: List[tuple[str, str]] = []
    for idx in range(1, len(parts), 2):
        header = parts[idx].strip()
        body = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
        if body:
            sections.append((header, body))
    return sections

def build_citation(meta: Dict[str, str]) -> str:
    title = meta.get("title", "")
    doc_number = meta.get("document_number", "")
    legal_type = meta.get("legal_type", "")
    location = " | ".join(
        [part for part in [meta.get("article", ""), meta.get("clause", ""), meta.get("point", "")] if part]
    )
    left = f"{title} ({doc_number})" if doc_number else title
    if legal_type:
        left = f"{left} - {legal_type}"
    return f"{left} | {location}".strip(" |")

def make_chunk_text(title: str, article: str, clause: str, point: str, body: str) -> str:
    prefix = "\n".join([item for item in [title, article, clause, point] if item])
    return f"{prefix}\n{body}".strip()

def parse_legal_markdown(text: str, meta: Dict[str, str], config: ChunkConfig) -> List[Dict[str, str]]:
    clean_text = normalize_whitespace(text)
    article_sections = split_sections(clean_text, ARTICLE_RE)
    chunks: List[Dict[str, str]] = []

    for article_header, article_body in article_sections:
        clause_sections = split_sections(article_body, CLAUSE_RE)
        for clause_header, clause_body in clause_sections:
            point_sections = split_sections(clause_body, POINT_RE)
            for point_header, point_body in point_sections:
                base_text = make_chunk_text(
                    meta.get("title", ""),
                    article_header,
                    clause_header,
                    point_header,
                    point_body.strip(),
                )
                for piece in split_long_text(base_text, config.max_chunk_chars, config.overlap_chars):
                    item = dict(meta)
                    item.update(
                        {
                            "article": article_header,
                            "clause": clause_header,
                            "point": point_header,
                            "text": piece,
                        }
                    )
                    item["citation"] = build_citation(item)
                    item["location_key"] = " | ".join(
                        [part for part in [article_header, clause_header, point_header] if part]
                    )
                    chunks.append(item)
    return chunks

def build_index_text(chunk: Dict[str, str]) -> str:
    fields = [
        chunk.get("title", ""),
        chunk.get("document_number", ""),
        chunk.get("legal_type", ""),
        chunk.get("legal_sectors", ""),
        chunk.get("issuing_authority", ""),
        chunk.get("article", ""),
        chunk.get("clause", ""),
        chunk.get("point", ""),
        chunk.get("text", ""),
    ]
    return "\n".join([field for field in fields if field]).strip()

def load_legal_dataframe(sector: Optional[str] = None, year_from: Optional[int] = None) -> pl.DataFrame:
    meta = load_dataset("minhnguyent546/vietnamese-legal-documents", "metadata", split="data").to_polars()
    content = load_dataset("minhnguyent546/vietnamese-legal-documents", "content", split="data").to_polars()

    meta = meta.with_columns(pl.col("id").cast(pl.Utf8))
    content = content.with_columns(pl.col("id").cast(pl.Utf8))
    df = content.join(meta, on="id", how="left").drop_nulls(subset=["content", "title"])

    # Xử lý đa nhãn: chuyển về string, fill null, và dùng literal match (nhanh + an toàn)
    if sector:
        sector_col = pl.col("legal_sectors").cast(pl.Utf8).fill_null("")
        df = df.filter(sector_col.str.contains(sector, literal=True))

    if year_from:
        df = df.filter(pl.col("issuance_date").str.to_date("%d/%m/%Y", strict=False).dt.year() >= year_from)
    return df

def iter_chunks(df: pl.DataFrame, config: ChunkConfig) -> Iterable[Dict[str, str]]:
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Chunking legal documents"):
        content_text = row.get("content")
        if not content_text or not isinstance(content_text, str):
            continue

        meta = {
            "doc_id": safe_str(row.get("id")),
            "title": safe_str(row.get("title")),
            "document_number": safe_str(row.get("document_number")),
            "legal_type": safe_str(row.get("legal_type")),
            "legal_sectors": safe_str(row.get("legal_sectors")),
            "issuing_authority": safe_str(row.get("issuing_authority")),
            "issuance_date": safe_str(row.get("issuance_date")),
            "signers": safe_str(row.get("signers")),
            "url": safe_str(row.get("url")),
        }

        try:
            for chunk in parse_legal_markdown(content_text, meta, config):
                yield chunk
        except Exception as exc:
            print(f"Skip doc_id={meta['doc_id']} because {type(exc).__name__}: {exc}")

def build_runtime_config() -> RuntimeConfig:
    if not torch.cuda.is_available():
        return RuntimeConfig(
            device="cpu",
            encode_batch_size=16,
            upsert_batch_size=200,
            use_fp16=False,
            max_seq_length=768,
        )

    device_name = torch.cuda.get_device_name(0).lower()
    if "t4" in device_name:
        return RuntimeConfig(
            device="cuda",
            encode_batch_size=128,
            upsert_batch_size=1500,
            use_fp16=True,
            max_seq_length=1024,
        )

    return RuntimeConfig(
        device="cuda",
        encode_batch_size=96,
        upsert_batch_size=1000,
        use_fp16=True,
        max_seq_length=1024,
    )

def configure_torch_runtime(runtime: RuntimeConfig) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_grad_enabled(False)

    if runtime.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("high")

def index_chunks(
    chunks: List[Dict[str, str]],
    db_path: str = "./chroma_db",
    collection_name: str = "legal_chunks",
    embedding_model: str = "AITeamVN/Vietnamese_Embedding_v2",
    batch_size: Optional[int] = None,
    upsert_batch_size: Optional[int] = None,
) -> None:
    runtime = build_runtime_config()
    configure_torch_runtime(runtime)

    encode_batch_size = batch_size or runtime.encode_batch_size
    write_batch_size = upsert_batch_size or runtime.upsert_batch_size

    model = SentenceTransformer(embedding_model, device=runtime.device)
    model.max_seq_length = runtime.max_seq_length
    model.eval()
    if runtime.device == "cuda" and runtime.use_fp16:
        model.half()

    if runtime.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not found. Falling back to CPU.")
    print(
        f"Embedding runtime -> device={runtime.device}, "
        f"encode_batch_size={encode_batch_size}, "
        f"upsert_batch_size={write_batch_size}, "
        f"fp16={runtime.use_fp16}, "
        f"max_seq_length={runtime.max_seq_length}"
    )

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    documents = [build_index_text(chunk) for chunk in chunks]
    metadatas = chunks
    ids = [
        f"{chunk['doc_id']}::{chunk.get('article', '')}::{chunk.get('clause', '')}::{chunk.get('point', '')}::{idx}"
        for idx, chunk in enumerate(chunks)
    ]

    for start in tqdm(range(0, len(chunks), write_batch_size), desc="Indexing to Chroma"):
        end = min(start + write_batch_size, len(chunks))
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]

        embedding_parts = []
        for sub_start in range(0, len(batch_docs), encode_batch_size):
            sub_end = min(sub_start + encode_batch_size, len(batch_docs))
            sub_docs = batch_docs[sub_start:sub_end]
            with torch.inference_mode():
                sub_embeddings = model.encode(
                    sub_docs,
                    batch_size=encode_batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    device=runtime.device,
                )
            embedding_parts.append(sub_embeddings)

        embeddings = embedding_parts[0] if len(embedding_parts) == 1 else np.concatenate(embedding_parts, axis=0)

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
        )

        gc.collect()
        if runtime.device == "cuda":
            torch.cuda.empty_cache()

def build_and_index(
    db_path: str = "./chroma_db",
    collection_name: str = "legal_chunks",
    sector: Optional[str] = None,
    year_from: Optional[int] = None,
) -> int:
    df = load_legal_dataframe(sector=sector, year_from=year_from)
    config = ChunkConfig()
    chunks = list(iter_chunks(df, config))
    index_chunks(chunks, db_path=db_path, collection_name=collection_name)
    return len(chunks)