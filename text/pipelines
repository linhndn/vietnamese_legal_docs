domains.json
     │
     ├─ [index time] build_and_index_domain() → chroma_db/dan_su/
     │                                        → chroma_db/hinh_su/
     │                                        → chroma_db/giao_thong/  ...
     │
     └─ [query time] DomainRouter.route(query)
                          │
                          ▼ top-2 domains
                     Dense Retrieval (multi-collection)
                          │
                          ▼
                     Lexical scoring + Constraint boost
                          │
                          ▼
                     Vietnamese_Reranker (Reranker 1)
                          │
                          ▼
                     Top-6 chunks → LLM (Groq)