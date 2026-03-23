VECTORSTORE_DIR    = "faiss_index"
EMBEDDING_MODEL    = "nomic-embed-text"
LLM_MODEL          = "llama3"

CACHE_FILE               = "semantic_cache.json"
CACHE_SIMILARITY_THRESHOLD = 0.85   # tune up to reduce false hits

BM25_WEIGHT  = 0.4
DENSE_WEIGHT = 0.6
TOP_K        = 4
