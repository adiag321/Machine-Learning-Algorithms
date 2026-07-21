## Real-World RAG Examples

############################################### 
# 1. Cleaning Text Chunks
############################################### 

# Raw text extracted from a parsed PDF page
raw_lines = ["  Attention mechanism is key. ", "   ", "Embeddings map text to vectors.  ", ""]

# --- Traditional For Loop Way ---
cleaned_lines_loop = []
for line in raw_lines:
    stripped = line.strip()
    if len(stripped) > 0:
        cleaned_lines_loop.append(stripped)

# --- Sleek List Comprehension Way ---
cleaned_lines_comp = [line.strip() for line in raw_lines if len(line.strip()) > 0]

print("Loop version: ", cleaned_lines_loop)
print("Comprehension way: ", cleaned_lines_comp)
# Both Output: ['Attention mechanism is key.', 'Embeddings map text to vectors.']


############################################### 
# 2. Binding Pipeline Metrics
############################################### 
chunk_ids = ["id_01", "id_02", "id_03"]
chunks = ["Attention maps queries...", "Embeddings represent text...", "LLMs generate answers..."]
sources = ["attention.pdf", "embeddings.pdf", "llm_report.pdf"]

# Zipping three lists concurrently to build structured payloads
vector_db_payloads = []
for doc_id, text, source_file in zip(chunk_ids, chunks, sources):
    payload = {
        "id": doc_id,
        "text": text,
        "metadata": {"source": source_file}
    }
    vector_db_payloads.append(payload)

print(vector_db_payloads[0])
# Output: {'id': 'id_01', 'text': 'Attention maps queries...', 'metadata': {'source': 'attention.pdf'}}