# ============================================================
# RAG + LoRA Fine-Tuned Pipeline
# Run in the SAME Colab session as training/eval (models already loaded),
# or fresh (will reload the fine-tuned model too).
# ============================================================

# ---- CELL 1: Install RAG-specific dependencies ----
!pip install -q sentence-transformers chromadb

# ---- CELL 2: Upload schema_corpus.json ----
# Upload schema_corpus.json to this Colab session's file panel.

# ---- CELL 3: Build the vector store from the schema corpus ----
import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("schema_corpus.json") as f:
    corpus = json.load(f)

print(f"Loaded {len(corpus)} schema documents.")

# Small, CPU-friendly embedding model - no GPU needed for this part,
# it's just converting text into vectors that capture semantic meaning.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory vector store for this session. For a real deployment this would
# persist to disk, but for a portfolio demo, in-memory is simpler and sufficient.
chroma_client = chromadb.Client()
# Delete if it exists already (avoids errors on re-running this cell)
try:
    chroma_client.delete_collection("schema_corpus")
except Exception:
    pass
collection = chroma_client.create_collection("schema_corpus")

doc_texts = [doc["text"] for doc in corpus]
doc_ids = [doc["doc_id"] for doc in corpus]
doc_embeddings = embedder.encode(doc_texts).tolist()

collection.add(
    ids=doc_ids,
    embeddings=doc_embeddings,
    documents=doc_texts,
    metadatas=[{"tmdl_definition": doc["tmdl_definition"], "type": doc["type"]} for doc in corpus],
)

print(f"Vector store built with {collection.count()} documents.")

# ---- CELL 4: Retrieval function ----
def retrieve_schema_context(query_text, top_k=2):
    """
    Given an input (e.g. a cube SQL snippet or metric description),
    retrieve the top_k most relevant schema documents.
    """
    query_embedding = embedder.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    retrieved_docs = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    context_lines = []
    for doc_text, meta in zip(retrieved_docs, retrieved_metadata):
        context_lines.append(f"- {doc_text}")

    return "\n".join(context_lines)


# Quick sanity check - try retrieving context for a sample query
sample_query = "SELECT SUM(revenue) FROM fact_sales WHERE region = 'APAC'"
print("Sample retrieval for:", sample_query)
print(retrieve_schema_context(sample_query))

# ---- CELL 5: RAG-augmented generation ----
def generate_with_rag(model, tokenizer, instruction, input_text, max_new_tokens=200):
    schema_context = retrieve_schema_context(input_text, top_k=2)

    # This is the key architectural change from plain fine-tuning:
    # we inject retrieved schema context INTO the prompt before generation.
    # The model still uses its fine-tuned pattern knowledge, but now
    # grounded in specific (retrieved) schema details.
    prompt = (
        f"### Relevant Schema Context:\n{schema_context}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Output:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("### Output:")[-1].strip()
    return response


# ---- CELL 6: Three-way comparison on test set ----
# base (no fine-tune, no RAG) vs fine-tuned only vs fine-tuned + RAG
import torch

with open("test.json") as f:
    test_data = json.load(f)

three_way_results = []

for i, example in enumerate(test_data):
    print(f"Running example {i+1}/{len(test_data)} ({example['category']})...")

    with finetuned_model.disable_adapter():
        base_output = generate_response(finetuned_model, tokenizer, example["instruction"], example["input"])

    finetuned_output = generate_response(finetuned_model, tokenizer, example["instruction"], example["input"])

    rag_output = generate_with_rag(finetuned_model, tokenizer, example["instruction"], example["input"])

    three_way_results.append({
        "category": example["category"],
        "input": example["input"],
        "expected": example["output"],
        "base_output": base_output,
        "finetuned_output": finetuned_output,
        "finetuned_rag_output": rag_output,
    })

with open("eval_results_three_way.json", "w") as f:
    json.dump(three_way_results, f, indent=2)

print("Done. Download eval_results_three_way.json and bring it back for scoring.")
