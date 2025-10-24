import joblib, numpy as np, faiss
from sentence_transformers import SentenceTransformer

print("\n📘 Welcome to your RAG-Based AI Teaching Assistant!")
print("This assistant retrieves relevant transcript chunks from the Python course.")
print("------------------------------------------------------------")

# Load saved vectors
try:
    art = joblib.load("vectors.joblib")
    embs = art["embeddings"].astype("float32")
    texts = art["texts"]
except Exception as e:
    print(f"❌ Error loading vectors.joblib: {e}")
    exit()

# Build FAISS index
faiss.normalize_L2(embs)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

# Load encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Starter question list
sample_questions = [
    "Explain Python functions with an example",
    "What are loops in Python?",
    "How do conditional statements work?",
    "What is list comprehension in Python?",
    "Explain the use of dictionaries in Python",
    "What is recursion?",
    "How do you handle exceptions in Python?",
    "Explain object-oriented programming concepts",
    "What is the difference between tuples and lists?",
    "Explain how 'for' and 'while' loops differ"
]

print("\n💡 Try asking one of these sample questions:\n")
for q in sample_questions:
    print(f"  ➤ {q}")
print("\n(Type 'exit' or 'quit' anytime to stop.)\n")

# Main interactive loop
while True:
    query = input("\n🔹 Ask your question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("\n👋 Exiting assistant. Goodbye!")
        break

    # Encode and retrieve
    qv = encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)
    D, I = index.search(qv, 5)

    print("\n🔍 Top relevant transcript snippets:\n")
    for r, i in enumerate(I[0], 1):
        snippet = texts[i][:250].replace("\n", " ")
        print(f"{r}. (score={D[0][r-1]:.3f})  {snippet}…")

    print("\n---------------------------------------------")
