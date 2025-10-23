from sentence_transformers import SentenceTransformer, util, losses
import chromadb
import time

# 1. Load model SBERT/ 
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4️⃣ Inisialisasi ChromaDB (local)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="kerentanan")
print("metadata : ", collection.metadata)

query = "apa itu network"

start = time.time()
# proses
search_embeddings = model.encode(query, convert_to_tensor=True, device='cuda').tolist()
resquery = collection.query(
    query_embeddings=search_embeddings,
    n_results=2
)

end = time.time()
print("Durasi:", end - start, "detik")
for doc, score in zip(resquery["documents"][0], resquery["distances"][0]):
    print(f"Skor: {score:.4f}")
    print(f"Teks: {doc}\n")

print("result: ", resquery['documents'][0][0], "\nsimilarity: ", 1 - resquery['distances'][0][0])