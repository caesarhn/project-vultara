from sentence_transformers import SentenceTransformer, util
import json
import chromadb

# Misalnya data JSON sudah di-load seperti ini:
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Ubah setiap entri menjadi 1 kalimat dan simpan dalam list
corpus = [
    f"nama: {item['nama']}, tech: {item['tech']}, versi: {item['versi']}, deskripsi: {item['deskripsi']}"
    for item in data
]

# 1. Load model SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Encode corpus sekali saja
# corpus_embeddings = model.encode(corpus, convert_to_tensor=True, device='cuda')
corpus_embeddings = model.encode(corpus, convert_to_tensor=True, device='cuda').tolist()
# print("corpus : ", corpus_embeddings.shape)

# 4️⃣ Inisialisasi ChromaDB (local)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="kerentanan", metadata={"hnsw:space": "cosine"})

# 5️⃣ Tambahkan data ke ChromaDB
ids = [f"doc-{i}" for i in range(len(corpus))]
collection.add(
    documents=corpus,
    embeddings=corpus_embeddings,
    ids=ids
)
print("Data kerentanan berhasil disimpan di ChromaDB!")

# # 4. Masukkan query dari pengguna
# query = "carikan kelemahan pada server apache"
# query_embedding = model.encode(query, convert_to_tensor=True)

# # 5. Hitung kemiripan cosine
# cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

# # 6. Urutkan berdasarkan similarity
# top_k = 3
# top_results = sorted(list(enumerate(cosine_scores)), key=lambda x: x[1], reverse=True)[:top_k]

# # 7. Tampilkan hasil
# print(f"\nQuery: {query}\nTop {top_k} Hasil Paling Relevan:")
# for idx, score in top_results:
#     print(f"Score: {score:.4f} - \"{corpus[idx]}\"")
