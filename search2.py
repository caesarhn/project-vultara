from sentence_transformers import SentenceTransformer, util
import json

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
corpus_embeddings = model.encode(corpus, convert_to_tensor=True, device='cuda')
# print("corpus : ", corpus_embeddings.shape)

# 4. Masukkan query dari pengguna
query = "nama: CVE-2021-44228, tech: Apache Log4j, versi: 2.14.1, deskripsi: Log4Shell memungkinkan eksekusi kode jarak jauh."
query_embedding = model.encode(query, convert_to_tensor=True, device='cuda')

# 5. Hitung kemiripan cosine
cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

# 6. Urutkan berdasarkan similarity
top_k = 3
top_results = sorted(list(enumerate(cosine_scores)), key=lambda x: x[1], reverse=True)[:top_k]

# 7. Tampilkan hasil
print(f"\nQuery: {query}\nTop {top_k} Hasil Paling Relevan:")
for idx, score in top_results:
    print(f"Score: {score:.4f} - \"{corpus[idx]}\"")
