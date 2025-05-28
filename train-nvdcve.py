from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import json
import chromadb
import re

# Misalnya data JSON sudah di-load seperti ini:
with open('nvdcve-1.1-2024.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Ubah setiap entri menjadi 1 kalimat dan simpan dalam list
corpus = []
corpus_id = []

for item in tqdm(data["CVE_Items"], desc="Generating to Sentences"):
    cve_id = item["cve"]["CVE_data_meta"]["ID"]
    cpe_uris = []

    for node in item.get("configurations", {}).get("nodes", []):
        for cpe in node.get("cpe_match", []):
            cpe_clean = re.sub(r"[^a-zA-Z0-9\s]", "", cpe["cpe23Uri"])
            #clean version of cpe
            cpe_clean = cpe_clean[5:]
            cpe_detail = f'{cpe_clean}, version {cpe.get("versionStartIncluding", "none")} - {cpe.get("versionEndIncluding", "none")}'
            cpe_uris.append(cpe_detail)
    
    cve_desc = item["cve"]["description"]["description_data"][0]["value"]
    impact_data = (
        item.get("impact", {})
            .get("baseMetricV3", {})
            .get("cvssV3", {})
    )
    # "attackVector" : "NETWORK",
    #       "attackComplexity" : "LOW",
    #       "privilegesRequired" : "NONE",
    #       "userInteraction" : "NONE",
    #       "scope" : "UNCHANGED",
    #       "confidentialityImpact" : "HIGH",
    #       "integrityImpact" : "HIGH",
    #       "availabilityImpact" : "HIGH",
    #       "baseScore" : 9.8,
    #       "baseSeverity" : "CRITICAL"
    impact_string = f"this is impact list for this document: attack via {impact_data.get('attackVector')}, attack complexity is {impact_data.get('attackComplexity')}, is need privileges? {impact_data.get('privilegesRequired')}, is need user interaction? {impact_data.get('userInteraction')}, scope is {impact_data.get('scope')}, confidentiality impact is {impact_data.get('confidentialityImpact')}, integrity impact is {impact_data.get('integrityImpact')}, availability impact is {impact_data.get('availabilityImpact')}, base score is {impact_data.get('baseScore')}, base severity is {impact_data.get('baseSeverity')}"
    impact_string = impact_string.lower()

    add = f"id: {cve_id} list cpe: {cpe_uris} description: {cve_desc} {impact_string}"
    corpus.append(add)
    corpus_id.append(cve_id)

print("length : ", len(corpus))
print("example: \n")
for i in range(5):
    print("data: ", corpus[i], "\n")

# 1. Load model SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Encode corpus sekali saja
# corpus = corpus[:8000]
# corpus_id = corpus_id[:8000]
corpus_embeddings = []
# corpus_embeddings = model.encode(corpus, convert_to_tensor=True, device='cuda').tolist()
for text in tqdm(corpus, desc="Encoding corpus"):
    emb = model.encode(text, convert_to_tensor=True, device="cuda")
    corpus_embeddings.append(emb.tolist())
# print("corpus : ", corpus_embeddings.shape)

# 4️⃣ Inisialisasi ChromaDB (local)
print("Create chromadb ..")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="kerentanan", metadata={"hnsw:space": "cosine"})

# 5️⃣ Tambahkan data ke ChromaDB
ids = [f"doc-{i}" for i in range(len(corpus))]
# collection.add(
#     documents=corpus_id,
#     embeddings=corpus_embeddings,
#     ids=ids
# )

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# Set batch_size sesuai kemampuan RAM/server
batch_size = 100

# Upload batch ke ChromaDB dengan loading tqdm
for i, (doc_batch, emb_batch, id_batch, docid_batch) in enumerate(
    tqdm(
        zip(
            batchify(corpus, batch_size),
            batchify(corpus_embeddings, batch_size),
            batchify(ids, batch_size),
            batchify(corpus_id, batch_size)
        ),
        total=len(corpus) // batch_size + 1,
        desc="Uploading to ChromaDB"
    )
):
    collection.add(
        documents=docid_batch,
        embeddings=emb_batch,
        ids=id_batch
    )

print("Data kerentanan berhasil disimpan di ChromaDB!")