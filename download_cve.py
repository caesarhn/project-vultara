import requests
import gzip
import json
import shutil

# URL JSON feed dari NVD (contoh: data CVE tahun 2024)
url = "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2024.json.gz"
filename_gz = "nvdcve-1.1-2024.json.gz"
filename_json = "nvdcve-1.1-2024.json"

# 1️⃣ Download file
print(f"🔽 Downloading {url} ...")
response = requests.get(url, stream=True)
with open(filename_gz, "wb") as f:
    shutil.copyfileobj(response.raw, f)

print(f"✅ File downloaded: {filename_gz}")

# 2️⃣ Unzip file .gz
with gzip.open(filename_gz, "rb") as f_in:
    with open(filename_json, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"✅ File unzipped: {filename_json}")

# 3️⃣ Parse JSON
with open(filename_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Contoh: Ambil daftar CVE ID
print("📄 Contoh data CVE:")
for i, cve_item in enumerate(data["CVE_Items"][:10]):  # ambil 10 CVE pertama
    cve_id = cve_item["cve"]["CVE_data_meta"]["ID"]
    deskripsi = cve_item["cve"]["description"]["description_data"][0]["value"]
    print(f"{i+1}. {cve_id} - {deskripsi[:100]}...")  # ringkas deskripsi

print("✅ Parsing selesai.")
