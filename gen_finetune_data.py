import json
from tqdm import tqdm
import re
import csv
import random
import pandas as pd


data_csv = [
    ["id", "query", "document"]
]

# Load data cve:
with open('nvdcve-1.1-2024.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# load data retrieval tambahan
data_general = pd.read_csv("base_finetune_retrieval.csv", quotechar="'", sep=';', quoting=3)

for row in tqdm(data_general.itertuples(), total=data_general.shape[0], desc="Generating general retrieval"):
    data_csv.append(["noid", f"apa itu {row.query}", row.passage])
    data_csv.append(["noid", f"what is {row.query}", row.passage])

for item in tqdm(data["CVE_Items"], desc="Generating to Sentences"):
    cve_id = item["cve"]["CVE_data_meta"]["ID"]
    cpe_uris = []

    #make part of variant query
    vulnerable_on = []
    vulnerable_on_en = []
    vulnerable_on_id = []

    for node in item.get("configurations", {}).get("nodes", []):
        for cpe in node.get("cpe_match", []):
            cpe_clean = re.sub(r"[^a-zA-Z0-9\s]", "", cpe["cpe23Uri"])
            #clean version of cpe
            cpe_clean = cpe_clean[5:]
            cpe_detail = f'{cpe_clean} version {cpe.get("versionStartIncluding", "none")} - {cpe.get("versionEndIncluding", "none")}'
            cpe_uris.append(cpe_detail)
            vulnerable_on_en.append(f"vulnerable on {cpe_detail}, ")
            vulnerable_on_id.append(f"rentan di {cpe_detail}, ")
            vulnerable_on.append(cpe_detail)
    
    cve_desc = item["cve"]["description"]["description_data"][0]["value"]
    impact_data = (
        item.get("impact", {})
            .get("baseMetricV3", {})
            .get("cvssV3", {})
    )

    #make document for target embedding
    impact_string = f"this is impact list for this document: attack via {impact_data.get('attackVector')}, attack complexity is {impact_data.get('attackComplexity')}, is need privileges? {impact_data.get('privilegesRequired')}, is need user interaction? {impact_data.get('userInteraction')}, scope is {impact_data.get('scope')}, confidentiality impact is {impact_data.get('confidentialityImpact')}, integrity impact is {impact_data.get('integrityImpact')}, availability impact is {impact_data.get('availabilityImpact')}, base score is {impact_data.get('baseScore')}, base severity is {impact_data.get('baseSeverity')}"
    doc = f"id: {cve_id} list cpe: {cpe_uris} description: {cve_desc} {impact_string}"
    impact_string = impact_string.lower()

    #make query
    is_privilege_en = "with privilege, " if impact_data.get('privilegesRequired') == "HIGH" else "with privilege low, " if impact_data.get('privilegesRequired') == "LOW" else "with no privilege, "
    is_privilege_id = "dengan privilege, " if impact_data.get('privilegesRequired') == "HIGH" else "dengan privilege rendah, " if impact_data.get('privilegesRequired') == "LOW" else "tanpa privilege, "
    user_interaction_en = "need user interaction, " if impact_data.get('userInteraction') == "REQUIRED" else "no user interaction, "
    user_interaction_id = "perlu interaksi user, " if impact_data.get('userInteraction') == "REQUIRED" else "tanpa interaksi user, "

    attribute = [f"attack via {impact_data.get('attackVector')}, ", f"complexity {impact_data.get('attackComplexity')}, ", f"privilege {impact_data.get('privilegesRequired')}, ", f"user {impact_data.get('userInteraction')}, ", f"scope {impact_data.get('scope')}, ", f"confidentiality {impact_data.get('confidentialityImpact')}, ", f"integrity {impact_data.get('integrityImpact')}, ", f"availability {impact_data.get('availabilityImpact')}, ", f"score {impact_data.get('baseScore')}, ", f"severity {impact_data.get('baseSeverity')}, "]
    attribute_en = f"attack vector {impact_data.get('attackVector')}, attack complexity {impact_data.get('attackComplexity')}, {is_privilege_en}, {user_interaction_en}, scope {impact_data.get('scope')}, confidentiality impact {impact_data.get('confidentialityImpact')}, integrity impact is {impact_data.get('integrityImpact')}, availability impact is {impact_data.get('availabilityImpact')}, base score is {impact_data.get('baseScore')}, base severity is {impact_data.get('baseSeverity')} "
    attribute_id = f"vektor serangan {impact_data.get('attackVector')}, kompleksitas serangan {impact_data.get('attackComplexity')}, {is_privilege_id}, {user_interaction_id}, scope {impact_data.get('scope')}, dampak kerahasiaan {impact_data.get('confidentialityImpact')}, dampak integritas {impact_data.get('integrityImpact')}, dampak ketersediaan {impact_data.get('availabilityImpact')}, skor dasar {impact_data.get('baseScore')}, skor dampak kerusakan {impact_data.get('baseSeverity')} "

    #add normal variant
    data_csv.append([cve_id, f"{vulnerable_on_en} {attribute_en}", doc])
    data_csv.append([cve_id, f"{vulnerable_on_id} {attribute_id}", doc])

    #add another variant
    data_csv.append([cve_id, f"{", ".join([str(item).lower() for item in attribute])} {", ".join([str(item) for item in vulnerable_on])}", doc])
    data_csv.append([cve_id, f"{", ".join([str(item) for item in vulnerable_on])} {", ".join([str(item).lower() for item in attribute])}", doc])
    random.shuffle(vulnerable_on)
    random.shuffle(attribute)
    data_csv.append([cve_id, f"{", ".join([str(item).lower() for item in attribute])} {", ".join([str(item) for item in vulnerable_on])}", doc])
    data_csv.append([cve_id, f"{", ".join([str(item) for item in vulnerable_on])} {", ".join([str(item).lower() for item in attribute])}", doc])

for i in range(10):
    print("data: ", data_csv[i])

filename = "data_finetune_v1.csv"

# Tulis file CSV
with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    for row in tqdm(data_csv, desc="generating csv"):
        writer.writerow(row)

print(f"File '{filename}' berhasil dibuat.")

