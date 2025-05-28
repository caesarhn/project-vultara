# About Project

This project aims to create a semantic search engine that is used to find vulnerability documents from CVE documentation.

## project flows
1. install virtual environtment in python
2. download dataset (CVE document) by execute file: download_cve.py
3. use (embedd the data) or finetune the model

### embedd data
if you dont need fine tuned the model you can use the model to embedded the data immediately and the embed data is ready to use for semantic search.
1. embedd the cve data (that downloaded before) by execute file: train-nvdcve.py
2. it will generated chromadb database in forder: /chromadb
3. to use the semantic search example use file: search.py
you can embedd query from anywhere with the same model used in embedd process and use the embedded query to chromadb that automaticaly used the cosine similarity fungtion.

### fine-tuned the model
1. generate training data by execute file: gen_finetune_data.py (will use base_finetune_retrieval.csv and CVE data)
2. training the pretrained model by execute file: fine-tuned.py (will automatically saved the model when finished)
3. use the model (same to embedd data steps)
