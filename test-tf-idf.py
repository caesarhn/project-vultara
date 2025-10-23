from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import nltk
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def custom_tokenizer(text):
    # Pertahankan pola khusus: CVE-YYYY-NNNN, nama produk dengan versi (Apache 2.4)
    tech_patterns = r'''(?x)
        (?:[A-Za-z]{3,}-\d{4}-\d{4,6})       # CVE IDs
        |(?:\b[A-Z][a-z]+\s*\d+\.\d+\b)      # Produk dengan versi (WordPress 5.3)
        |(?:\b[A-Z][a-z]+\s*\d+\.\d+\.\d+\b)
        |\b\d+\.\d+\.\d+\b        # Versi seperti 1.2.3
        |(?:\b[A-Z]{2,}\b)                    # Singkatan (SQL, HTTP)
        |(?:\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)  # Alamat IP
        |(?:\b[a-z]+_\w+\b)                   # Kata dengan underscore
        |(?:\b\w+\b)                          # Kata biasa
    '''
    
    tokens = re.findall(tech_patterns, text)
    
    # Filter token lebih lanjut
    tokens = [token.lower() for token in tokens if len(token) > 2]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    return tokens

df = pd.read_csv("data_finetune_v1.csv")
print(df.iloc[3000:3105, 2])
print(custom_tokenizer(df.iloc[3005, 2]))

# Contoh dokumen
documents = df.iloc[3000:3005, 2].to_list()

# Membuat TF-IDF Vectorizer
security_vocab = [
    'sql injection', 'xss', 'csrf', 'rce', 'buffer overflow',
    'list', 'zero-day', 'auth bypass', 'version', 'affected',
    'affected issue', 'affects', 'affects unknown', 'assigned',
    'assigned vulnerability', 'associated', 'associated identifier', 'attack',
    'attack complexity', 'attack may', 'attack rather', 'attack remotely',
    'attack via', 'availability', 'availability impact', 'base', 'base score',
    'base severity', 'classified', 'classified problematic', 'complexity',
    'complexity attack', 'complexity low', 'confidentiality',
    'confidentiality impact', 'cpe',
    'description', 'description vulnerability', 'difficult', 'difficult exploit',
    'disclosed', 'disclosed public', 'document', 'document attack', 'engineers'
]
vectorizer = TfidfVectorizer(
    # tokenizer=custom_tokenizer,
    stop_words=None,  # Sudah dihandle di tokenizer
    lowercase=False,  # Karena kita ingin mempertahankan case untuk beberapa token
    max_features=5000,
    ngram_range=(1, 2),  # Untuk menangkap frasa seperti "remote code execution"
    min_df=1,           # Abaikan term yang muncul di <2 dokumen
    max_df=1.0,        # Abaikan term yang muncul di >95% dokumen
    analyzer='word',
    vocabulary=security_vocab
)

# Menghitung TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# Mendapatkan fitur (kata-kata)
feature_names = vectorizer.get_feature_names_out()

# Menampilkan hasil
print("Fitur (kata-kata):", feature_names)
print("\nMatriks TF-IDF:")
print(tfidf_matrix.toarray())

# Untuk dokumen baru
new_doc = [""]
new_tfidf = vectorizer.transform(new_doc)
print("\nTF-IDF untuk dokumen baru:", new_tfidf.toarray())