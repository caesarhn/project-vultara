import ctypes
import numpy as np
import pandas as pd

# Load DLL
lib = ctypes.CDLL('./tfidf.dll')

# set argument and return for cu code
lib.tfidf.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.c_intp
]

lib.make_corpus.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
lib.make_corpus.restype = ctypes.POINTER(ctypes.c_char_p) 

lib.countTf.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
lib.countTf.restype = ctypes.POINTER(ctypes.c_float) 

lib.countIdf.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]
lib.countIdf.restype = ctypes.POINTER(ctypes.c_float) 

lib.countTfIdf.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]
lib.countTfIdf.restype = ctypes.POINTER(ctypes.c_float) 

def createCorpus(data):
    #variable
    corpus_len = ctypes.c_int(0)
    corpus_list = []
    
    #encode data from list[str] to list[byte]
    encoded = [w.encode('utf-8') for w in data]
    ArrayType = ctypes.c_char_p * len(encoded)
    c_word_array = ArrayType(*encoded)

    #create corpus
    corpus = lib.make_corpus(c_word_array, len(word), ctypes.byref(corpus_len))
    for i in range(corpus_len.value):  # atau corpus_count jika kamu tahu jumlah uniknya
        if not corpus[i]:
            break  # Safety: berhenti jika NULL
        corpus_list.append(corpus[i].decode('utf-8'))
    
    return corpus_list

def tfIdf(data):
    print("test")
    list_of_lists = data['passage'].apply(str.split)
    flattened = [word for sentence in list_of_lists for word in sentence]
    encoded_doc = [w.encode('utf-8') for w in flattened]  # list of bytes
    ArrayType = ctypes.c_char_p * len(encoded_doc)
    c_doc_array = ArrayType(*encoded_doc)

    count_word = data['passage'].apply(lambda x: len(x.split()))
    list_index = count_word.tolist()
    doc_len = len(list_index)
    doc_count = len(flattened)
    c_doc_index = (ctypes.c_int * len(list_index))(*list_index)

    r_cor_len = ctypes.c_int(0)
    tfidf = lib.countTfIdf(c_doc_array, doc_count, doc_len, c_doc_index, ctypes.byref(r_cor_len))

    return tfidf

if __name__ == "__main__" :
    data = [0.32, 0.13, 0.45, 0.62, 0.14, 0.32]
    data = np.array(data, dtype=np.float32)
    result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = np.array(result, dtype=np.float32)

    word = ["akuu", "anjaii", "mabar", "wanjaii", "karena"]
    encoded = [w.encode('utf-8') for w in word]  # list of bytes
    ArrayType = ctypes.c_char_p * len(encoded)
    c_word_array = ArrayType(*encoded)

    doc = [
        "akuu", "anjaii", "mabar", "wanjaii", 
        "akuu", "anjaii", "mabar", "akuu", 
        "mabar", "wanjaii", "akuu", "anjaii",
        "akuu", "anjaii", "mabar", "wanjaii", 
        "akuu", "anjaii", "karena", "akuu", 
        "mabar", "wanjaii", "akuu", "anjaii"
    ]
    doc_count = 12
    doc_index = [4, 4, 4, 4, 4, 4]
    doc_len = 6
    c_doc_index = (ctypes.c_int * len(doc_index))(*doc_index)
    encoded_doc = [w.encode('utf-8') for w in doc]  # list of bytes
    ArrayType = ctypes.c_char_p * len(encoded_doc)
    c_doc_array = ArrayType(*encoded_doc)

    lib.tfidf(data, result, len(data))
    # lib.tfidf2(word, len(word))

    corpus_len = ctypes.c_int(0)
    corpus = lib.make_corpus(c_word_array, len(word), ctypes.byref(corpus_len))
    corpus_list = []
    for i in range(corpus_len.value):  # atau corpus_count jika kamu tahu jumlah uniknya
        if not corpus[i]:
            break  # Safety: berhenti jika NULL
        corpus_list.append(corpus[i].decode('utf-8'))

    print(result)
    print(corpus_list)

    tf = lib.countTf(corpus, corpus_len, c_doc_array, doc_count, doc_len, c_doc_index)
    print(tf)

    idf = lib.countIdf(corpus, corpus_len, c_doc_array, doc_count, doc_len, c_doc_index)

    datas = pd.read_csv("./data.csv", sep=";")
    print(datas.head)
    tfidf = tfIdf(datas)
