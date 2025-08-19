import tenseal as ts
import torch
import numpy as np
import time
import os
import sys
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

print('Loading pre-converted word vectors...')

load_start = time.time()

base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

coeff_bit_sizes = [40, 20, 20, 40]
poly_mod_degree = 8192

glove = KeyedVectors.load(os.path.join(base_path, "glove_vectors.kv")) # Loading KV File for faster vector retreival

# --- Encryption context setup
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_mod_degree,
    coeff_mod_bit_sizes=coeff_bit_sizes
)
context.global_scale = 2 ** 40
context.generate_galois_keys()
load_time = time.time() - load_start

print(f'Word vectors and context loaded in {load_time:.2f} seconds.')


# === Function to process and encrypt a sentence
def process_sentence(sentence: str):
    """
    Embeds and encrypts a sentence using GloVe + CKKS via TenSEAL.

    Returns:
        dict: Encrypted vector and parameters.
        ts.Context: The TenSEAL context for decryption.
        float: Time (seconds) spent on encryption.
    """
    start_time = time.time()

    # --- Input validation
    if not isinstance(sentence, str):
        raise ValueError("Sentence must be a string.")
    if not isinstance(coeff_bit_sizes, list) or not all(isinstance(b, int) and b > 0 for b in coeff_bit_sizes):
        raise ValueError("coeff_bit_sizes must be a list of positive integers.")
    if not isinstance(poly_mod_degree, int) or poly_mod_degree <= 0:
        raise ValueError("poly_mod_degree must be a positive integer.")

    # --- Word embedding
    tokens = sentence.lower().split()
    vectors = [torch.tensor(glove[token]) if token in glove else torch.zeros(glove.vector_size) for token in tokens]
    if not vectors:
        raise ValueError("No valid tokens found for embedding.")
    
    concat_vector = torch.cat(vectors)  # shape = [300 * number_of_words]

    ctx_start = time.time()

    # --- Encryption
    enc_vector = ts.ckks_vector(context, concat_vector.tolist())
    serialized = enc_vector.serialize()
    enc_time = time.time() - ctx_start

    total_time = time.time() - start_time
    return {
        "input": sentence,
        "params": {
            "coeff_bit_sizes": coeff_bit_sizes,
            "poly_mod_degree": poly_mod_degree
        },
        "vector_length": concat_vector.shape[0],
        "encrypted_vector": serialized.hex()
    }, enc_time, total_time

# === Decryption function
def decrypt_vector(serialized_hex: str, context: ts.Context):
    """
    Decrypts a serialized hex-encoded encrypted vector using the original TenSEAL context.

    Returns:
        list of float: Decrypted vector.
        float: Time (seconds) spent on decryption.
    """
    start = time.time()
    encrypted_bytes = bytes.fromhex(serialized_hex)
    enc_vector = ts.ckks_vector_from(context, encrypted_bytes)
    decrypted = enc_vector.decrypt()
    decrypt_time = time.time() - start
    return decrypted, decrypt_time

def vector_to_sentence(decrypted_vector, glove, topn=1):
    """
    Approximates each word in a sentence from a decrypted vector by splitting it into 300-dim chunks
    and finding the top-N closest words from GloVe for each.

    Parameters:
        decrypted_vector (list): The decrypted vector from CKKS (flattened)
        glove (KeyedVectors): The GloVe embeddings
        topn (int): Number of closest matches per word/chunk

    Returns:
        list of list: Each inner list contains top-N closest words for that word/chunk
    """
    vector_size = glove.vector_size
    chunks = [decrypted_vector[i:i+vector_size] for i in range(0, len(decrypted_vector), vector_size)]

    results = []

    for chunk in chunks:
        chunk_vec = np.array(chunk).reshape(1, -1)
        try:
            similar = glove.similar_by_vector(chunk_vec[0], topn=topn)
            results.append([word for word, _ in similar])
        except Exception as e:
            print("Similarity error, using fallback:", e)
            glove_vectors = glove.vectors
            all_words = list(glove.index_to_key)
            sims = cosine_similarity(chunk_vec, glove_vectors)[0]
            top_indices = np.argsort(sims)[::-1][:topn]
            results.append([all_words[i] for i in top_indices])

    return results

def encrypt(input: str):
    # Encrypt
    encryption, enc_time, process_time = process_sentence(input)
    print("\nEncrypted output preview (hex):", encryption["encrypted_vector"][:100])
    print(f"\n Encryption-only time: {enc_time:.4f} seconds")
    print(f"Total encryption time: {process_time:.4f} seconds")

    return encryption

def decrypt(input):
    start = time.time()

    # Decrypt
    decrypted_vector, decrypt_time = decrypt_vector(input["encrypted_vector"], context)
    print(f"Decryption time: {decrypt_time:.4f} seconds")
    print("\nDecrypted vector (first 10 values):", decrypted_vector[:10])

    # Interpret decrypted vector
    approx_words = vector_to_sentence(decrypted_vector, glove, topn=1)
    sentence = ''
    for i in approx_words:
        sentence += i[0] + " "
    print("Decrypted Sentence: " + sentence)

    total = time.time() - start

    print(f"Total decryption time: {total:.4f} seconds")

    return sentence