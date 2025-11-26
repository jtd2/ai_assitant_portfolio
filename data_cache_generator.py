import os
import pickle
import numpy as np
import faiss
import tiktoken
import openai
import time
import random
from tqdm import tqdm
# Import exceptions directly from the openai package
from openai import RateLimitError, APIError, APITimeoutError

# --- Config ---
TXT_PATH = "/content/training_doc.txt"  # Your .txt file
CACHE_DIR = "/content/ada_progs_txt_cache"  # Folder to save cache files
openai.api_key = os.environ.get("OPENAI_API_KEY", "*****")  # Set your key securely

# --- Helper Functions ---
def load_text_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, max_tokens=300):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        chunk.append(word)
        if len(tokenizer.encode(" ".join(chunk))) > max_tokens:
            chunks.append(" ".join(chunk[:-1]))
            chunk = [word]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def get_embedding_safe(text, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = openai.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        # Catch the correct exception types from the updated library
        except (RateLimitError, APIError, APITimeoutError) as e:
            print(f"⚠️ Error: {str(e)} — retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return None

def save_cache(embeddings, chunks, index, base_path):
    os.makedirs(base_path, exist_ok=True)
    np.save(os.path.join(base_path, "embeddings.npy"), embeddings)
    with open(os.path.join(base_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, os.path.join(base_path, "index.faiss"))

# --- Main Script ---
if __name__ == "__main__":
    print("📄 Reading .txt file...")
    text = load_text_file(TXT_PATH)

    print("✂️ Splitting text into chunks...")
    chunks = split_text(text)
    print(f"Total chunks: {len(chunks)}")

    print("🧠 Generating embeddings...")
    embeddings = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
        emb = get_embedding_safe(chunk)
        if emb:
            embeddings.append(emb)
        else:
            print(f"❌ Skipping chunk {i} due to repeated errors.")
            embeddings.append([0.0] * 1536)  # Fallback embedding

        if i % 50 == 0 and i != 0:
            sleep_time = random.uniform(2, 5)
            print(f"⏳ Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

    embeddings_array = np.array(embeddings).astype("float32")

    print("📦 Building FAISS index...")
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)

    print(f"💾 Saving cache to {CACHE_DIR}")
    save_cache(embeddings_array, chunks, index, CACHE_DIR)

    print("✅ Done! Embeddings and index saved.")
