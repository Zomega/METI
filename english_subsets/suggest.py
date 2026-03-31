from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download


def load_glove_model():
    print("Fetching GloVe Model from Hugging Face Hub...")
    # Using a confirmed working mirror
    file_path = hf_hub_download(
        repo_id="JeremiahZ/glove", filename="glove.6B.50d.txt")

    print(f"Loading GloVe Model from {file_path}...")
    embeddings_dict = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    print(f"Done. {len(embeddings_dict)} words loaded.")
    return embeddings_dict


# Load model (50d)
embeddings = load_glove_model()

# Prepare for fast lookup
words = list(embeddings.keys())
word_to_idx = {word: i for i, word in enumerate(words)}
vectors = np.array([embeddings[w] for w in words])

# Normalize for cosine similarity
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
norms[norms == 0] = 1e-10  # prevent division by zero
normalized_vectors = vectors / norms


def find_nearest_neighbors(query_vec, top_n=10):
    # Normalize query
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    query_vec = query_vec / query_norm

    # Cosine similarities
    similarities = np.dot(normalized_vectors, query_vec)

    # Get indices of top_n most similar
    # We use argpartition for efficiency on large vocab
    best_indices = np.argsort(similarities)[::-1][:top_n]

    return [(words[i], similarities[i]) for i in best_indices]


def get_basic_words():
    import pandas as pd
    # Standardized vocab file
    vocab_path = "vocab.csv"
    try:
        df = pd.read_csv(vocab_path)
        basic_words = set(df["word"].dropna().str.lower().unique())
        return basic_words
    except:
        return set()


basic_english_set = get_basic_words()
print(f"Loaded {len(basic_english_set)} basic English words to use as filters.")


print("Loading Transformer Reranker...")
# Lightweight model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModel.from_pretrained(model_name)


def get_transformer_embedding(text):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()


# Simple word frequency from the GloVe file itself (lower index = higher frequency)
# This serves as a proxy for IDF
word_rank = {word: i for i, word in enumerate(words)}
max_rank = len(words)

# A small dictionary of common basic opposites to help the first stage retrieval
BASIC_OPPOSITES = {
    "good": "bad",
    "bad": "good",
    "happy": "sad",
    "sad": "happy",
    "large": "small",
    "small": "large",
    "fast": "slow",
    "slow": "fast",
    "hot": "cold",
    "cold": "hot",
    "easy": "hard",
    "hard": "easy",
    "old": "new",
    "new": "old",
    "rich": "poor",
    "poor": "rich",
    "clean": "dirty",
    "dirty": "clean",
    "true": "false",
    "false": "true"
}


def suggest_complex(simple_phrase, top_n=10):
    parts = simple_phrase.lower().split()
    query_vec = np.zeros(50, dtype="float32")
    count = 0

    # 1. RETRIEVAL (GloVe)
    # Special case for "not X"
    if "not" in parts and len(parts) == 2:
        other_word = parts[1] if parts[0] == "not" else parts[0]
        if other_word in BASIC_OPPOSITES:
            # Pivot to the opposite word's vector
            target = BASIC_OPPOSITES[other_word]
            if target in embeddings:
                query_vec = embeddings[target]
                count = 1
        elif other_word in embeddings:
            # Fallback to subtraction if no direct opposite defined
            query_vec = -1.0 * embeddings[other_word]
            count = 1
    else:
        for p in parts:
            if p in embeddings:
                weight = np.log(word_rank[p] + 2)
                query_vec += (embeddings[p] * weight)
                count += 1

    if count == 0:
        return f"None of the words in '{simple_phrase}' were found."

    # Get more candidates for re-ranking
    neighbors = find_nearest_neighbors(query_vec, top_n=1000)

    candidates = []
    for word, score in neighbors:
        word_lower = word.lower()
        if word_lower in parts:
            continue
        if word_lower in basic_english_set:
            continue
        if not word_lower.isalpha():
            continue
        if len(word_lower) < 3:
            continue

        is_inflection = False
        for p in parts:
            if p in word_lower or word_lower in p:
                if abs(len(p) - len(word_lower)) <= 2:
                    is_inflection = True
                    break
        if is_inflection:
            continue
        candidates.append(word)

    # 2. RERANKING (Transformer)
    # We re-rank a larger pool now
    pool_size = 300
    print(
        f"Reranking {len(candidates[:pool_size])} candidates for '{simple_phrase}'...")
    query_emb = get_transformer_embedding(simple_phrase)

    candidate_scores = []
    for cand in candidates[:pool_size]:
        cand_emb = get_transformer_embedding(cand)
        score = np.dot(query_emb, cand_emb) / \
            (np.linalg.norm(query_emb) * np.linalg.norm(cand_emb))
        candidate_scores.append((cand, score))

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    return candidate_scores[:top_n]


# Test cases
test_cases = [
    "put up with",
    "food bird",
    "large water",
    "fast transport",
    "not good",
    "not happy",
    "many people",
    "small house"
]

print("\nSuggestions for simple phrases:")
for test in test_cases:
    print(f"\n'{test}':")
    results = suggest_complex(test)
    if isinstance(results, list):
        for word, score in results[:5]:
            print(f"  - {word} ({score:.4f})")
    else:
        print(results)

# Optional: Load master list to restrict "complex" words to those NOT in basic english?
# We can do that later if needed.
