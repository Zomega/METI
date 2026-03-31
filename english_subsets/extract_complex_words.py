from nltk.corpus import wordnet as wn
import pandas as pd

# Load core vocab to exclude
df = pd.read_csv("vocab.csv")
core_vocab = set(df["word"].dropna().str.lower().unique())

print("Extracting complex words from WordNet...")
all_words = set(list(wn.all_lemma_names()))

# Filter: 
# 1. Not in core vocab
# 2. Is alphabetic (no multi-word expressions like 'air_conditioner' for now)
# 3. Not too short
complex_words = [w for w in all_words if w.lower() not in core_vocab and w.isalpha() and len(w) > 3]

print(f"Total candidate complex words: {len(complex_words)}")

# Save the first 100k
target_list = sorted(complex_words)[:100000]
with open("complex_words.txt", "w") as f:
    f.write("\n".join(target_list))

print(f"Saved 100,000 words to 'complex_words.txt'")
