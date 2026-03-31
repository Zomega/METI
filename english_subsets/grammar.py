import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import random
from tokenizer import untokenize

# Load the core vocabulary and categorize by POS


def get_vocab_by_pos():
    df = pd.read_csv("vocab.csv")
    words = df["word"].dropna().str.lower().unique()

    pos_map = {
        "N": [], "V": [], "ADJ": [], "ADV": [], "DET": ["the", "a", "an", "this", "that", "all", "some"],
        "PRON_SUBJ": ["i", "he", "she", "it", "we", "they"],
        "PRON_OBJ": ["me", "him", "her", "it", "us", "them"],
        "CONJ": ["and", "but", "or", "because"],
        "PREP": ["with", "at", "from", "into", "for", "on", "by", "than"]
    }

    for w in words:
        syns = wn.synsets(w)
        if not syns:
            continue
        best_pos = syns[0].pos()
        if best_pos == 'n':
            pos_map["N"].append(w.upper())
        elif best_pos == 'v':
            pos_map["V"].append(w.upper())
        elif best_pos in ['a', 's']:
            pos_map["ADJ"].append(w.upper())
        elif best_pos == 'r':
            pos_map["ADV"].append(w.upper())

    return pos_map


VOCAB = get_vocab_by_pos()


def generate_random_tokens():
    """Simple random generator following the CFG logic."""
    def get_np(is_subject=True):
        if random.random() < 0.3:
            p_list = VOCAB["PRON_SUBJ"] if is_subject else VOCAB["PRON_OBJ"]
            return [random.choice(p_list).upper()]
        np = []
        if random.random() < 0.7:
            np.append(random.choice(VOCAB["DET"]).upper())
        if random.random() < 0.4:
            np.append(random.choice(VOCAB["ADJ"]).upper())
        np.append(random.choice(VOCAB["N"]).upper())
        if random.random() < 0.3:
            np.append("[PL]")
        return np

    def get_vp():
        vp = []
        if random.random() < 0.3:
            vp.append("BE")
            if random.random() < 0.4:
                vp.append("[ED]")
            if random.random() < 0.5:
                vp.extend(get_np(is_subject=False))
            else:
                vp.append(random.choice(VOCAB["ADJ"]).upper())
        else:
            vp.append(random.choice(VOCAB["V"]).upper())
            if random.random() < 0.4:
                vp.append(random.choice(["[ED]", "[ING]"]))
            if random.random() < 0.6:
                vp.extend(get_np(is_subject=False))
        return vp

    tokens = get_np(is_subject=True) + get_vp()
    return " ".join(tokens)


if __name__ == "__main__":
    print("Generated Random Tokenized Sentences:")
    print("-" * 40)
    for _ in range(5):
        tokens = generate_random_tokens()
        print(f"Tokens: {tokens}")
        print(f"Human:  {untokenize(tokens)}")
        print()
