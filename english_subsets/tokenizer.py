import re
import pandas as pd
from morphology import apply_modifier, decompose_word
import nltk
from nltk.corpus import wordnet as wn

# Load the core vocabulary


def load_vocab():
    try:
        df = pd.read_csv("vocab.csv")
        return set(df["word"].dropna().str.lower().unique())
    except:
        return set()


VOCAB = load_vocab()


def get_wn_pos(nltk_tag):
    """Map NLTK POS tags to WordNet POS tags."""
    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def tokenize(sentence):
    """
    Decomposes a sentence into core words and modifiers (case-insensitive).
    """
    # Normalize punctuation spacing for cleaner splitting
    sentence = re.sub(r"([.,!?;:])", r" \1 ", sentence)
    words = sentence.split()

    # Get POS tags
    pos_tags = nltk.pos_tag(words)
    tokens = []

    PRONOUN_MAP = {"me": "i", "us": "we",
                   "him": "he", "her": "she", "them": "they"}

    for i, (w, tag) in enumerate(pos_tags):
        w_lower = w.lower()
        if not w_lower.isalpha():
            tokens.append(w)
            continue

        wn_pos = get_wn_pos(tag)

        # 1. Pronouns
        if w_lower in PRONOUN_MAP:
            tokens.append(PRONOUN_MAP[w_lower].upper())
            continue

        # 2. Irregular Verbs (BE)
        if w_lower in ["is", "am", "are", "was", "were", "be"]:
            root = wn.morphy(w_lower, wn.VERB)
            if root:
                mod = "[ED]" if w_lower in ["was", "were"] else ""
                tokens.append(f"{root.upper()} {mod}".strip())
                continue

        # 3. Specific Verb Tenses
        if wn_pos == wn.VERB:
            if tag in ['VBD', 'VBN']:
                root = wn.morphy(w_lower, wn.VERB)
                if root and root.lower() in VOCAB:
                    tokens.append(f"{root.upper()} [ED]")
                    continue
            elif tag == 'VBG':
                root = wn.morphy(w_lower, wn.VERB)
                if root and root.lower() in VOCAB:
                    tokens.append(f"{root.upper()} [ING]")
                    continue

        # 4. Try Decomposition
        interpretations = decompose_word(w_lower)
        found_decomp = False

        valid_interps = []
        for interp in interpretations:
            parts = interp.split()
            if len(parts) != 2:
                continue
            mod = parts[1] if not parts[0].startswith("[") else parts[0]
            is_valid = False
            if mod in ["[PL]", "[ER]"] and wn_pos == wn.NOUN:
                is_valid = True
            elif mod in ["[ING]", "[ED]"] and wn_pos in [wn.VERB, wn.ADJ, wn.NOUN]:
                is_valid = True
            elif mod in ["[CMP]", "[SUP]"] and wn_pos in [wn.ADJ, wn.ADJ_SAT]:
                is_valid = True
            elif mod == "[LY]" and wn_pos == wn.ADV:
                is_valid = True
            elif mod == "[UN]":
                is_valid = True
            if is_valid:
                valid_interps.append(interp)

        for interp in (valid_interps + interpretations):
            parts = interp.split()
            if len(parts) != 2:
                continue
            root = parts[1].lower() if parts[0].startswith(
                "[") else parts[0].lower()
            if root in VOCAB:
                tokens.append(interp.upper())
                found_decomp = True
                break

        if found_decomp:
            continue

        # 5. Direct match
        if w_lower in VOCAB:
            tokens.append(w_lower.upper())
            continue

        # 6. Lemmatization fallback
        root = wn.morphy(w_lower, wn_pos if wn_pos else wn.VERB)
        if root and root.lower() in VOCAB:
            tokens.append(root.upper())
        else:
            # If still nothing, it's a true UNK
            tokens.append(f"[UNK:{w.upper()}]")

    return " ".join(tokens)


def untokenize(token_string):
    """
    Reconstructs a sentence from tokens.
    """
    tokens = token_string.split()
    output = []

    OBJ_MAP = {"i": "me", "we": "us",
               "he": "him", "she": "her", "they": "them"}

    i = 0
    while i < len(tokens):
        t = tokens[i]
        t_lower = t.lower()

        if t.startswith("[UNK:"):
            output.append(t[5:-1].lower())
            i += 1
            continue

        if t == "[UN]" and (i + 1) < len(tokens):
            root = tokens[i+1]
            output.append(apply_modifier(root.lower(), "[UN]"))
            i += 2
            continue

        if (i + 1) < len(tokens) and tokens[i+1].startswith("[") and tokens[i+1].endswith("]"):
            mod = tokens[i+1]
            if mod != "[UN]":
                output.append(apply_modifier(t.lower(), mod))
                i += 2
                continue

        if t == "BE":
            prev = output[-1].lower() if output else ""
            if prev in ["i"]:
                output.append("am")
            elif prev in ["he", "she", "it", "this", "that"]:
                output.append("is")
            elif prev.endswith("s") or prev in ["we", "they", "children", "people"]:
                output.append("are")
            else:
                output.append("is")
            i += 1
            continue

        if t_lower in OBJ_MAP:
            is_start = (i == 0)
            nt = tokens[i+1].lower() if (i+1) < len(tokens) else ""
            is_v = (nt == "be" or nt.endswith("[ing]") or nt.endswith("[ed]"))
            if is_start or is_v:
                output.append(t_lower)
            else:
                output.append(OBJ_MAP[t_lower])
            i += 1
            continue

        output.append(t.lower() if t.isalpha() else t)
        i += 1

    result = " ".join(output)
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    if result:
        result = result[0].upper() + result[1:]
    return result
