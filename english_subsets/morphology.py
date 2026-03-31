import inflect
from nltk.corpus import wordnet as wn
import re
from pyinflect import getInflection
from difflib import SequenceMatcher

# Initialize inflect engine for pluralization
p = inflect.engine()


def get_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_derivations(word, pos=None, target_pos=None):
    synsets = wn.synsets(word, pos=pos)
    results = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            for related in lemma.derivationally_related_forms():
                if target_pos is None or related.synset().pos() == target_pos:
                    results.add(related.name())
    return list(results)


def is_pos(word, pos_list):
    """Check if a word can function as one of the target POS types."""
    synsets = wn.synsets(word)
    if not synsets:
        return True
    return any(s.pos() in pos_list for s in synsets)


def apply_modifier(word, mod):
    """Applies a morphological modifier to a root word."""
    word = word.lower()
    mod = mod.upper()

    if mod == "[PL]":
        if not is_pos(word, [wn.NOUN]):
            return word
        infl = getInflection(word, tag='NNS')
        if infl:
            return infl[0]
        res = p.plural(word)
        return res if res else word + "s"

    if mod == "[ER]":
        if not is_pos(word, [wn.VERB]):
            return word
        derivs = get_derivations(word, pos=wn.VERB, target_pos=wn.NOUN)
        matches = [d for d in derivs if d.endswith(
            ('er', 'or')) and d.startswith(word[:3])]
        if matches:
            return sorted(matches, key=len)[0]
        if word.endswith('e'):
            return word + "r"
        if word.endswith('y') and word[-2] not in 'aeiou':
            return word[:-1] + "ier"
        if re.search(r"([^aeiou][aeiou][bcdfghjklmnpqrstvwxyz])$", word):
            return word + word[-1] + "er"
        return word + "er"

    if mod == "[ING]":
        if not is_pos(word, [wn.VERB]):
            return word
        infl = getInflection(word, tag='VBG')
        if infl:
            return infl[0]
        return word + "ing"

    if mod == "[ED]":
        if not is_pos(word, [wn.VERB]):
            return word
        infl = getInflection(word, tag='VBD')
        if infl:
            return infl[0]
        return word + "ed"

    if mod == "[LY]":
        if not is_pos(word, [wn.ADJ, wn.ADJ_SAT]):
            return word
        derivs = get_derivations(word, pos=wn.ADJ, target_pos=wn.ADV)
        if derivs:
            return sorted(derivs, key=len)[0]
        if word.endswith('y'):
            return word[:-1] + "ily"
        if word.endswith('le'):
            return word[:-1] + "y"
        return word + "ly"

    if mod == "[UN]":
        if not is_pos(word, [wn.ADJ, wn.ADJ_SAT, wn.VERB, wn.NOUN]):
            return word
        un_word = "un" + word
        if wn.synsets(un_word):
            return un_word
        synsets = wn.synsets(word)
        for syn in synsets:
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    n = ant.name().lower()
                    if n.startswith(('un', 'in', 'im', 'dis', 'ir', 'non')):
                        return n
        return un_word

    if mod == "[CMP]":
        if not is_pos(word, [wn.ADJ, wn.ADJ_SAT]):
            return word
        infl = getInflection(word, tag='JJR')
        if infl:
            return infl[0]
        return word + "er"

    if mod == "[SUP]":
        if not is_pos(word, [wn.ADJ, wn.ADJ_SAT]):
            return word
        infl = getInflection(word, tag='JJS')
        if infl:
            return infl[0]
        return word + "est"

    return word


def aggressive_search(word, target_pos=None):
    """Deep search in WordNet for related roots."""
    roots = set()
    synsets = wn.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            for related in lemma.derivationally_related_forms():
                if target_pos is None or related.synset().pos() == target_pos:
                    roots.add(related.name())
            for pert in lemma.pertainyms():
                if target_pos is None or pert.synset().pos() == target_pos:
                    roots.add(pert.name())
    return list(roots)


def get_best_root(derived_word, target_pos=wn.VERB):
    """Find the root word using aggressive search and fuzzy ranking."""
    results = aggressive_search(derived_word, target_pos)
    if not results:
        return None
    valid_results = [
        r for r in results if derived_word.lower().startswith(r.lower()[:2])]
    if not valid_results:
        return None
    scored = []
    for r in valid_results:
        score = get_similarity(derived_word.lower(), r.lower())
        scored.append((r, score))
    scored.sort(key=lambda x: (-x[1], len(x[0])))
    return scored[0][0].upper()


def is_semantically_related(word, root, mod):
    """Check if the word and proposed root are related."""
    word, root = word.lower(), root.lower()
    if mod in ["[PL]", "[CMP]", "[SUP]"]:
        return True
    if mod == "[UN]":
        if word.startswith("un") and word[2:] == root:
            return True
        synsets = wn.synsets(word)
        for syn in synsets:
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    if ant.name().lower() == root:
                        return True
        return False
    synsets = wn.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            for related in lemma.derivationally_related_forms():
                if related.name().lower() == root:
                    return True
            for pert in lemma.pertainyms():
                if pert.name().lower() == root:
                    return True
    if mod in ["[ING]", "[ED]", "[ER]"]:
        morph_root = wn.morphy(word, wn.VERB)
        if morph_root and morph_root.lower() == root:
            return True
    return False


def decompose_word(word):
    """Decomposes a word into potential ROOT [MOD] interpretations."""
    word_lower = word.strip().lower()
    interpretations = []

    def add_interp(root, mod, is_prefix=False):
        if not is_semantically_related(word_lower, root.lower(), mod):
            return
        if is_prefix:
            interpretations.append(f"{mod} {root.upper()}")
        else:
            interpretations.append(f"{root.upper()} {mod}")

    # 1. Plural
    singular = p.singular_noun(word_lower)
    if singular and singular != word_lower:
        add_interp(singular, "[PL]")

    # 2. Negation
    if word_lower.startswith(('un', 'in', 'im', 'dis', 'ir', 'non')):
        for pre in ['un', 'in', 'im', 'dis', 'ir', 'non']:
            if word_lower.startswith(pre):
                root = word_lower[len(pre):]
                if wn.synsets(root):
                    add_interp(root, "[UN]", is_prefix=True)

    # 3. Adj Modifiers
    if word_lower.endswith(('er', 'est')):
        root = wn.morphy(word_lower, wn.ADJ)
        if not root or root == word_lower:
            pot = None
            if word_lower.endswith('iest'):
                pot = word_lower[:-4] + "y"
            elif word_lower.endswith('est'):
                pot = word_lower[:-3]
            elif word_lower.endswith('ier'):
                pot = word_lower[:-3] + "y"
            elif word_lower.endswith('er'):
                pot = word_lower[:-2]
            if pot:
                if len(pot) > 1 and pot[-1] == pot[-2]:
                    if wn.synsets(pot[:-1], pos=wn.ADJ):
                        pot = pot[:-1]
                if wn.synsets(pot, pos=wn.ADJ):
                    root = pot
        if root and root != word_lower:
            add_interp(root, "[SUP]" if word_lower.endswith(
                "est") else "[CMP]")

    # 4. Verb Modifiers
    if word_lower.endswith('ing'):
        root = get_best_root(word_lower, target_pos=wn.VERB)
        if root:
            add_interp(root, "[ING]")
        else:
            pot = word_lower[:-3]
            if len(word_lower) > 5 and word_lower[-4] == word_lower[-5]:
                pot = word_lower[:-4]
            if wn.synsets(pot, pos=wn.VERB):
                add_interp(pot, "[ING]")

    if word_lower.endswith(('er', 'or')):
        root = get_best_root(word_lower, target_pos=wn.VERB)
        if root:
            add_interp(root, "[ER]")
        else:
            pot = word_lower[:-2]
            if wn.synsets(pot, pos=wn.VERB):
                add_interp(pot, "[ER]")

    if word_lower.endswith('ed'):
        root = wn.morphy(word_lower, wn.VERB)
        if root and root != word_lower:
            add_interp(root, "[ED]")
        else:
            add_interp(word_lower[:-2], "[ED]")

    if word_lower.endswith('ly'):
        root = get_best_root(word_lower, target_pos=wn.ADJ)
        if root:
            add_interp(root, "[LY]")
        else:
            if word_lower.endswith('ily'):
                add_interp(word_lower[:-3] + "y", "[LY]")
            else:
                add_interp(word_lower[:-2], "[LY]")

    return list(set(interpretations)) if interpretations else [word]
