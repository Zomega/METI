# Project TODOs

## Beam Search POS Validation
- [ ] **Real-time POS Checking**: Implement a mechanism to check the grammatical validity of each sentence as it is being formed during LLM generation.
- [ ] **Beam Search Backtracking**: If a generated word violates the expected Part-of-Speech for its context (e.g., "it turns" being incorrectly handled as a plural noun `TURN [PL]` instead of a verb), the system should backtrack to that stage of the beam search and sample an alternative.
- [ ] **Grammar-Aware Constraints**: Integrate the `tokenizer.py` POS-tagging logic directly into the `LogitsProcessor` to ensure that every word generated is not only in the core vocabulary but also grammatically appropriate for its position.

## Replicating TinyStories with RootMod
This section outlines the steps to train a tiny model (SLM) that natively speaks in "Primitive English."

### 1. Data Synthesis (The TinyStories Method)
- [ ] **Concept Triple Generation**: Randomly select 3 words from `vocab.csv` (1 Noun, 1 Verb, 1 Adjective) for each story.
- [ ] **Story Features**: Randomly assign features like Dialogue, Plot Twist, or Moral Value to prompts.
- [ ] **Prompt Template**: Use `generate.py` with: *"Write a short story using the words {NOUN}, {VERB}, and {ADJECTIVE}. The story must include {FEATURE} and use only the most simple words possible."*
- [ ] **Concept Definitions**: Define 10,000 "complex" words using only core primitives to build logical reasoning.

### 2. Primitive Tokenization
- [ ] **RootMod Transform**: Run all synthetic data through `tokenizer.tokenize()`.
- [ ] **Cleanup**: Strip any non-core tokens. Every training example must be pure `ROOT [MOD]` tokens.

### 3. Super-Tiny Model Architecture (ST-RootMod)
Based on *TinyStories* and *SuperTinyLanguageModels* research:
- [ ] **Extreme Parameter Target**: **500,000 parameters**.
- [ ] **Vocabulary Strategy (The "Primitive 1024")**: 
    - Consolidate the current ~2,200 words down to a strict **1,024 token set** based on Ogden's 850 Basic English words.
    - Every token in the 1024 set must be a high-value semantic prime.
- [ ] **Architectural Specs for 500k**:
    - **Vocabulary (V)**: 1,024.
    - **Hidden Dim (d)**: 128.
    - **Layers (L)**: 2-3.
    - **Heads (H)**: 4.
- [ ] **Parameter Optimization**:
    - **Weight Tying**: Tie **Input Embedding** to **Output Head** (Shared $V \times d = 131,072$ params).
    - **Rational FFN**: Use a narrower FFN ratio (e.g., 2x or 3x instead of 4x) to save parameters for more attention heads.

### 4. Training & Evaluation
- [ ] **Objective**: Causal Language Modeling (Next-Primitive Prediction).
- [ ] **Regularization**: Use high dropout (0.1 - 0.2) and weight decay, as tiny models overfit quickly on high-quality synthetic data.
- [ ] **Validation**: 
    - **Zero-Shot Completion**: Test logic like `THE SUN BE [ED] HOT SO THE ICE BECOME ...`
    - **Round-Trip Untokenization**: Verify generated thoughts are human-coherent via `untokenize()`.

## Current Known Issues
- **Morphological Ambiguity**: Words like "turns" are occasionally misidentified (e.g., as `TURN [PL]`) when the context clearly requires a verb.
- **Contraction Handling**: Improve the tokenizer's handling of `'s`, `n't`, and other common English contractions.
