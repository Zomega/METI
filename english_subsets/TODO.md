# Project TODOs

## Beam Search POS Validation
- [ ] **Real-time POS Checking**: Implement a mechanism to check the grammatical validity of each sentence as it is being formed during LLM generation.
- [ ] **Beam Search Backtracking**: If a generated word violates the expected Part-of-Speech for its context (e.g., "it turns" being incorrectly handled as a plural noun `TURN [PL]` instead of a verb), the system should backtrack to that stage of the beam search and sample an alternative.
- [ ] **Grammar-Aware Constraints**: Integrate the `tokenizer.py` POS-tagging logic directly into the `LogitsProcessor` to ensure that every word generated is not only in the core vocabulary but also grammatically appropriate for its position.

## Current Known Issues
- **Morphological Ambiguity**: Words like "turns" are occasionally misidentified (e.g., as `TURN [PL]`) when the context clearly requires a verb. This creates "logical noise" in the primitive mapping.
- **Contraction Handling**: Improve the tokenizer's handling of `'s`, `n't`, and other common English contractions.
