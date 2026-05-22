"""Leave-As-Is (LAI) Baseline Corrector.

This baseline corrector performs no actual lexical normalization or spelling
correction. It returns the raw token sequence exactly as inputted, serving as
the fundamental lower-bound performance metric for evaluations.
"""

from typing import List


class LAIBaseline:
    """Standard Leave-As-Is baseline model that copies input tokens directly to output."""

    def __init__(self) -> None:
        """Initializes the LAIBaseline model and prints status logs."""
        print("[System] LAI Baseline initialized successfully.")

    def predict(self, sentence_tokens: List[str]) -> List[str]:
        """Predicts corrected tokens for a sentence (returns unchanged tokens).

        Args:
            sentence_tokens: List of original token strings.

        Returns:
            List[str]: Unchanged list of original token strings.
        """
        return list(sentence_tokens)