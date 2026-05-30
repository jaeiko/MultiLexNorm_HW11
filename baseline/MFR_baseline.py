"""Most Frequent Replacement (MFR) Baseline Corrector.

This corrector builds a statistical replacement mapping using training instances.
For each raw token, it identifies the most frequent normalized replacement token
from the training corpus and applies it. If no history exists, it falls back to Leave-As-Is.
"""

import os
import sys
from typing import Any, Dict, List


class MFRBaseline:
    """Statistical token correction baseline based on Most Frequent Replacement frequencies."""

    def __init__(self, train_data: List[Dict[str, List[str]]]) -> None:
        """Initializes the MFRBaseline and compiles frequency tables.

        Loads training dataset items and builds a vocabulary dictionary mapping
        raw tokens to their respective normalized candidates and counts.

        Args:
            train_data: List of dictionary records, each containing 'raw' and 'norm' token lists.
        """
        # Configure parent directory path to lookup dynamic utility modules if needed
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)

        try:
            from utils import counting
            self.counts: Dict[str, Dict[str, int]] = counting(train_data)
            print("[System] MFR Baseline initialized (built frequency map via counting utility).")
        except ImportError:
            print("[Warning] utils.py not found. Falling back to self-contained counting logic.")
            self.counts = {}
            for sample in train_data:
                raw_tokens = sample.get("raw", [])
                norm_tokens = sample.get("norm", [])
                if len(raw_tokens) == len(norm_tokens):
                    for r, n in zip(raw_tokens, norm_tokens):
                        if r not in self.counts:
                            self.counts[r] = {}
                        self.counts[r][n] = self.counts[r].get(n, 0) + 1

    def predict(self, sentence_tokens: List[str]) -> List[str]:
        """Predicts normalized token replacements using the compiled frequency maps.

        Args:
            sentence_tokens: List of original token strings.

        Returns:
            List[str]: Normalized token strings.
        """
        corrected_tokens: List[str] = []
        for word in sentence_tokens:
            if word in self.counts:
                # Select the candidate replacement token with highest count
                best_correction = max(self.counts[word], key=self.counts[word].get)
                corrected_tokens.append(best_correction)
            else:
                corrected_tokens.append(word)
        return corrected_tokens