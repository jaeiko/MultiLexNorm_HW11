"""ByT5 Sequence-to-Seq (Seq2Seq) Baseline Corrector.

This corrector leverages the pre-trained `google/byt5-small` model to perform
character-level sequence-to-sequence translation (lexical normalization) on
individual tokens. It encapsulates input tokens with tag markers to guide predictions.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ByT5Baseline:
    """Seq2Seq token correction baseline model using pre-trained ByT5 transformers.

    Attributes:
        tokenizer (AutoTokenizer): Pre-trained ByT5 tokenizer instance.
        model (AutoModelForSeq2SeqLM): Pre-trained Seq2Seq LM instance.
        open_tag (str): Leading sentinel tag prepended to input words.
        close_tag (str): Trailing sentinel tag appended to input words.
    """

    def __init__(self, model_checkpoint: str = "google/byt5-small") -> None:
        """Initializes the ByT5 Baseline Corrector model.

        Args:
            model_checkpoint: Pre-trained transformer identifier path or name.
        """
        print(f"[System] Loading ByT5 model checkpoint '{model_checkpoint}' (this may take some time)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.model.eval()
        
        self.open_tag: str = "<w>"
        self.close_tag: str = "</w>"

    def predict(self, sentence_tokens: List[str]) -> List[str]:
        """Predicts corrected tokens by translating each token independently through ByT5.

        Args:
            sentence_tokens: List of original token strings.

        Returns:
            List[str]: Translated/normalized token strings.
        """
        corrected_tokens: List[str] = []
        
        with torch.no_grad():
            for word in sentence_tokens:
                # Wrap token in tagging sentinels
                prompt = f"{self.open_tag}{word}{self.close_tag}"
                
                # Perform byte-level tokenization and inference
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(**inputs, max_new_tokens=20)
                
                # Decode predictions and append to output list
                corrected_word = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                corrected_tokens.append(corrected_word)
                
        return corrected_tokens