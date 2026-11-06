"""Text embedding support used by the quantum prediction-head trainers.

This module is adapted from RelBench's ``examples/text_embedder.py`` and is
included here so that ``quantum_model`` does not depend on the examples
package at runtime. RelBench is distributed under the MIT License; see the
repository-level ``LICENSE`` file for the copyright and license notice.
"""

from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    """Encode text with SentenceTransformers' averaged GloVe model."""

    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        """Encode text after replacing missing or non-string values safely."""
        cleaned: List[str] = []
        for sentence in sentences:
            if sentence is None:
                cleaned.append("")
            elif isinstance(sentence, str):
                cleaned.append(sentence)
            else:
                try:
                    if isinstance(sentence, float) and sentence != sentence:
                        cleaned.append("")
                    else:
                        cleaned.append(str(sentence))
                except Exception:
                    cleaned.append("")

        return self.model.encode(cleaned, convert_to_tensor=True)
