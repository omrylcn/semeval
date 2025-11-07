"""Text encoder implementations.

This package provides various encoder implementations for converting
text into dense vector representations.

Available Encoders
------------------
SentenceTransformerEncoder
    Wrapper for Sentence Transformers library models
HuggingFaceEncoder
    Wrapper for raw HuggingFace transformer models with mean pooling

Examples
--------
>>> from semeval.core.encoders import SentenceTransformerEncoder
>>> encoder = SentenceTransformerEncoder("model-name")
>>> embeddings = encoder.encode(["text1", "text2"])
"""

from .sentence_transformer_encoder import SentenceTransformerEncoder
from .huggingface_encoder import HuggingFaceEncoder

__all__ = [
    'SentenceTransformerEncoder',
    'HuggingFaceEncoder',
]
