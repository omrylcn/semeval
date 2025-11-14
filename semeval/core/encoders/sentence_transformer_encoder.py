"""Sentence Transformer encoder implementation.

This module provides an encoder wrapper for Sentence Transformers models.
"""

from typing import List, Optional, Union

import numpy as np
import torch

from ..base_encoder import BaseEncoder


class SentenceTransformerEncoder(BaseEncoder):
    """Encoder for Sentence Transformer models.

    This encoder wraps the Sentence Transformers library for easy
    integration with the evaluation framework.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model name or local path to model
    device : str, optional
        Device for computation ('cpu', 'cuda', 'mps').
        If None, automatically selects best available device
    trust_remote_code : bool, optional
        Whether to trust remote code when loading model.
        Default is False

    Attributes
    ----------
    model : SentenceTransformer
        Underlying Sentence Transformer model
    _model_name : str
        Model identifier

    Examples
    --------
    >>> encoder = SentenceTransformerEncoder(
    ...     "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    ... )
    >>> embeddings = encoder.encode(["test text 1", "test text 2"])
    >>> embeddings.shape
    (2, 768)
    >>>
    >>> # With specific device
    >>> encoder = SentenceTransformerEncoder("model-name", device="cuda")
    >>> encoder.model.device
    device(type='cuda', index=0)
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        trust_remote_code: bool = False
    ):
        """Initialize Sentence Transformer encoder."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

        self._model_name = model_name_or_path

        # Auto-detect device if not specified or "auto"
        if device is None or device == "auto":
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=trust_remote_code
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Encode text(s) into embeddings.

        Parameters
        ----------
        texts : str or list of str
            Single text or list of texts to encode
        convert_to_tensor : bool, optional
            If True, return torch.Tensor instead of numpy.ndarray.
            Default is False
        show_progress_bar : bool, optional
            If True, show progress bar during encoding.
            Default is False
        batch_size : int, optional
            Batch size for encoding. Default is 32
        normalize_embeddings : bool, optional
            If True, normalize embeddings to unit length.
            Default is False
        **kwargs : dict
            Additional parameters passed to SentenceTransformer.encode()

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Embeddings of shape (n_texts, embedding_dim)

        Raises
        ------
        ValueError
            If texts is empty
        RuntimeError
            If encoding fails

        Examples
        --------
        >>> encoder = SentenceTransformerEncoder("model-name")
        >>> # Single text
        >>> emb = encoder.encode("test text")
        >>> emb.shape
        (1, 768)
        >>>
        >>> # Multiple texts with progress bar
        >>> texts = ["text1", "text2", "text3"]
        >>> embs = encoder.encode(texts, show_progress_bar=True, batch_size=2)
        Batches: 100%|██████████| 2/2 [00:00<00:00, 10.25it/s]
        >>> embs.shape
        (3, 768)
        >>>
        >>> # Return as tensor
        >>> embs_tensor = encoder.encode(texts, convert_to_tensor=True)
        >>> type(embs_tensor)
        <class 'torch.Tensor'>
        """
        if isinstance(texts, str):
            if not texts.strip():
                raise ValueError("Input text cannot be empty")
            texts = [texts]
        elif not texts:
            raise ValueError("Input texts list cannot be empty")

        return self.model.encode(
            texts,
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings.

        Returns
        -------
        int
            Embedding dimension

        Examples
        --------
        >>> encoder = SentenceTransformerEncoder("model-name")
        >>> encoder.get_embedding_dim()
        768
        """
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        """Return model name or identifier.

        Returns
        -------
        str
            Model name/identifier

        Examples
        --------
        >>> encoder = SentenceTransformerEncoder("emrecan/bert-base-turkish")
        >>> encoder.model_name
        'emrecan/bert-base-turkish'
        """
        return self._model_name
