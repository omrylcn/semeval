"""Abstract base class for text encoders.

This module defines the interface that all encoder implementations must follow.
Encoders are responsible for converting text into dense vector representations.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for text encoders.

    This class defines the interface that all encoder implementations
    must follow. Encoders transform text into dense vector representations
    (embeddings) that can be used for semantic similarity and retrieval tasks.

    Notes
    -----
    Subclasses must implement:
    - encode(): Convert text to embeddings
    - get_embedding_dim(): Return embedding dimension
    - model_name (property): Return model identifier

    Examples
    --------
    >>> class MyEncoder(BaseEncoder):
    ...     def __init__(self, model):
    ...         self.model = model
    ...         self._name = "my-model"
    ...
    ...     def encode(self, texts, **kwargs):
    ...         return self.model.encode(texts)
    ...
    ...     def get_embedding_dim(self):
    ...         return 768
    ...
    ...     @property
    ...     def model_name(self):
    ...         return self._name
    """

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
        batch_size: int = 32,
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
        **kwargs : dict
            Additional encoder-specific parameters

        Returns
        -------
        embeddings : numpy.ndarray or torch.Tensor
            Embeddings of shape (n_texts, embedding_dim).
            If input is a single string, shape is (1, embedding_dim).

        Raises
        ------
        ValueError
            If input texts are empty or invalid format
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
        >>> # Multiple texts
        >>> embs = encoder.encode(["text1", "text2", "text3"])
        >>> embs.shape
        (3, 768)
        >>>
        >>> # Return as tensor
        >>> embs_tensor = encoder.encode(["text1", "text2"], convert_to_tensor=True)
        >>> type(embs_tensor)
        <class 'torch.Tensor'>
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings.

        Returns
        -------
        int
            Embedding dimension (e.g., 768 for BERT-base)

        Examples
        --------
        >>> encoder = SentenceTransformerEncoder("model-name")
        >>> encoder.get_embedding_dim()
        768
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name or identifier.

        Returns
        -------
        str
            Model name/identifier (e.g., "emrecan/bert-base-turkish-cased")

        Examples
        --------
        >>> encoder = SentenceTransformerEncoder("emrecan/bert-base-turkish")
        >>> encoder.model_name
        'emrecan/bert-base-turkish'
        """
        pass

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode texts in batches (default implementation).

        This is a convenience method that calls encode() with batching.
        Subclasses can override this for custom batching logic.

        Parameters
        ----------
        texts : list of str
            List of texts to encode
        batch_size : int, optional
            Batch size for encoding. Default is 32
        show_progress_bar : bool, optional
            If True, show progress bar. Default is True
        **kwargs : dict
            Additional parameters passed to encode()

        Returns
        -------
        numpy.ndarray
            Embeddings of shape (n_texts, embedding_dim)

        Examples
        --------
        >>> encoder = SentenceTransformerEncoder("model-name")
        >>> texts = ["text1", "text2", ..., "text1000"]
        >>> embeddings = encoder.encode_batch(texts, batch_size=64)
        >>> embeddings.shape
        (1000, 768)
        """
        return self.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            **kwargs
        )

    def __repr__(self) -> str:
        """Return string representation of encoder.

        Returns
        -------
        str
            String representation

        Examples
        --------
        >>> encoder = SentenceTransformerEncoder("model-name")
        >>> repr(encoder)
        'SentenceTransformerEncoder(model_name=model-name, embedding_dim=768)'
        """
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name}, "
            f"embedding_dim={self.get_embedding_dim()})"
        )
