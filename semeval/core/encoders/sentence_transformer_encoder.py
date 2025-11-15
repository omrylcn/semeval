"""Sentence Transformer encoder implementation.

This module provides an encoder wrapper for Sentence Transformers models.
"""

from typing import List, Optional, Union

import numpy as np
import torch

from ..base_encoder import BaseEncoder
from ..exceptions import EncoderError, ModelLoadError
from ..logging import get_logger, log_execution_time

logger = get_logger("semeval")


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
        trust_remote_code: bool = False,
    ):
        """Initialize Sentence Transformer encoder."""
        logger.info(f"Initializing SentenceTransformer encoder: {model_name_or_path}")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.error("sentence-transformers library not installed")
            raise ModelLoadError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers",
                model_name=model_name_or_path,
            ) from e

        self._model_name = model_name_or_path

        # Auto-detect device if not specified or "auto"
        if device is None or device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Auto-detected device: {device}")
        else:
            logger.info(f"Using specified device: {device}")

        try:
            with log_execution_time(logger, "model_loading"):
                self.model = SentenceTransformer(
                    model_name_or_path,
                    device=device,
                    trust_remote_code=trust_remote_code,
                )
            logger.info(
                f"Model loaded successfully: {model_name_or_path} "
                f"(device: {device}, dim: {self.model.get_sentence_embedding_dimension()})"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {model_name_or_path} - {str(e)}")
            raise ModelLoadError(
                f"Failed to load SentenceTransformer model: {str(e)}",
                model_name=model_name_or_path,
            ) from e

    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        **kwargs,
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
        EncoderError
            If texts is empty or encoding fails

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
                logger.error("Empty text provided for encoding")
                raise EncoderError(
                    "Input text cannot be empty",
                    model_name=self._model_name,
                    num_texts=1,
                )
            texts = [texts]
        elif not texts:
            logger.error("Empty texts list provided for encoding")
            raise EncoderError(
                "Input texts list cannot be empty",
                model_name=self._model_name,
                num_texts=0,
            )

        num_texts = len(texts)
        logger.debug(
            f"Encoding {num_texts} texts (batch_size={batch_size}, normalize={normalize_embeddings})"
        )

        try:
            with log_execution_time(logger, "text_encoding"):
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=convert_to_tensor,
                    show_progress_bar=show_progress_bar,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    **kwargs,
                )
            logger.info(
                f"Encoded {num_texts} texts successfully "
                f"(shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'})"
            )
            return embeddings
        except Exception as e:
            logger.error(
                f"Encoding failed for {num_texts} texts with model {self._model_name}: {str(e)}"
            )
            raise EncoderError(
                f"Failed to encode texts: {str(e)}",
                model_name=self._model_name,
                num_texts=num_texts,
                device=(
                    str(self.model.device) if hasattr(self.model, "device") else None
                ),
            ) from e

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
