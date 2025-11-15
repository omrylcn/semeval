"""HuggingFace transformer encoder implementation.

This module provides an encoder for raw HuggingFace transformer models
with mean pooling strategy.
"""

from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ..base_encoder import BaseEncoder
from ..exceptions import EncoderError, ModelLoadError
from ..logging import get_logger, log_execution_time

logger = get_logger("semeval")


class HuggingFaceEncoder(BaseEncoder):
    """Encoder for raw HuggingFace transformer models.

    This encoder wraps HuggingFace transformers with mean pooling
    for sentence embeddings.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path
    device : str, optional
        Device for computation ('cpu', 'cuda', 'mps').
        If None, automatically selects best available device
    trust_remote_code : bool, optional
        Whether to trust remote code when loading model.
        Default is False
    max_length : int, optional
        Maximum sequence length for tokenization.
        Default is 512

    Attributes
    ----------
    tokenizer : AutoTokenizer
        HuggingFace tokenizer
    model : AutoModel
        HuggingFace model
    device : str
        Computation device
    max_length : int
        Maximum sequence length
    _model_name : str
        Model identifier

    Examples
    --------
    >>> encoder = HuggingFaceEncoder("dbmdz/bert-base-turkish-cased")
    >>> embeddings = encoder.encode(["test text 1", "test text 2"])
    >>> embeddings.shape
    (2, 768)
    >>>
    >>> # With GPU
    >>> encoder = HuggingFaceEncoder("model-name", device="cuda")
    >>> encoder.device
    'cuda'
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        max_length: int = 512,
    ):
        """Initialize HuggingFace encoder."""
        logger.info(f"Initializing HuggingFace encoder: {model_name}")

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            logger.error("transformers library not installed")
            raise ModelLoadError(
                "transformers is not installed. "
                "Install it with: pip install transformers",
                model_name=model_name,
            ) from e

        self._model_name = model_name
        self.max_length = max_length

        # Load tokenizer and model
        try:
            with log_execution_time(logger, "tokenizer_loading"):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
            logger.debug(f"Tokenizer loaded: {model_name}")

            with log_execution_time(logger, "model_loading"):
                self.model = AutoModel.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
            logger.debug(f"Model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {model_name} - {str(e)}")
            raise ModelLoadError(
                f"Failed to load HuggingFace model or tokenizer: {str(e)}",
                model_name=model_name,
            ) from e

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Auto-detected device: {device}")
        else:
            logger.info(f"Using specified device: {device}")

        self.device = device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        logger.info(
            f"Model initialized successfully: {model_name} "
            f"(device: {device}, dim: {self.model.config.hidden_size}, max_length: {max_length})"
        )

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to model output.

        Parameters
        ----------
        model_output : torch.Tensor
            Model output (last hidden state)
        attention_mask : torch.Tensor
            Attention mask

        Returns
        -------
        torch.Tensor
            Pooled embeddings

        Notes
        -----
        This implementation uses attention mask to avoid pooling
        over padding tokens.
        """
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        )

        # Sum embeddings (masking padding)
        sum_embeddings = torch.sum(model_output * input_mask_expanded, dim=1)

        # Sum attention mask (to get actual sequence lengths)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Mean pooling
        return sum_embeddings / sum_mask

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
            Additional parameters (not used, kept for compatibility)

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
        >>> encoder = HuggingFaceEncoder("dbmdz/bert-base-turkish-cased")
        >>> # Single text
        >>> emb = encoder.encode("test text")
        >>> emb.shape
        (1, 768)
        >>>
        >>> # Multiple texts with batching
        >>> texts = ["text1", "text2", "text3", "text4"]
        >>> embs = encoder.encode(texts, batch_size=2, show_progress_bar=True)
        >>> embs.shape
        (4, 768)
        """
        if isinstance(texts, str):
            if not texts.strip():
                logger.error("Empty text provided for encoding")
                raise EncoderError(
                    "Input text cannot be empty",
                    model_name=self._model_name,
                    num_texts=1,
                    device=self.device,
                )
            texts = [texts]
        elif not texts:
            logger.error("Empty texts list provided for encoding")
            raise EncoderError(
                "Input texts list cannot be empty",
                model_name=self._model_name,
                num_texts=0,
                device=self.device,
            )

        num_texts = len(texts)
        logger.debug(
            f"Encoding {num_texts} texts (batch_size={batch_size}, normalize={normalize_embeddings})"
        )

        try:
            all_embeddings = []

            # Process in batches
            num_batches = (len(texts) + batch_size - 1) // batch_size

            iterator = range(0, len(texts), batch_size)
            if show_progress_bar:
                iterator = tqdm(iterator, total=num_batches, desc="Encoding")

            with log_execution_time(logger, "text_encoding"):
                for i in iterator:
                    batch_texts = texts[i : i + batch_size]

                    # Tokenize
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)

                    # Encode
                    with torch.no_grad():
                        outputs = self.model(**encoded)

                    # Mean pooling
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state, encoded["attention_mask"]
                    )

                    # Normalize if requested
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    all_embeddings.append(embeddings.cpu())

            # Concatenate all batches
            final_embeddings = torch.cat(all_embeddings, dim=0)

            logger.info(
                f"Encoded {num_texts} texts successfully (shape: {final_embeddings.shape})"
            )

            if convert_to_tensor:
                return final_embeddings

            return final_embeddings.numpy()
        except Exception as e:
            logger.error(
                f"Encoding failed for {num_texts} texts with model {self._model_name}: {str(e)}"
            )
            raise EncoderError(
                f"Failed to encode texts: {str(e)}",
                model_name=self._model_name,
                num_texts=num_texts,
                device=self.device,
            ) from e

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings.

        Returns
        -------
        int
            Embedding dimension

        Examples
        --------
        >>> encoder = HuggingFaceEncoder("dbmdz/bert-base-turkish-cased")
        >>> encoder.get_embedding_dim()
        768
        """
        return self.model.config.hidden_size

    @property
    def model_name(self) -> str:
        """Return model name or identifier.

        Returns
        -------
        str
            Model name/identifier

        Examples
        --------
        >>> encoder = HuggingFaceEncoder("dbmdz/bert-base-turkish-cased")
        >>> encoder.model_name
        'dbmdz/bert-base-turkish-cased'
        """
        return self._model_name
