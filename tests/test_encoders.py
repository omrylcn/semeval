"""Tests for encoder implementations."""

import numpy as np
import pytest
import torch

from semeval.core.encoders import SentenceTransformerEncoder
from semeval.core.exceptions import EncoderError


class TestSentenceTransformerEncoder:
    """Tests for SentenceTransformerEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a small test encoder."""
        # Use a very small model for fast testing
        return SentenceTransformerEncoder(
            "sentence-transformers/paraphrase-albert-small-v2", device="cpu"
        )

    def test_encoder_initialization(self, encoder):
        """Test encoder can be initialized."""
        assert encoder is not None
        assert encoder.model is not None
        assert encoder.model_name == "sentence-transformers/paraphrase-albert-small-v2"

    def test_encoder_device_auto_detection(self):
        """Test device auto-detection."""
        encoder = SentenceTransformerEncoder(
            "sentence-transformers/paraphrase-albert-small-v2", device="auto"
        )
        # Should select a valid device (cpu, cuda, or mps)
        device = str(encoder.model.device)
        assert device in ["cpu", "cuda:0", "mps:0"] or device.startswith("cuda")

    def test_encode_single_text(self, encoder):
        """Test encoding a single text string."""
        text = "This is a test sentence"
        embedding = encoder.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1  # Single text
        assert embedding.shape[1] > 0  # Has embedding dimension

    def test_encode_multiple_texts(self, encoder):
        """Test encoding multiple texts."""
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = encoder.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3  # Three texts
        assert embeddings.shape[1] > 0  # Has embedding dimension

    def test_encode_with_tensor_output(self, encoder):
        """Test encoding with tensor output."""
        texts = ["Test sentence"]
        embeddings = encoder.encode(texts, convert_to_tensor=True)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 1

    def test_encode_with_normalization(self, encoder):
        """Test encoding with normalization."""
        texts = ["Test sentence"]
        embeddings = encoder.encode(texts, normalize_embeddings=True)

        # Check that embeddings are normalized (L2 norm ≈ 1)
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_encode_empty_string_raises_error(self, encoder):
        """Test that encoding empty string raises ValueError."""
        with pytest.raises(EncoderError, match="Input text cannot be empty"):
            encoder.encode("")

    def test_encode_empty_list_raises_error(self, encoder):
        """Test that encoding empty list raises ValueError."""
        with pytest.raises(EncoderError, match="Input texts list cannot be empty"):
            encoder.encode([])

    def test_encode_whitespace_only_raises_error(self, encoder):
        """Test that encoding whitespace-only string raises ValueError."""
        with pytest.raises(EncoderError, match="Input text cannot be empty"):
            encoder.encode("   ")

    def test_get_embedding_dim(self, encoder):
        """Test getting embedding dimension."""
        dim = encoder.get_embedding_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_embedding_consistency(self, encoder):
        """Test that same text produces same embedding."""
        text = "Consistent test sentence"
        emb1 = encoder.encode(text)
        emb2 = encoder.encode(text)

        np.testing.assert_array_almost_equal(emb1, emb2, decimal=6)

    def test_batch_encoding(self, encoder):
        """Test batch encoding with different batch sizes."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

        # Encode with different batch sizes
        emb_batch1 = encoder.encode(texts, batch_size=2)
        emb_batch2 = encoder.encode(texts, batch_size=5)

        # Results should be the same regardless of batch size
        np.testing.assert_array_almost_equal(emb_batch1, emb_batch2, decimal=5)

    def test_encoder_with_unicode_text(self, encoder):
        """Test encoding Unicode text (Turkish characters)."""
        # This tests SemEval's ability to handle Unicode, not model's Turkish support
        texts = [
            "Merhaba dünya",
            "Bu bir test cümlesidir",
            "Türkçe karakter desteği",
        ]
        embeddings = encoder.encode(texts)

        # SemEval should handle Unicode without crashing
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0
        # Embeddings should be valid numbers (not NaN or Inf)
        assert np.isfinite(embeddings).all()

    def test_encoder_missing_dependency_error(self):
        """Test that missing sentence-transformers raises ImportError."""
        # This test would need to mock the import to work properly
        # For now, we just verify the encoder requires sentence-transformers
        try:
            import sentence_transformers  # noqa: F401

            # If import succeeds, we can't test the error case
            pytest.skip("sentence-transformers is installed")
        except ImportError:
            with pytest.raises(
                ImportError, match="sentence-transformers is not installed"
            ):
                SentenceTransformerEncoder("any-model")


    def test_encoder_produces_valid_embeddings(self, encoder):
        """Test that encoder produces valid numerical embeddings."""
        texts = ["Text one", "Text two", "Text three"]
        embeddings = encoder.encode(texts)

        # All embeddings should be finite (not NaN or Inf)
        assert np.isfinite(embeddings).all()

        # Embeddings should have reasonable magnitudes (not all zeros)
        assert not np.allclose(embeddings, 0)

        # Each embedding should have non-zero norm
        for emb in embeddings:
            assert np.linalg.norm(emb) > 0


class TestHuggingFaceEncoder:
    """Tests for HuggingFaceEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a small test encoder."""
        # Use a very small model for fast testing
        from semeval.core.encoders import HuggingFaceEncoder

        return HuggingFaceEncoder(
            "prajjwal1/bert-tiny",  # Very small model (~17MB)
            device="cpu",
        )

    def test_encoder_initialization(self, encoder):
        """Test encoder can be initialized."""
        assert encoder is not None
        assert encoder.model is not None
        assert encoder.tokenizer is not None
        assert encoder.model_name == "prajjwal1/bert-tiny"

    def test_encoder_device_auto_detection(self):
        """Test device auto-detection."""
        from semeval.core.encoders import HuggingFaceEncoder

        encoder = HuggingFaceEncoder("prajjwal1/bert-tiny", device=None)
        # Should select a valid device (cpu, cuda, or mps)
        assert encoder.device in ["cpu", "cuda", "mps"]

    def test_encode_single_text(self, encoder):
        """Test encoding a single text string."""
        text = "This is a test sentence"
        embedding = encoder.encode(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1  # Single text
        assert embedding.shape[1] > 0  # Has embedding dimension

    def test_encode_multiple_texts(self, encoder):
        """Test encoding multiple texts."""
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = encoder.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3  # Three texts
        assert embeddings.shape[1] > 0  # Has embedding dimension

    def test_encode_with_tensor_output(self, encoder):
        """Test encoding with tensor output."""
        texts = ["Test sentence"]
        embeddings = encoder.encode(texts, convert_to_tensor=True)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 1

    def test_encode_with_normalization(self, encoder):
        """Test encoding with normalization."""
        texts = ["Test sentence"]
        embeddings = encoder.encode(texts, normalize_embeddings=True)

        # Check that embeddings are normalized (L2 norm ≈ 1)
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_encode_empty_string_raises_error(self, encoder):
        """Test that encoding empty string raises ValueError."""
        with pytest.raises(EncoderError, match="Input text cannot be empty"):
            encoder.encode("")

    def test_encode_empty_list_raises_error(self, encoder):
        """Test that encoding empty list raises ValueError."""
        with pytest.raises(EncoderError, match="Input texts list cannot be empty"):
            encoder.encode([])

    def test_encode_whitespace_only_raises_error(self, encoder):
        """Test that encoding whitespace-only string raises ValueError."""
        with pytest.raises(EncoderError, match="Input text cannot be empty"):
            encoder.encode("   ")

    def test_get_embedding_dim(self, encoder):
        """Test getting embedding dimension."""
        dim = encoder.get_embedding_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_embedding_consistency(self, encoder):
        """Test that same text produces same embedding."""
        text = "Consistent test sentence"
        emb1 = encoder.encode(text)
        emb2 = encoder.encode(text)

        np.testing.assert_array_almost_equal(emb1, emb2, decimal=6)

    def test_batch_encoding(self, encoder):
        """Test batch encoding with different batch sizes."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

        # Encode with different batch sizes
        emb_batch1 = encoder.encode(texts, batch_size=2)
        emb_batch2 = encoder.encode(texts, batch_size=5)

        # Results should be the same regardless of batch size
        np.testing.assert_array_almost_equal(emb_batch1, emb_batch2, decimal=5)

    def test_encoder_produces_valid_embeddings(self, encoder):
        """Test that encoder produces valid numerical embeddings."""
        texts = ["Text one", "Text two", "Text three"]
        embeddings = encoder.encode(texts)

        # All embeddings should be finite (not NaN or Inf)
        assert np.isfinite(embeddings).all()

        # Embeddings should have reasonable magnitudes (not all zeros)
        assert not np.allclose(embeddings, 0)

        # Each embedding should have non-zero norm
        for emb in embeddings:
            assert np.linalg.norm(emb) > 0

    def test_encoder_max_length_parameter(self):
        """Test custom max_length parameter."""
        from semeval.core.encoders import HuggingFaceEncoder

        encoder = HuggingFaceEncoder("prajjwal1/bert-tiny", max_length=256)
        assert encoder.max_length == 256

    def test_encoder_model_name_property(self, encoder):
        """Test model_name property."""
        assert encoder.model_name == "prajjwal1/bert-tiny"

    def test_mean_pooling_method(self, encoder):
        """Test mean pooling produces expected shape."""
        # Create dummy tensors
        batch_size, seq_len, hidden_size = 2, 10, encoder.get_embedding_dim()
        model_output = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        pooled = encoder._mean_pooling(model_output, attention_mask)

        assert pooled.shape == (batch_size, hidden_size)
        assert torch.isfinite(pooled).all()

    def test_encoder_missing_dependency_error(self):
        """Test that missing transformers raises ImportError."""
        # This test would need to mock the import to work properly
        # For now, we just verify the encoder requires transformers
        try:
            import transformers  # noqa: F401

            # If import succeeds, we can't test the error case
            pytest.skip("transformers is installed")
        except ImportError:
            from semeval.core.encoders import HuggingFaceEncoder

            with pytest.raises(
                ImportError, match="transformers is not installed"
            ):
                HuggingFaceEncoder("any-model")
