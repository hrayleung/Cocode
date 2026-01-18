"""Tests for embedding provider abstraction - TDD for common interface."""

import pytest
from unittest.mock import patch, MagicMock


class TestJinaProviderLateChunking:
    """Test that JinaProvider honors late chunking configuration."""

    @patch("src.embeddings.jina.get_embeddings_batch")
    def test_jina_provider_batch_uses_late_chunking_when_configured(self, mock_batch):
        """JinaProvider.get_embeddings_batch should pass use_late_chunking=True."""
        mock_batch.return_value = [[0.1] * 256, [0.2] * 256]
        
        from src.embeddings import JinaProvider
        provider = JinaProvider()
        provider.get_embeddings_batch(["text1", "text2"])
        
        mock_batch.assert_called_once_with(["text1", "text2"], use_late_chunking=True)


class TestEmbeddingProviderProtocol:
    """Test that embedding providers follow a common protocol."""

    def test_embedding_provider_protocol_exists(self):
        """EmbeddingProvider protocol should be importable from src.embeddings."""
        from src.embeddings import EmbeddingProvider
        
        # Should be a Protocol or ABC
        assert hasattr(EmbeddingProvider, 'get_embedding')
        assert hasattr(EmbeddingProvider, 'get_embeddings_batch')

    def test_openai_provider_implements_protocol(self):
        """OpenAI provider should implement EmbeddingProvider protocol."""
        from src.embeddings import EmbeddingProvider, OpenAIProvider
        
        # OpenAIProvider should be a subclass/implementer of EmbeddingProvider
        assert hasattr(OpenAIProvider, 'get_embedding')
        assert hasattr(OpenAIProvider, 'get_embeddings_batch')

    def test_jina_provider_implements_protocol(self):
        """Jina provider should implement EmbeddingProvider protocol."""
        from src.embeddings import EmbeddingProvider, JinaProvider
        
        assert hasattr(JinaProvider, 'get_embedding')
        assert hasattr(JinaProvider, 'get_embeddings_batch')

    def test_get_provider_returns_configured_provider(self):
        """get_provider() should return the configured embedding provider."""
        from src.embeddings import get_provider, EmbeddingProvider
        
        provider = get_provider()
        assert hasattr(provider, 'get_embedding')
        assert hasattr(provider, 'get_embeddings_batch')
