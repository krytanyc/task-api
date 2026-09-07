import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Avoid starting Qdrant, downloading embeddings, or calling an LLM in unit tests.
# The production configuration is intentionally still validated at import time.
os.environ.setdefault("GROQ_API_KEY", "test-groq-api-key")
from app.main import app


@pytest.fixture(autouse=True)
def mock_rag_chain():
    with patch("app.main.build_rag_chain", return_value=(MagicMock(), MagicMock())):
        yield

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
