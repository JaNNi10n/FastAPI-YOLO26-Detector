from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:  # the context manager runs the lifespan (model load + warm-up)
        yield c


@pytest.fixture(scope="session")
def sample_bytes() -> bytes:
    return (ASSETS / "elephant_zebra.jpg").read_bytes()
