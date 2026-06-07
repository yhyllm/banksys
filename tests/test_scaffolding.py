"""Test project scaffolding: imports and basic structure."""


def test_src_package_importable():
    """Verify the src package can be imported."""
    import src

    assert src is not None


def test_requirements_parseable():
    """Verify requirements.txt files exist and are non-empty."""
    from pathlib import Path

    proj_root = Path(__file__).resolve().parent.parent

    req = proj_root / "requirements.txt"
    assert req.exists(), "requirements.txt missing"
    assert len(req.read_text().strip().splitlines()) > 0, "requirements.txt is empty"

    req_dev = proj_root / "requirements-dev.txt"
    assert req_dev.exists(), "requirements-dev.txt missing"
    assert len(req_dev.read_text().strip().splitlines()) > 0, (
        "requirements-dev.txt is empty"
    )


def test_core_deps_importable():
    """Verify core production dependencies can be imported."""
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    import sklearn  # noqa: F401
    import pandas  # noqa: F401
    import numpy  # noqa: F401
    import joblib  # noqa: F401


def test_dev_deps_importable():
    """Verify dev dependencies are available."""
    import pytest  # noqa: F401
    import ruff  # noqa: F401
    import httpx  # noqa: F401


def test_dockerfile_exists():
    """Verify Dockerfile exists and contains expected instructions."""
    from pathlib import Path

    dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile missing"

    content = dockerfile.read_text()
    assert "FROM python" in content
    assert "COPY requirements.txt" in content
    assert "uvicorn" in content


def test_models_dir_is_gitignored():
    """Verify models/ is in .gitignore so artifacts stay out of git."""
    from pathlib import Path

    proj_root = Path(__file__).resolve().parent.parent
    gitignore = proj_root / ".gitignore"
    assert gitignore.exists(), ".gitignore missing"

    content = gitignore.read_text()
    assert "models/" in content, "models/ not found in .gitignore"
