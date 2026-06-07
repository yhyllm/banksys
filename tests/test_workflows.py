"""Test CI/CD workflow files exist and have correct structure."""

from pathlib import Path


def _workflow_dir():
    return Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _read_workflow(name):
    """Read a workflow file as raw text."""
    return (_workflow_dir() / name).read_text()


def test_ci_workflow_exists():
    """CI workflow file must exist."""
    assert (_workflow_dir() / "ci.yml").exists(), "ci.yml missing"


def test_ci_triggers_on_pr():
    """CI must trigger on pull_request to main."""
    text = _read_workflow("ci.yml")
    assert "pull_request:" in text, "CI must trigger on pull_request"


def test_ci_includes_ruff_format():
    """CI must run ruff format check."""
    assert "ruff format --check" in _read_workflow("ci.yml")


def test_ci_includes_ruff_check():
    """CI must run ruff lint."""
    assert "ruff check" in _read_workflow("ci.yml")


def test_ci_includes_pytest_cov():
    """CI must run pytest with coverage threshold."""
    text = _read_workflow("ci.yml")
    assert "pytest" in text, "CI must run pytest"
    assert "cov-fail-under=80" in text, "CI must enforce coverage >= 80%"


def test_ci_includes_docker_build():
    """CI must build Docker image."""
    assert "docker build" in _read_workflow("ci.yml")


def test_cd_workflow_exists():
    """CD workflow file must exist."""
    assert (_workflow_dir() / "cd.yml").exists(), "cd.yml missing"


def test_cd_triggers_on_push_main():
    """CD must trigger on push to main."""
    text = _read_workflow("cd.yml")
    assert "branches: [main]" in text, "CD must trigger on push to main"


def test_cd_uses_secrets():
    """CD must reference SSH secrets, not hardcoded values."""
    cd = _workflow_dir() / "cd.yml"
    text = cd.read_text()
    assert "secrets.SSH_HOST" in text, "CD must use SSH_HOST from Secrets"
    assert "secrets.SSH_USER" in text, "CD must use SSH_USER from Secrets"
    assert "secrets.SSH_PRIVATE_KEY" in text, "CD must use SSH_PRIVATE_KEY from Secrets"


def test_cd_includes_health_check():
    """CD must run a health check after deployment."""
    cd = _workflow_dir() / "cd.yml"
    text = cd.read_text()
    assert "/health" in text, "CD must run health check"
    assert "curl" in text, "CD must use curl for health check"
