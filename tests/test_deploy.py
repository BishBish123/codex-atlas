"""Deploy-artifact tests (no Docker, no Fly.io required).

These tests validate that the deploy artifacts exist and are syntactically
valid. They run as part of ``make test`` / ``uv run pytest -m "not integration"``
because they carry no marker — they are fast, hermetic, and require no
external services.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

# Repo root — every path is relative to this.
REPO = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo(*parts: str) -> Path:
    return REPO.joinpath(*parts)


# ---------------------------------------------------------------------------
# 1. Dockerfile present and structurally sane
# ---------------------------------------------------------------------------


class TestDockerfile:
    def test_exists(self) -> None:
        assert _repo("Dockerfile").is_file(), "Dockerfile not found"

    def test_has_two_stages(self) -> None:
        text = _repo("Dockerfile").read_text()
        froms = re.findall(r"^FROM\s+\S+\s+AS\s+\S+", text, re.MULTILINE | re.IGNORECASE)
        assert len(froms) == 2, f"Expected 2 named FROM stages, got: {froms}"

    def test_builder_stage_uses_uv(self) -> None:
        text = _repo("Dockerfile").read_text()
        assert "uv" in text.lower(), "Dockerfile must reference uv"

    def test_runtime_stage_copies_from_builder(self) -> None:
        text = _repo("Dockerfile").read_text()
        assert "COPY --from=builder" in text, "Runtime stage must COPY --from=builder"

    def test_has_healthcheck(self) -> None:
        text = _repo("Dockerfile").read_text()
        assert "HEALTHCHECK" in text, "Dockerfile must define a HEALTHCHECK"

    def test_exposes_port_8000(self) -> None:
        text = _repo("Dockerfile").read_text()
        assert "EXPOSE 8000" in text, "Dockerfile must EXPOSE 8000"

    def test_non_root_user(self) -> None:
        text = _repo("Dockerfile").read_text()
        assert re.search(r"^USER\s+\S+", text, re.MULTILINE), (
            "Dockerfile must set a non-root USER"
        )

    def test_default_cmd_runs_atlas_mcp(self) -> None:
        text = _repo("Dockerfile").read_text()
        assert "atlas-mcp" in text, "CMD must invoke atlas-mcp"
        assert "--transport" in text, "CMD must include --transport flag"


# ---------------------------------------------------------------------------
# 2. fly.toml present and valid TOML
# ---------------------------------------------------------------------------


class TestFlyToml:
    def test_exists(self) -> None:
        assert _repo("fly.toml").is_file(), "fly.toml not found"

    def test_valid_toml(self) -> None:
        text = _repo("fly.toml").read_bytes()
        parsed = tomllib.loads(text.decode())
        assert "app" in parsed, "fly.toml must have an 'app' key"

    def test_has_http_service(self) -> None:
        text = _repo("fly.toml").read_bytes()
        parsed = tomllib.loads(text.decode())
        assert "http_service" in parsed, "fly.toml must define [http_service]"

    def test_internal_port_8000(self) -> None:
        text = _repo("fly.toml").read_bytes()
        parsed = tomllib.loads(text.decode())
        assert parsed["http_service"]["internal_port"] == 8000

    def test_has_vm_section(self) -> None:
        text = _repo("fly.toml").read_bytes()
        parsed = tomllib.loads(text.decode())
        assert "vm" in parsed, "fly.toml must have [[vm]]"

    def test_memory_bumped_for_indexing(self) -> None:
        text = _repo("fly.toml").read_bytes()
        parsed = tomllib.loads(text.decode())
        vm = parsed["vm"]
        # Accept list (TOML array-of-tables) or dict (inline table)
        vm_entry = vm[0] if isinstance(vm, list) else vm
        assert vm_entry.get("memory_mb", 0) >= 1024, (
            "fly.toml vm.memory_mb must be >= 1024 to handle indexing workload"
        )

    def test_primary_region_set(self) -> None:
        text = _repo("fly.toml").read_bytes()
        parsed = tomllib.loads(text.decode())
        assert parsed.get("primary_region"), "fly.toml must set primary_region"

    def test_sentinel_present(self) -> None:
        """fly.toml must use the REPLACE-ME sentinel so first-time deployers
        get a clear error rather than silently reusing a stale app name."""
        text = _repo("fly.toml").read_text()
        assert "REPLACE-ME" in text, (
            "fly.toml app name must contain REPLACE-ME sentinel "
            "(users replace it before deploying)"
        )


# ---------------------------------------------------------------------------
# 3. deploy.sh present and bash-syntax-clean
# ---------------------------------------------------------------------------


class TestDeployScript:
    def test_exists(self) -> None:
        assert _repo("scripts", "deploy.sh").is_file(), "scripts/deploy.sh not found"

    def test_is_executable(self) -> None:
        p = _repo("scripts", "deploy.sh")
        assert p.stat().st_mode & 0o111, "scripts/deploy.sh must be executable"

    def test_bash_syntax_clean(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(_repo("scripts", "deploy.sh"))],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"

    def test_checks_fly_on_path(self) -> None:
        text = _repo("scripts", "deploy.sh").read_text()
        assert "command -v fly" in text, "deploy.sh must check for fly CLI on PATH"

    def test_checks_fly_auth(self) -> None:
        text = _repo("scripts", "deploy.sh").read_text()
        assert "fly auth whoami" in text, "deploy.sh must verify fly auth"

    def test_runs_fly_deploy(self) -> None:
        text = _repo("scripts", "deploy.sh").read_text()
        assert "fly deploy" in text, "deploy.sh must call fly deploy"

    def test_rejects_replace_me_sentinel(self) -> None:
        text = _repo("scripts", "deploy.sh").read_text()
        assert "REPLACE-ME" in text, (
            "deploy.sh must detect and reject the REPLACE-ME sentinel app name"
        )

    def test_pushes_atlas_secrets(self) -> None:
        text = _repo("scripts", "deploy.sh").read_text()
        for secret in ("GROQ_API_KEY", "ATLAS_PG_DSN", "LANGFUSE_PUBLIC_KEY"):
            assert secret in text, f"deploy.sh must reference secret {secret}"


# ---------------------------------------------------------------------------
# 4. .dockerignore present
# ---------------------------------------------------------------------------


class TestDockerignore:
    def test_exists(self) -> None:
        assert _repo(".dockerignore").is_file(), ".dockerignore not found"

    def test_excludes_venv(self) -> None:
        text = _repo(".dockerignore").read_text()
        assert ".venv" in text, ".dockerignore must exclude .venv"

    def test_excludes_tests(self) -> None:
        text = _repo(".dockerignore").read_text()
        assert "tests" in text, ".dockerignore must exclude tests/"

    def test_excludes_evals(self) -> None:
        text = _repo(".dockerignore").read_text()
        assert "evals" in text, ".dockerignore must exclude evals/"

    def test_excludes_git(self) -> None:
        text = _repo(".dockerignore").read_text()
        assert ".git" in text, ".dockerignore must exclude .git"

    def test_excludes_examples(self) -> None:
        text = _repo(".dockerignore").read_text()
        assert "examples" in text, ".dockerignore must exclude examples/"


# ---------------------------------------------------------------------------
# 5. CI docker workflow present
# ---------------------------------------------------------------------------


class TestDockerWorkflow:
    def test_exists(self) -> None:
        assert _repo(".github", "workflows", "docker.yml").is_file(), (
            ".github/workflows/docker.yml not found"
        )

    def test_builds_dockerfile(self) -> None:
        text = _repo(".github", "workflows", "docker.yml").read_text()
        assert "docker build" in text, "docker.yml must run docker build"

    def test_validates_healthz(self) -> None:
        text = _repo(".github", "workflows", "docker.yml").read_text()
        # TCP probe or healthz — either is acceptable
        assert any(term in text for term in ("healthz", "nc -z", "port 8000")), (
            "docker.yml must validate the container port/health"
        )


# ---------------------------------------------------------------------------
# 6. README references the deploy section artifacts
# ---------------------------------------------------------------------------


class TestReadmeDeploySection:
    @pytest.fixture(autouse=True)
    def readme_text(self) -> None:  # type: ignore[return]
        self._text = _repo("README.md").read_text()

    def test_deploy_section_exists(self) -> None:
        assert "## Deploy" in self._text, "README must have a '## Deploy' section"

    def test_references_deploy_script(self) -> None:
        assert "deploy.sh" in self._text, "README Deploy section must mention deploy.sh"

    def test_references_fly_toml(self) -> None:
        assert "fly.toml" in self._text, "README Deploy section must mention fly.toml"

    def test_references_required_secrets(self) -> None:
        assert "ATLAS_PG_DSN" in self._text, (
            "README must list ATLAS_PG_DSN as a required secret"
        )

    def test_cost_estimate_mentioned(self) -> None:
        assert any(
            phrase in self._text for phrase in ["$3", "$5", "month", "cost", "/mo"]
        ), "README Deploy section should include a cost estimate"
