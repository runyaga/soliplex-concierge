"""Fetch team skill mirrors from Gitea, deploy revisions and reload skills."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import pathlib
import subprocess
import time

from soliplex_concierge.provisioner import soliplex_api
from soliplex_concierge.skills_deployer import deploy
from soliplex_concierge.skills_deployer.deploy import Outcome
from soliplex_concierge.skills_deployer.state import DeployState


@dataclasses.dataclass(frozen=True)
class ReconcilerConfig:
    """Describe the team remotes and backend reload settings."""

    deploy_config: pathlib.Path
    remotes: dict[str, str]
    backend_url: str
    secret_file: pathlib.Path
    service_subject: str = "soliplex-provisioner"
    interval: float = 60.0
    git_timeout: float = 120.0

    def __post_init__(self) -> None:
        """Require a finite positive git timeout."""
        if (
            isinstance(self.git_timeout, bool)
            or not isinstance(self.git_timeout, (int, float))
            or not math.isfinite(self.git_timeout)
            or self.git_timeout <= 0
        ):
            message = "git_timeout must be a finite positive number"
            raise ValueError(message)

    @classmethod
    def from_json(cls, path: pathlib.Path) -> ReconcilerConfig:
        """Read reconciler configuration from a JSON file."""
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(
            deploy_config=pathlib.Path(data["deploy_config"]),
            remotes=dict(data["remotes"]),
            backend_url=data["backend_url"],
            secret_file=pathlib.Path(data["secret_file"]),
            service_subject=data.get(
                "service_subject", "soliplex-provisioner"
            ),
            interval=data.get("interval", 60.0),
            git_timeout=data.get("git_timeout", 120.0),
        )


def _git(args: list[str], timeout: float) -> str:
    """Run a git command with a timeout and return its stdout."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        check=True,
        timeout=timeout,
    ).stdout.decode()


def _config(repo: pathlib.Path, key: str, timeout: float) -> list[str]:
    """Read all local configuration values from a repository."""
    return (
        _git(
            ["-C", str(repo), "config", "--local", "--null", "--get-all", key],
            timeout,
        )
        .removesuffix("\0")
        .split("\0")
    )


def _sync_mirror(repo: pathlib.Path, remote: str, timeout: float) -> bool:
    """Clone a missing mirror or update a verified existing mirror."""
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--mirror", remote, str(repo)], timeout)
        return True
    if (
        _config(repo, "core.bare", timeout) != ["true"]
        or _config(repo, "remote.origin.url", timeout) != [remote]
        or _config(repo, "remote.origin.fetch", timeout) != ["+refs/*:refs/*"]
    ):
        return False
    _git(["-C", str(repo), "fetch", "--prune", "origin"], timeout)
    return True


def _reload_skills(config: ReconcilerConfig) -> None:
    """Mint an admin service token and ask the backend to reload skills."""
    secret = soliplex_api.read_secret(config.secret_file)
    token = soliplex_api.mint_service_token(
        secret, {"sub": config.service_subject}
    )
    soliplex_api.call(
        config.backend_url,
        "POST",
        "/v1/installation/reload/skills",
        token=token,
    )


def _reload_problem(error: Exception) -> str:
    """Format a fixed reload failure message with any HTTP status."""
    code = getattr(error, "code", None) or getattr(
        error.__cause__, "code", None
    )
    if isinstance(code, int):
        return f"skill reload failed (HTTP {code})"
    return "skill reload failed"


def run_once(config: ReconcilerConfig) -> list[Outcome]:
    """Fetch team mirrors, deploy each fetched team and reload the backend."""
    deploy_config = deploy.DeployConfig.from_json(config.deploy_config)
    outcomes: dict[str, Outcome] = {}
    fetched: list[str] = []
    for team in sorted(config.remotes):
        try:
            synced = _sync_mirror(
                deploy_config.repos[team],
                config.remotes[team],
                config.git_timeout,
            )
        except subprocess.CalledProcessError as error:
            outcomes[team] = Outcome(
                team,
                "",
                "rejected",
                None,
                [],
                [f"fetch failed for {team} (exit {error.returncode})"],
            )
        except Exception:
            outcomes[team] = Outcome(
                team,
                "",
                "rejected",
                None,
                [],
                [f"fetch failed for {team}"],
            )
        else:
            if synced:
                fetched.append(team)
            else:
                outcomes[team] = Outcome(
                    team,
                    "",
                    "rejected",
                    None,
                    [],
                    ["mirror configuration differs"],
                )
    for team in fetched:
        try:
            outcomes[team] = deploy.deploy(deploy_config, team)
        except Exception:
            outcomes[team] = Outcome(
                team,
                "",
                "rejected",
                None,
                [],
                [f"deploy failed for {team}"],
            )
    results = [outcomes[team] for team in sorted(config.remotes)]
    state = DeployState(deploy_config.state_root)
    with (deploy_config.state_root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = state.read_target()
        if target is not None and target.generation != state.acknowledged():
            try:
                state.update_current_links()
                _reload_skills(config)
                state.acknowledge(target.generation)
            except Exception as error:
                problem = _reload_problem(error)
                accepted = any(
                    outcome.status == "accepted" for outcome in results
                )
                results = [
                    dataclasses.replace(
                        outcome, problems=[*outcome.problems, problem]
                    )
                    if outcome.status == "accepted" or not accepted
                    else outcome
                    for outcome in results
                ]
    return results


def _report(outcomes: list[Outcome]) -> int:
    """Print every outcome's lines and return the exit code."""
    for outcome in outcomes:
        for line in outcome.lines():
            print(line)
    return int(any(outcome.status == "rejected" for outcome in outcomes))


def main(argv: list[str] | None = None) -> int:
    """Run one reconciliation pass or repeat on the configured interval."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("once", "loop"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--config", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    config = ReconcilerConfig.from_json(args.config)
    if args.command == "once":
        return _report(run_once(config))
    try:
        while True:
            _report(run_once(config))
            time.sleep(config.interval)
    except KeyboardInterrupt:
        return 0
