"""Deploy a team's main revision and report attempts from hooks or the CLI."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import typing

from soliplex_concierge.skills_deployer import content
from soliplex_concierge.skills_deployer import loader
from soliplex_concierge.skills_deployer import manifest
from soliplex_concierge.skills_deployer.model import Check
from soliplex_concierge.skills_deployer.model import Limits
from soliplex_concierge.skills_deployer.model import Violation
from soliplex_concierge.skills_deployer.state import DeployState
from soliplex_concierge.skills_deployer.state import Target


@dataclasses.dataclass(frozen=True)
class DeployConfig:
    """Describe the repositories and validation inputs for an installation."""

    state_root: pathlib.Path
    repos: dict[str, pathlib.Path]
    installation: pathlib.Path
    baseline_paths: list[pathlib.Path]
    baseline_names: set[str]
    room_paths: list[pathlib.Path]
    room_revision: str
    loader_command: list[str]
    limits: Limits = Limits()
    loader_timeout: float = 120.0

    @classmethod
    def from_json(cls, path: pathlib.Path) -> DeployConfig:
        """Read deployment configuration from a JSON file."""
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(
            state_root=pathlib.Path(data["state_root"]),
            repos={k: pathlib.Path(v) for k, v in data["repos"].items()},
            installation=pathlib.Path(data["installation"]),
            baseline_paths=[pathlib.Path(p) for p in data["baseline_paths"]],
            baseline_names=set(data["baseline_names"]),
            room_paths=[pathlib.Path(p) for p in data["room_paths"]],
            room_revision=data["room_revision"],
            loader_command=data["loader_command"],
            limits=Limits(**data.get("limits", {})),
            loader_timeout=data.get("loader_timeout", 120.0),
        )


@dataclasses.dataclass(frozen=True)
class Outcome:
    """Report a team's deploy status and any validation failures."""

    team: str
    revision: str
    status: typing.Literal["unchanged", "accepted", "rejected"]
    target: Target | None
    violations: list[Violation]
    problems: list[str]

    def lines(self) -> list[str]:
        """Format status and diagnostics as lines addressed to the team."""
        return [
            f"{self.team}: {self.status} {self.revision}",
            *(
                f"{self.team}: {v.check} {v.path}: {v.message}"
                for v in self.violations
            ),
            *(f"{self.team}: {problem}" for problem in self.problems),
        ]


def _strings(value: object) -> bool:
    """Check whether a JSON value is a list of strings."""
    return isinstance(value, list) and all(
        isinstance(item, str) for item in value
    )


def _reject_constant(value: str) -> typing.NoReturn:
    """Reject nonstandard numeric constants in loader JSON."""
    msg = f"Invalid JSON constant: {value}"
    raise ValueError(msg)


def _parse_result(output: bytes) -> loader.LoadResult:
    """Parse and check the loader's versioned result document."""
    data = json.loads(output, parse_constant=_reject_constant)
    if (
        not isinstance(data, dict)
        or set(data)
        != {"loader_result", "ok", "skills", "rooms", "shadowed", "problems"}
        or type(data["loader_result"]) is not int
        or data["loader_result"] != loader.LOADER_RESULT_VERSION
        or type(data["ok"]) is not bool
        or not isinstance(data["skills"], dict)
        or not all(isinstance(v, dict) for v in data["skills"].values())
        or not _strings(data["rooms"])
        or not isinstance(data["shadowed"], list)
        or not all(
            _strings(item) and len(item) == 3 for item in data["shadowed"]
        )
        or not _strings(data["problems"])
    ):
        msg = "Loader output must be a LoadResult JSON document"
        raise ValueError(msg)
    del data["loader_result"]
    return loader.LoadResult(**data)


def _run_loader(
    config: DeployConfig, request: loader.LoadRequest
) -> tuple[loader.LoadResult, bool]:
    """Validate a candidate in a separate process with a bounded runtime."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as stream:
        stream.write(request.to_json())
        stream.flush()
        cache_rejection = False
        try:
            process = subprocess.run(
                [*config.loader_command, stream.name],
                capture_output=True,
                timeout=config.loader_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            problems = [
                f"Loader timed out after {config.loader_timeout} seconds"
            ]
        except OSError as error:
            problems = [f"Loader cannot start: {error}"]
        else:
            try:
                result = _parse_result(process.stdout)
            except (ValueError, UnicodeError):
                problems = ["Loader output must be a LoadResult JSON document"]
            else:
                cache_rejection = True
                problems = list(result.problems)
                if process.returncode != 0 or not result.ok:
                    if not problems:
                        problems.append(
                            f"Loader rejected generation "
                            f"(exit {process.returncode})"
                        )
                if not problems:
                    return result, cache_rejection
    return loader.LoadResult(False, {}, [], [], problems), cache_rejection


def _reject(
    state: DeployState,
    team: str,
    revision: str,
    context: dict,
    violations: list[Violation],
    problems: list[str],
) -> Outcome:
    """Cache validation failures and report the rejected revision."""
    violations = violations + [
        Violation(Check.GENERATION, "", p) for p in problems
    ]
    state.reject(team, revision, context, violations)
    return Outcome(
        team, revision, "rejected", state.read_target(), violations, problems
    )


def _attempt(config: DeployConfig, team: str, state: DeployState) -> Outcome:
    """Prepare, validate and commit one revision while holding the lock."""
    state.recover()
    repo = config.repos[team]
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "refs/heads/main"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    target = state.read_target()
    context = {
        "generation": 0 if target is None else target.generation,
        "room_revision": config.room_revision,
        "validator": 1,
    }
    if state.committed_revision(team) == revision or state.is_rejected(
        team, revision, context
    ):
        return Outcome(team, revision, "unchanged", target, [], [])
    generation = (
        {"teams": {}, "loader_result": {"skills": {}}}
        if target is None
        else state.read_generation(target.generation)
    )
    teams = {k: v for k, v in generation["teams"].items() if k != team}
    expected = {
        name: loader.ExpectedSkill(**spec)
        for name, spec in generation["loader_result"]["skills"].items()
        if spec["owner"] != team
    }
    entries, violations = manifest.read_tree(repo, revision)
    violations.extend(
        manifest.check_structure(
            team,
            entries,
            other_names=set(expected) | config.baseline_names,
            limits=config.limits,
        )
    )
    if violations:
        return _reject(state, team, revision, context, violations, [])
    versions = config.state_root / "versions"
    seq = 1 + sum(p.is_dir() for p in (versions / team).glob(f"{revision}-*"))
    try:
        version = content.write_version(
            repo,
            team,
            revision,
            entries,
            versions,
            seq=seq,
            limits=config.limits,
        )
    except content.ContentRejected as error:
        return _reject(state, team, revision, context, error.violations, [])
    teams[team] = str(version.path.relative_to(config.state_root))
    expected.update(
        (s.name, loader.ExpectedSkill(team, s.instructions_digest))
        for s in version.skills
    )
    allow_list = sorted(set(expected) | config.baseline_names)
    request = loader.LoadRequest(
        installation=config.installation,
        team_paths={k: config.state_root / v for k, v in teams.items()},
        baseline_paths=config.baseline_paths,
        allow_list=allow_list,
        room_paths=config.room_paths,
        expected=expected,
    )
    result, cache_rejection = _run_loader(config, request)
    if not result.ok:
        version.path.chmod(version.path.stat().st_mode | stat.S_IRWXU)
        for child in version.path.rglob("*"):
            child.chmod(child.stat().st_mode | stat.S_IRWXU)
        shutil.rmtree(version.path)
        if not cache_rejection:
            return Outcome(
                team, revision, "rejected", target, [], result.problems
            )
        return _reject(state, team, revision, context, [], result.problems)
    loader_result = json.loads(result.to_json())
    loader_result["skills"] = {
        name: dataclasses.asdict(spec) for name, spec in expected.items()
    }
    number = state.write_generation(
        teams,
        baseline={str(p): "" for p in config.baseline_paths},
        allow_list=allow_list,
        room_revision=config.room_revision,
        loader_result=loader_result,
    )
    target = state.commit(number, config.room_revision)
    return Outcome(team, revision, "accepted", target, [], [])


def deploy(config: DeployConfig, team: str) -> Outcome:
    """Serialize one team's deploy attempt for the whole installation."""
    config.state_root.mkdir(parents=True, exist_ok=True)
    with (config.state_root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _attempt(config, team, DeployState(config.state_root))


def _remove_version(path: pathlib.Path) -> None:
    """Make an uncommitted version directory writable and remove it."""
    path.chmod(path.stat().st_mode | stat.S_IRWXU)
    for child in path.rglob("*"):
        child.chmod(child.stat().st_mode | stat.S_IRWXU)
    shutil.rmtree(path)


def _bootstrap(config: DeployConfig, state: DeployState) -> list[Outcome]:
    """Prepare, validate and commit the first generation for all teams."""
    state.recover()
    if state.read_target() is not None:
        msg = "already bootstrapped"
        raise RuntimeError(msg)
    teams = sorted(config.repos)
    revisions: dict[str, str] = {}
    trees: dict[str, tuple[list, list[Violation]]] = {}
    names: dict[str, set[str]] = {}
    for team in teams:
        repo = config.repos[team]
        revision = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "rev-parse",
                "--verify",
                "refs/heads/main",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        entries, violations = manifest.read_tree(repo, revision)
        revisions[team] = revision
        trees[team] = (entries, violations)
        names[team] = {
            entry.path.split("/", 1)[0]
            for entry in entries
            if "/" in entry.path
        }
    versions_root = config.state_root / "versions"
    versions = {}
    outcomes = []
    for team in teams:
        entries, violations = trees[team]
        other_names = config.baseline_names.union(
            *(names[other] for other in teams if other != team)
        )
        violations.extend(
            manifest.check_structure(
                team,
                entries,
                other_names=other_names,
                limits=config.limits,
            )
        )
        if not violations:
            try:
                versions[team] = content.write_version(
                    config.repos[team],
                    team,
                    revisions[team],
                    entries,
                    versions_root,
                    seq=1,
                    limits=config.limits,
                )
            except content.ContentRejected as error:
                violations = error.violations
        if violations:
            outcomes.append(
                Outcome(
                    team, revisions[team], "rejected", None, violations, []
                )
            )
    if outcomes:
        for version in versions.values():
            _remove_version(version.path)
        return outcomes
    team_versions = {
        team: str(version.path.relative_to(config.state_root))
        for team, version in versions.items()
    }
    expected = {
        skill.name: loader.ExpectedSkill(team, skill.instructions_digest)
        for team, version in versions.items()
        for skill in version.skills
    }
    allow_list = sorted(set(expected) | config.baseline_names)
    request = loader.LoadRequest(
        installation=config.installation,
        team_paths={
            team: config.state_root / path
            for team, path in team_versions.items()
        },
        baseline_paths=config.baseline_paths,
        allow_list=allow_list,
        room_paths=config.room_paths,
        expected=expected,
    )
    result, _ = _run_loader(config, request)
    if not result.ok:
        for version in versions.values():
            _remove_version(version.path)
        return [
            Outcome(
                team, revisions[team], "rejected", None, [], result.problems
            )
            for team in teams
        ]
    loader_result = json.loads(result.to_json())
    loader_result["skills"] = {
        name: dataclasses.asdict(spec) for name, spec in expected.items()
    }
    number = state.write_generation(
        team_versions,
        baseline={str(p): "" for p in config.baseline_paths},
        allow_list=allow_list,
        room_revision=config.room_revision,
        loader_result=loader_result,
    )
    target = state.commit(number, config.room_revision)
    return [
        Outcome(team, revisions[team], "accepted", target, [], [])
        for team in teams
    ]


def bootstrap(config: DeployConfig) -> list[Outcome]:
    """Build and commit the first generation from every team at once."""
    config.state_root.mkdir(parents=True, exist_ok=True)
    with (config.state_root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _bootstrap(config, DeployState(config.state_root))


def main(argv: list[str] | None = None) -> int:
    """Run one deploy or reconcile every configured team from the CLI."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("deploy", "reconcile", "hook", "bootstrap"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--config", type=pathlib.Path, required=True)
        if command in ("deploy", "hook"):
            subparser.add_argument("--team", required=True)
    args = parser.parse_args(argv)
    config = DeployConfig.from_json(args.config)
    if args.command == "bootstrap":
        try:
            outcomes = bootstrap(config)
        except RuntimeError as error:
            print(error)
            return 1
        for outcome in outcomes:
            for line in outcome.lines():
                print(line)
        return int(not all(o.status == "accepted" for o in outcomes))
    if args.command == "hook":
        sys.stdin.read()
    teams = (
        sorted(config.repos) if args.command == "reconcile" else [args.team]
    )
    rejected = False
    for team in teams:
        outcome = deploy(config, team)
        for line in outcome.lines():
            print(line)
        rejected |= outcome.status == "rejected"
    return int(rejected)
