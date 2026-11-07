"""Validate and commit installation skill bindings in provisioned rooms."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import io
import pathlib
import re
import shutil
import tempfile
import typing
from collections.abc import Mapping

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from soliplex_concierge.provisioner import landing_clone
from soliplex_concierge.skills_deployer import content
from soliplex_concierge.skills_deployer import deploy
from soliplex_concierge.skills_deployer import loader
from soliplex_concierge.skills_deployer import reconciler
from soliplex_concierge.skills_deployer.model import SKILL_FILENAME
from soliplex_concierge.skills_deployer.reconciler import ReconcilerConfig
from soliplex_concierge.skills_deployer.state import DeployState


@dataclasses.dataclass(frozen=True)
class BindResult:
    """Report a room binding status, commit and validation problems."""

    status: typing.Literal["committed", "unchanged", "refused"]
    commit: str | None
    problems: list[str]


def _skill_problems(
    config: deploy.DeployConfig,
    generation: dict,
    skills: list[str],
    team: str | None,
) -> list[str]:
    """Check requested names and committed cross-team visibility."""
    expected = generation["loader_result"]["skills"]
    problems = []
    for name in skills:
        if name in config.baseline_names:
            continue
        if name not in expected:
            problems.append(f"Unknown skill: {name}")
            continue
        owner = expected[name]["owner"]
        if team is None or owner == team:
            continue
        path = config.state_root / generation["teams"][owner]
        skill, _ = content.parse_skill_md(
            name,
            (path / name / SKILL_FILENAME).read_bytes(),
            limits=config.limits,
        )
        if skill is None or skill.visibility != "public":
            problems.append(f"Skill is private to {owner}: {name}")
    return problems


def _candidate(data: bytes, skills: list[str]) -> bytes | None:
    """Replace room bindings while preserving YAML comments and keys."""
    yaml = YAML()
    yaml.preserve_quotes = True
    try:
        room = yaml.load(data.decode("utf-8"))
    except (YAMLError, UnicodeError, ValueError):
        return None
    if not isinstance(room, Mapping) or (
        "skills" in room and not isinstance(room["skills"], Mapping)
    ):
        return None
    room["skills"] = (room.get("skills") or {}).copy()
    room["skills"]["installation_skill_names"] = [
        {"name": name, "defer_loading": False} for name in skills
    ]
    stream = io.StringIO()
    yaml.dump(room, stream)
    return stream.getvalue().encode("utf-8")


def _set_bindings(
    reconciler_config: ReconcilerConfig,
    config: deploy.DeployConfig,
    matching_paths: list[pathlib.Path],
    rooms_clone: pathlib.Path,
    room_id: str,
    skills: list[str],
    team: str | None,
    author_name: str,
    author_email: str,
) -> BindResult:
    """Prepare and validate one room change while both locks are held."""
    room = rooms_clone / room_id
    path = room / "room_config.yaml"
    if (
        re.fullmatch(r"[A-Za-z0-9._-]+", room_id) is None
        or room_id in {".", ".."}
        or room.is_symlink()
        or path.is_symlink()
        or room.resolve().parent != rooms_clone
    ):
        return BindResult("refused", None, ["Unsafe room path"])
    if not room.is_dir():
        return BindResult("refused", None, [f"Room does not exist: {room_id}"])
    if not path.is_file():
        return BindResult(
            "refused", None, [f"Room has no room_config.yaml: {room_id}"]
        )
    if path.stat().st_nlink > 1:
        return BindResult("refused", None, ["Unsafe room path"])
    duplicates = list(dict.fromkeys(n for n in skills if skills.count(n) > 1))
    if duplicates:
        return BindResult(
            "refused", None, [f"Duplicate skill: {n}" for n in duplicates]
        )
    state = DeployState(config.state_root)
    target = state.read_target()
    if target is None:
        return BindResult("refused", None, ["No committed skill generation"])
    generation = state.read_generation(target.generation)
    problems = _skill_problems(config, generation, skills, team)
    if problems:
        return BindResult("refused", None, problems)
    with tempfile.TemporaryDirectory() as directory:
        candidate = pathlib.Path(directory) / "rooms"
        shutil.copytree(rooms_clone, candidate, symlinks=True)
        current = path.read_bytes()
        updated = _candidate(current, skills)
        if updated is None:
            return BindResult("refused", None, ["Invalid room_config.yaml"])
        candidate_path = candidate / room_id / "room_config.yaml"
        candidate_path.write_bytes(updated)
        if updated == current:
            return BindResult("unchanged", None, [])
        request = loader.LoadRequest(
            installation=config.installation,
            team_paths={
                team: config.state_root / version
                for team, version in generation["teams"].items()
            },
            baseline_paths=config.baseline_paths,
            allow_list=generation["allow_list"],
            room_paths=[
                candidate if p in matching_paths else p
                for p in config.room_paths
            ],
            expected={
                name: loader.ExpectedSkill(**spec)
                for name, spec in generation["loader_result"]["skills"].items()
            },
        )
        result, _ = deploy._run_loader(config, request)
        if not result.ok:
            return BindResult("refused", None, result.problems)
        path.write_bytes(updated)
        try:
            commit = landing_clone.commit_room(
                rooms_clone,
                room_id,
                author_name=author_name,
                author_email=author_email,
                message=f"Bind skills in {room_id}",
            )
        except Exception:
            path.write_bytes(current)
            landing_clone._check(
                rooms_clone, "reset", "--quiet", "--", room_id
            )
            raise
        try:
            reconciler._reload_skills(reconciler_config)
        except Exception as error:
            problems.append(reconciler._reload_problem(error))
        return BindResult("committed", commit, problems)


def set_bindings(
    reconciler_config: ReconcilerConfig,
    rooms_clone: pathlib.Path,
    room_id: str,
    skills: list[str],
    *,
    team: str | None,
    author_name: str,
    author_email: str,
) -> BindResult:
    """Serialize room bindings against skill deploys and room writers."""
    rooms_clone = rooms_clone.resolve()
    config = deploy.DeployConfig.from_json(reconciler_config.deploy_config)
    matching_paths = [
        p for p in config.room_paths if p.resolve() == rooms_clone
    ]
    if not matching_paths:
        return BindResult("refused", None, ["Room clone is not in room_paths"])
    config.state_root.mkdir(parents=True, exist_ok=True)
    with (config.state_root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with landing_clone.locked(rooms_clone):
            return _set_bindings(
                reconciler_config,
                config,
                matching_paths,
                rooms_clone,
                room_id,
                skills,
                team,
                author_name,
                author_email,
            )


def main(argv: list[str] | None = None) -> int:
    """Bind or unbind a room's skills and print the result."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("bind")
    command.add_argument("--config", type=pathlib.Path, required=True)
    command.add_argument("--rooms-clone", type=pathlib.Path, required=True)
    command.add_argument("--room", required=True)
    command.add_argument("--skill", action="append", default=[])
    command.add_argument("--team")
    command.add_argument("--author-name", required=True)
    command.add_argument("--author-email", required=True)
    args = parser.parse_args(argv)
    result = set_bindings(
        ReconcilerConfig.from_json(args.config),
        args.rooms_clone,
        args.room,
        args.skill,
        team=args.team,
        author_name=args.author_name,
        author_email=args.author_email,
    )
    print(result.status)
    for problem in result.problems:
        print(problem)
    return int(result.status == "refused")
