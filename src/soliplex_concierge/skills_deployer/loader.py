"""Load a candidate skill generation in Soliplex and check every room."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import hashlib
import json
import pathlib
import sys
import typing
from unittest import mock

from pydantic_ai.models import test as ai_test
from soliplex import agents
from soliplex.config import agents as config_agents
from soliplex.config import installation as config_installation
from soliplex.config import rooms as config_rooms

from soliplex_concierge.skills_deployer import model

LOADER_RESULT_VERSION = 1


@dataclasses.dataclass(frozen=True)
class ExpectedSkill:
    """Owner team and normalised instruction digest expected for a skill."""

    owner: str
    digest: str


@dataclasses.dataclass(frozen=True)
class LoadRequest:
    """Inputs for validating a candidate skill generation in Soliplex."""

    installation: pathlib.Path
    team_paths: dict[str, pathlib.Path]
    baseline_paths: list[pathlib.Path] = dataclasses.field(
        default_factory=list
    )
    allow_list: list[str] | None = None
    room_paths: list[pathlib.Path] = dataclasses.field(default_factory=list)
    expected: dict[str, ExpectedSkill] = dataclasses.field(
        default_factory=dict
    )

    def to_json(self) -> str:
        """Serialize this request as a JSON document."""
        return json.dumps(
            {
                "installation": str(self.installation),
                "team_paths": {
                    team: str(path) for team, path in self.team_paths.items()
                },
                "baseline_paths": [str(path) for path in self.baseline_paths],
                "allow_list": self.allow_list,
                "room_paths": [str(path) for path in self.room_paths],
                "expected": {
                    name: {"owner": spec.owner, "digest": spec.digest}
                    for name, spec in self.expected.items()
                },
            }
        )

    @classmethod
    def from_json(cls, text: str) -> LoadRequest:
        """Parse a request from a JSON document."""
        data = json.loads(text)
        allow_list = data.get("allow_list")
        return cls(
            installation=pathlib.Path(data["installation"]),
            team_paths={
                team: pathlib.Path(path)
                for team, path in data["team_paths"].items()
            },
            baseline_paths=[
                pathlib.Path(path) for path in data.get("baseline_paths", ())
            ],
            allow_list=(
                None
                if allow_list is None
                else [str(item) for item in allow_list]
            ),
            room_paths=[
                pathlib.Path(path) for path in data.get("room_paths", ())
            ],
            expected={
                name: ExpectedSkill(
                    owner=spec["owner"],
                    digest=spec["digest"],
                )
                for name, spec in data.get("expected", {}).items()
            },
        )


@dataclasses.dataclass(frozen=True)
class LoadResult:
    """Outcome of loading a candidate skill generation in Soliplex."""

    ok: bool
    skills: dict[str, dict[str, typing.Any]]
    rooms: list[str]
    shadowed: list[tuple[str, str, str]]
    problems: list[str]

    def to_json(self) -> str:
        """Serialize this result as a versioned JSON document."""
        return json.dumps(
            {
                "loader_result": LOADER_RESULT_VERSION,
                "ok": self.ok,
                "skills": self.skills,
                "rooms": self.rooms,
                "shadowed": [list(item) for item in self.shadowed],
                "problems": self.problems,
            }
        )


def instruction_digest(instructions: str) -> str:
    """Compute the SHA-256 hex digest of normalised skill instructions."""
    return hashlib.sha256(instructions.encode("utf-8")).hexdigest()


def _describe_exception(exc: BaseException) -> str:
    """Format an exception and its direct cause into one message."""
    if exc.__cause__ is not None:
        return f"{exc}{exc.__cause__}"
    return str(exc)


def _owner_team(
    skill_path: pathlib.Path,
    team_paths: collections.abc.Mapping[str, pathlib.Path],
) -> str | None:
    """Find the team slug whose version directory contains a skill path."""
    resolved = skill_path.resolve()
    for team, team_path in team_paths.items():
        if resolved.is_relative_to(team_path.resolve()):
            return team
    return None


def _check_resources(
    name: str,
    skill_cfg: typing.Any,
    problems: list[str],
) -> None:
    """Check that every file under a skill's resources is readable."""
    resources_dir = skill_cfg.path / model.RESOURCES_DIRNAME
    if not resources_dir.is_dir():
        return
    for resource in sorted(resources_dir.rglob("*")):
        if resource.is_file():
            rel_path = resource.relative_to(skill_cfg.path).as_posix()
            try:
                asyncio.run(skill_cfg.capability.read_resource(rel_path))
            except Exception as exc:
                problems.append(
                    f"skill {name!r} resource {rel_path!r} failed: {exc}"
                )


def _check_skills(
    icfg: config_installation.InstallationConfig,
    request: LoadRequest,
    skills: dict[str, dict[str, typing.Any]],
    problems: list[str],
) -> None:
    """Check every discovered team skill and verify expected skills exist."""
    for name, skill_cfg in icfg.available_filesystem_skill_configs.items():
        try:
            owner = _owner_team(skill_cfg.path, request.team_paths)
            if (
                request.allow_list is not None
                and owner is None
                and name not in request.allow_list
            ):
                continue
            skills[name] = {
                "path": str(skill_cfg.path),
                "owner": owner,
                "errors": list(skill_cfg.errors),
                "digest": None,
            }
            if skill_cfg.errors:
                problems.append(
                    f"skill {name!r} failed validation: "
                    f"{', '.join(skill_cfg.errors)}"
                )
                continue
            digest = instruction_digest(skill_cfg.capability.instructions)
            skills[name]["digest"] = digest
            expected = request.expected.get(name)
            if owner is None and expected is None:
                continue
            if expected is None:
                problems.append(
                    f"unexpected team skill {name!r} owned by {owner!r}"
                )
            else:
                if owner != expected.owner:
                    problems.append(
                        f"skill {name!r} owner is {owner!r}, "
                        f"expected {expected.owner!r}"
                    )
                if digest != expected.digest:
                    problems.append(
                        f"skill {name!r} digest is {digest!r}, "
                        f"expected {expected.digest!r}"
                    )
            toolset = skill_cfg.capability.get_toolset()
            if toolset is not None and "run_script" in toolset.tools:
                problems.append(f"skill {name!r} exposes run_script")
            _check_resources(name, skill_cfg, problems)
        except Exception as exc:
            problems.append(
                f"skill {name!r} failed validation: {_describe_exception(exc)}"
            )

    for name in request.expected:
        if name not in skills:
            problems.append(f"expected skill {name!r} was not discovered")


def _load_rooms(
    icfg: config_installation.InstallationConfig,
    shadowed: list[tuple[str, str, str]],
    problems: list[str],
) -> dict[str, typing.Any]:
    """Load effective room configurations and record shadowed rooms."""
    room_configs: dict[str, typing.Any] = {}
    kept_paths: dict[str, pathlib.Path] = {}
    for room_path in icfg.room_paths:
        for config_path, config_yaml in config_installation._find_room_configs(
            room_path
        ):
            try:
                room_id = config_yaml["id"]
                if room_id in kept_paths:
                    shadowed.append(
                        (room_id, str(kept_paths[room_id]), str(config_path))
                    )
                    continue
                kept_paths[room_id] = config_path
                room_configs[room_id] = config_rooms.RoomConfig.from_yaml(
                    icfg,
                    config_path,
                    config_yaml,
                )
            except Exception as exc:
                problems.append(f"{config_path}: {_describe_exception(exc)}")
    return room_configs


def _check_rooms(
    room_configs: collections.abc.Mapping[str, typing.Any],
    rooms: list[str],
    problems: list[str],
) -> None:
    """Build an agent with a test model for every effective room."""
    with mock.patch.object(
        config_agents,
        "get_model_from_config",
        return_value=ai_test.TestModel(),
    ):
        for room_id, room_cfg in room_configs.items():
            try:
                agents.get_agent_from_configs(
                    agent_config=room_cfg.agent_config,
                    tool_configs=room_cfg.tool_configs,
                    mcp_client_toolset_configs=(
                        room_cfg.mcp_client_toolset_configs
                    ),
                    capability_config=room_cfg.skills,
                )
            except Exception as exc:
                problems.append(_describe_exception(exc))
            else:
                rooms.append(room_id)


def validate(request: LoadRequest) -> LoadResult:
    """Load the candidate generation in Soliplex and check skills and rooms.

    Runs once per process because Soliplex keeps global registries. Callers
    run :func:`main` in a separate process for each validation.
    """
    skills: dict[str, dict[str, typing.Any]] = {}
    rooms: list[str] = []
    shadowed: list[tuple[str, str, str]] = []
    problems: list[str] = []

    try:
        icfg = config_installation.load_installation(request.installation)
        icfg.filesystem_skills_paths = [
            *request.team_paths.values(),
            *request.baseline_paths,
        ]
        if request.allow_list is not None:
            icfg._skill_configs = [
                {"kind": "filesystem", "skill_name": name}
                for name in request.allow_list
            ]
        icfg._available_filesystem_skill_configs = None
        icfg._resolved_skill_configs = None
        icfg.room_paths = []
        icfg.reload_configurations()
        icfg.room_paths = list(request.room_paths)
        room_configs = _load_rooms(icfg, shadowed, problems)

        _check_skills(icfg, request, skills, problems)
        _check_rooms(room_configs, rooms, problems)
    except Exception as exc:
        problems.append(_describe_exception(exc))

    return LoadResult(
        ok=not problems,
        skills=skills,
        rooms=rooms,
        shadowed=shadowed,
        problems=problems,
    )


def main(argv: list[str] | None = None) -> int:
    """Read a request JSON file, validate it, and print the JSON result."""
    args = sys.argv[1:] if argv is None else argv
    try:
        (request_file,) = args
        text = pathlib.Path(request_file).read_text(encoding="utf-8")
        request = LoadRequest.from_json(text)
    except Exception:
        return 2

    result = validate(request)
    print(result.to_json())
    return 0 if result.ok else 1
