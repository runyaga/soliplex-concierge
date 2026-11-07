"""Unit tests for ``soliplex_concierge.skills_deployer.loader``."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from soliplex_concierge.skills_deployer import loader

INSTALLATION_YAML = "id: test-installation\noidc_paths: []\n"


def _write_installation(root: pathlib.Path) -> pathlib.Path:
    """Write a minimal Soliplex installation configuration in root."""
    inst = root / "environment"
    inst.mkdir(parents=True, exist_ok=True)
    (inst / "installation.yaml").write_text(
        INSTALLATION_YAML, encoding="utf-8"
    )
    return inst


def _write_skill(
    parent: pathlib.Path,
    name: str,
    instructions: str = "Follow the team instructions.",
) -> tuple[pathlib.Path, str]:
    """Write a valid skill directory and return its path and digest."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    body = (
        f"---\nname: {name}\ndescription: Skill {name}\n---\n{instructions}\n"
    )
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")
    return skill_dir, loader.instruction_digest(instructions.strip())


def _write_room(
    rooms_dir: pathlib.Path,
    room_id: str,
    skill_names: tuple[str, ...] = (),
    system_prompt: str = "Test prompt.",
) -> pathlib.Path:
    """Write a room configuration binding skills with an OpenAI provider."""
    room_dir = rooms_dir / room_id
    room_dir.mkdir(parents=True, exist_ok=True)
    skills_block = "\n".join(f"    - {name}" for name in skill_names)
    config_path = room_dir / "room_config.yaml"
    config_path.write_text(
        f"id: {room_id}\n"
        f"name: Room {room_id}\n"
        f"description: Test room {room_id}\n"
        "agent:\n"
        "  provider_type: openai\n"
        "  provider_base_url: http://openai.example.invalid/v1\n"
        "  model_name: test-model\n"
        f"  system_prompt: {system_prompt}\n"
        "skills:\n"
        "  installation_skill_names:\n"
        f"{skills_block}\n",
        encoding="utf-8",
    )
    return config_path


def test_valid_generation(tmp_path, monkeypatch):
    """A valid generation loads skills, reads resources, and builds rooms."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    alpha_dir, alpha_digest = _write_skill(
        teama_dir, "teama-alpha", "Use alpha carefully."
    )
    (alpha_dir / "resources" / "nested").mkdir(parents=True)
    (alpha_dir / "resources" / "guide.md").write_text(
        "Guide content\n", encoding="utf-8"
    )
    (alpha_dir / "resources" / "nested" / "detail.md").write_text(
        "Nested content\n", encoding="utf-8"
    )
    _, beta_digest = _write_skill(teama_dir, "teama-beta", "Use beta.")
    baseline_dir = tmp_path / "baseline"
    _write_skill(baseline_dir, "soliplex-docs", "Baseline docs.")
    rooms_dir = tmp_path / "rooms"
    _write_room(
        rooms_dir, "room-a", ("teama-alpha", "teama-beta", "soliplex-docs")
    )
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        baseline_paths=[baseline_dir],
        room_paths=[rooms_dir],
        expected={
            "teama-alpha": loader.ExpectedSkill("teama", alpha_digest),
            "teama-beta": loader.ExpectedSkill("teama", beta_digest),
        },
    )
    original_get_model = loader.config_agents.get_model_from_config
    patched_get_model = None
    real_get_agent = loader.agents.get_agent_from_configs

    def spy_get_agent(**kwargs):
        nonlocal patched_get_model
        patched_get_model = loader.config_agents.get_model_from_config
        return real_get_agent(**kwargs)

    monkeypatch.setattr(loader.agents, "get_agent_from_configs", spy_get_agent)

    result = loader.validate(request)

    assert result.ok is True
    assert result.problems == []
    assert result.rooms == ["room-a"]
    assert result.shadowed == []
    assert patched_get_model is not original_get_model
    patched_get_model.assert_called_once()
    assert loader.config_agents.get_model_from_config is original_get_model
    assert result.skills["teama-alpha"] == {
        "path": str(alpha_dir),
        "owner": "teama",
        "errors": [],
        "digest": alpha_digest,
    }
    assert result.skills["soliplex-docs"]["owner"] is None
    payload = json.loads(result.to_json())
    assert payload["loader_result"] == 1
    assert payload["ok"] is True


def test_placeholder_skill(tmp_path):
    """A placeholder skill reports its errors without stopping other skills."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    broken_dir = teama_dir / "teama-broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / "SKILL.md").write_text(
        "---\nname: teama-broken\ndescription: Broken\n---\n",
        encoding="utf-8",
    )
    _, valid_digest = _write_skill(teama_dir, "teama-valid")
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        expected={
            "teama-broken": loader.ExpectedSkill("teama", "0" * 64),
            "teama-valid": loader.ExpectedSkill("teama", valid_digest),
        },
    )

    result = loader.validate(request)

    assert result.ok is False
    errors = result.skills["teama-broken"]["errors"]
    assert errors
    assert result.skills["teama-broken"]["digest"] is None
    assert result.skills["teama-valid"]["digest"] == valid_digest
    assert result.problems == [
        f"skill 'teama-broken' failed validation: {', '.join(errors)}"
    ]


def test_expected_skill_in_baseline_path(tmp_path):
    """An expected team skill discovered only in a baseline path fails."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    teama_dir.mkdir(parents=True)
    baseline_dir = tmp_path / "baseline"
    _, digest = _write_skill(baseline_dir, "teama-alpha")
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        baseline_paths=[baseline_dir],
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert result.skills["teama-alpha"]["owner"] is None
    assert result.problems == [
        "skill 'teama-alpha' owner is None, expected 'teama'"
    ]


def test_wrong_owner(tmp_path):
    """A skill whose directory owner differs from expected fails validation."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, digest = _write_skill(teama_dir, "teama-alpha")
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        expected={"teama-alpha": loader.ExpectedSkill("teamb", digest)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert result.skills["teama-alpha"]["owner"] == "teama"
    assert any(
        "owner is 'teama', expected 'teamb'" in problem
        for problem in result.problems
    )


def test_unexpected_team_skill(tmp_path):
    """A team skill missing from the expected map is reported."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _write_skill(teama_dir, "teama-alpha")
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        expected={},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert any(
        "unexpected team skill 'teama-alpha'" in problem
        for problem in result.problems
    )


def test_wrong_digest(tmp_path):
    """A skill whose instruction digest differs from expected fails."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _write_skill(teama_dir, "teama-alpha", "Actual instructions.")
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        expected={"teama-alpha": loader.ExpectedSkill("teama", "0" * 64)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert any("digest is" in problem for problem in result.problems)


def test_missing_expected_skill(tmp_path):
    """An expected skill that was not discovered is reported."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    teama_dir.mkdir(parents=True)
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        expected={"teama-missing": loader.ExpectedSkill("teama", "0" * 64)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert any(
        "expected skill 'teama-missing' was not discovered" in problem
        for problem in result.problems
    )


def test_skill_with_scripts(tmp_path):
    """A team skill with a scripts directory exposes run_script and fails."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    skill_dir, digest = _write_skill(teama_dir, "teama-alpha")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "helper.py").write_text("print('hi')\n", encoding="utf-8")
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert any("run_script" in problem for problem in result.problems)


def test_unreadable_resource(tmp_path):
    """A resource file that read_resource cannot decode fails validation."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    skill_dir, digest = _write_skill(teama_dir, "teama-alpha")
    resources_dir = skill_dir / "resources"
    resources_dir.mkdir()
    (resources_dir / "bad.txt").write_bytes(b"\xff\xfe")
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert any("resources/bad.txt" in problem for problem in result.problems)


def test_skill_check_exception_does_not_stop_others(tmp_path, monkeypatch):
    """An exception checking one skill is recorded while other checks run."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, alpha_digest = _write_skill(teama_dir, "teama-alpha")
    _, beta_digest = _write_skill(teama_dir, "teama-beta")
    rooms_dir = tmp_path / "rooms"
    _write_room(rooms_dir, "room-a", ("teama-beta",))
    real_parse = loader.config_installation.cap_fs._parse_skill_file

    class BrokenCapability(
        loader.config_installation.cap_fs.FilesystemCapability
    ):
        error = OSError("unreadable instructions")

        @property
        def instructions(self) -> str:
            raise self.error

    def fake_parse(skill_path):
        cap = real_parse(skill_path)
        if skill_path.name == "teama-alpha":
            cap.__class__ = BrokenCapability
        return cap

    monkeypatch.setattr(
        loader.config_installation.cap_fs, "_parse_skill_file", fake_parse
    )
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        room_paths=[rooms_dir],
        expected={
            "teama-alpha": loader.ExpectedSkill("teama", alpha_digest),
            "teama-beta": loader.ExpectedSkill("teama", beta_digest),
        },
    )

    result = loader.validate(request)

    assert result.ok is False
    assert result.skills["teama-alpha"]["digest"] is None
    assert result.skills["teama-beta"]["digest"] == beta_digest
    assert result.rooms == ["room-a"]
    assert len(result.problems) == 1
    assert "teama-alpha" in result.problems[0]
    assert "unreadable instructions" in result.problems[0]


def test_room_binding_missing_skill(tmp_path):
    """A room referencing an unknown skill fails room configuration loading."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, digest = _write_skill(teama_dir, "teama-alpha")
    rooms_dir = tmp_path / "rooms"
    _write_room(rooms_dir, "room-a", ("no-such-skill",))
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        room_paths=[rooms_dir],
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert any("no-such-skill" in problem for problem in result.problems)


def test_first_room_build_failure_does_not_stop_second(tmp_path):
    """When one room fails agent construction, remaining rooms still build."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, digest = _write_skill(teama_dir, "teama-alpha")
    rooms_dir = tmp_path / "rooms"
    _write_room(
        rooms_dir, "room-a", ("teama-alpha",), system_prompt="./missing.txt"
    )
    _write_room(rooms_dir, "room-b", ("teama-alpha",))
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        room_paths=[rooms_dir],
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert result.rooms == ["room-b"]
    assert any("missing.txt" in problem for problem in result.problems)


def test_shadowed_room_reported(tmp_path):
    """A room shadowed by an earlier room path is recorded in shadowed."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, digest = _write_skill(teama_dir, "teama-alpha")
    first_rooms = tmp_path / "rooms-first"
    second_rooms = tmp_path / "rooms-second"
    kept = _write_room(first_rooms, "room-a", ("teama-alpha",))
    skipped = _write_room(second_rooms, "room-a", ("teama-alpha",))
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        room_paths=[first_rooms, second_rooms],
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )

    result = loader.validate(request)

    assert result.ok is True
    assert result.rooms == ["room-a"]
    assert result.shadowed == [("room-a", str(kept), str(skipped))]


def test_shadowed_room_reported_when_another_room_fails(tmp_path):
    """Shadowed rooms are recorded even when another room fails to load."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, digest = _write_skill(teama_dir, "teama-alpha")
    first_rooms = tmp_path / "rooms-first"
    second_rooms = tmp_path / "rooms-second"
    kept = _write_room(first_rooms, "room-a", ("teama-alpha",))
    _write_room(first_rooms, "room-b", ("no-such-skill",))
    skipped = _write_room(second_rooms, "room-a", ("teama-alpha",))
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        room_paths=[first_rooms, second_rooms],
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert result.shadowed == [("room-a", str(kept), str(skipped))]
    assert any("no-such-skill" in problem for problem in result.problems)


def test_room_without_id_does_not_stop_discovery(tmp_path):
    """A room file missing id is recorded while later rooms still load."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, digest = _write_skill(teama_dir, "teama-alpha")
    first_rooms = tmp_path / "rooms-first"
    second_rooms = tmp_path / "rooms-second"
    bad_dir = first_rooms / "room-a"
    bad_dir.mkdir(parents=True)
    bad_config = bad_dir / "room_config.yaml"
    bad_config.write_text(
        "name: Room without id\ndescription: Missing id\n",
        encoding="utf-8",
    )
    kept = _write_room(first_rooms, "room-b", ("teama-alpha",))
    skipped = _write_room(second_rooms, "room-b", ("teama-alpha",))
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        room_paths=[first_rooms, second_rooms],
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert result.rooms == ["room-b"]
    assert result.shadowed == [("room-b", str(kept), str(skipped))]
    assert len(result.problems) == 1
    assert str(bad_config) in result.problems[0]


def test_allow_list_hides_skill_needed_by_room(tmp_path):
    """An allow-list excluding a skill that a room binds causes a failure."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, alpha_digest = _write_skill(teama_dir, "teama-alpha")
    _, beta_digest = _write_skill(teama_dir, "teama-beta")
    rooms_dir = tmp_path / "rooms"
    _write_room(rooms_dir, "room-a", ("teama-beta",))
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        allow_list=["teama-alpha"],
        room_paths=[rooms_dir],
        expected={
            "teama-alpha": loader.ExpectedSkill("teama", alpha_digest),
            "teama-beta": loader.ExpectedSkill("teama", beta_digest),
        },
    )

    result = loader.validate(request)

    assert result.ok is False
    assert any("teama-beta" in problem for problem in result.problems)


def test_missing_installation_is_caught(tmp_path):
    """A missing installation file is caught and reported in problems."""
    request = loader.LoadRequest(
        installation=tmp_path / "missing",
        team_paths={},
    )

    result = loader.validate(request)

    assert result.ok is False
    assert len(result.problems) == 1


def test_main_good_input(tmp_path, capsys):
    """main returns 0 and prints the JSON result when validation succeeds."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    _, digest = _write_skill(teama_dir, "teama-alpha")
    rooms_dir = tmp_path / "rooms"
    _write_room(rooms_dir, "room-a", ("teama-alpha",))
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        allow_list=["teama-alpha"],
        room_paths=[rooms_dir],
        expected={"teama-alpha": loader.ExpectedSkill("teama", digest)},
    )
    req_file = tmp_path / "request.json"
    req_file.write_text(request.to_json(), encoding="utf-8")

    code = loader.main([str(req_file)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["loader_result"] == 1
    assert payload["ok"] is True


def test_main_failing_input(tmp_path, capsys, monkeypatch):
    """main returns 1 and prints the JSON result when validation fails."""
    inst = _write_installation(tmp_path)
    teama_dir = tmp_path / "teams" / "teama"
    teama_dir.mkdir(parents=True)
    request = loader.LoadRequest(
        installation=inst,
        team_paths={"teama": teama_dir},
        expected={"teama-missing": loader.ExpectedSkill("teama", "0" * 64)},
    )
    req_file = tmp_path / "request.json"
    req_file.write_text(request.to_json(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["loader", str(req_file)])

    code = loader.main()

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["loader_result"] == 1
    assert payload["ok"] is False


@pytest.mark.parametrize(
    "argv, content",
    [
        ([], None),
        (["missing.json"], None),
        (["bad.json"], "not-json"),
        (["empty.json"], "{}"),
        (["list.json"], "[]"),
        (["deep.json"], "[" * 5000 + "]" * 5000),
    ],
)
def test_main_unreadable_input(tmp_path, argv, content):
    """main returns 2 when the input file cannot be read or parsed."""
    resolved_argv = [str(tmp_path / arg) for arg in argv]
    if content is not None:
        (tmp_path / argv[0]).write_text(content, encoding="utf-8")

    code = loader.main(resolved_argv)

    assert code == 2
