"""Test room skill binding validation, commits, locking and CLI results."""

import dataclasses
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import urllib.error

import pytest
import yaml

from soliplex_concierge.provisioner import landing_clone
from soliplex_concierge.skills_deployer import bindings
from soliplex_concierge.skills_deployer import deploy
from soliplex_concierge.skills_deployer import loader
from soliplex_concierge.skills_deployer import reconciler
from soliplex_concierge.skills_deployer.state import DeployState

ROOM = "room-a"
AUTHOR = {"author_name": "Room operator", "author_email": "op@example.com"}
ROOM_TEXT = (
    "# Room settings\n"
    "id: room-a\n"
    'name: "Room A"\n'
    "agent:\n"
    "  model_name: test-model\n"
    "  system_prompt: |\n"
    "    Keep this prompt.\n"
    "skills:\n"
    "  local_paths: [] # Keep local skills\n"
    "  installation_skill_names: []\n"
    "metadata:\n"
    "  tags: [one, two]\n"
)


def _git(repo, *args):
    """Run git in a local repository and return its output."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_config(path, config):
    """Write deployment inputs as JSON for the public binding API."""
    data = dataclasses.asdict(config)
    data["baseline_names"] = sorted(config.baseline_names)
    path.write_text(json.dumps(data, default=str), encoding="utf-8")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Build committed skills, a cloned room store and a recording loader."""
    repos = {}
    for team, names in {
        "alpha": {"alpha-one": "", "alpha-two": "visibility: private\n"},
        "beta": {
            "beta-public": "visibility: public\n",
            "beta-private": "visibility: private\n",
            "beta-default": "",
        },
    }.items():
        repo = tmp_path / team
        landing_clone.ensure_clone(repo)
        for name, visibility in names.items():
            skill = repo / name
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Test skill\n"
                f"{visibility}---\nUse this skill.\n",
                encoding="utf-8",
            )
            landing_clone.commit_room(repo, name, **AUTHOR, message=name)
        repos[team] = repo
    source = tmp_path / "rooms-source"
    landing_clone.ensure_clone(source)
    (source / ROOM).mkdir()
    (source / ROOM / "room_config.yaml").write_text(
        ROOM_TEXT, encoding="utf-8"
    )
    (source / ROOM / "guide.txt").write_text("Keep the guide.\n")
    landing_clone.commit_room(source, ROOM, **AUTHOR, message="Initial room")
    rooms = tmp_path / "rooms"
    _git(tmp_path, "clone", "--quiet", str(source), str(rooms))
    installation = tmp_path / "installation"
    installation.mkdir()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    other_rooms = tmp_path / "other-rooms"
    other_rooms.mkdir()
    result_path = tmp_path / "loader-result.json"
    result_path.write_text(
        loader.LoadResult(True, {}, [ROOM], [], []).to_json()
    )
    script = tmp_path / "fake-loader.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path(__file__).parent\n"
        "request = pathlib.Path(sys.argv[1]).read_text()\n"
        "(root / 'request.json').write_text(request)\n"
        "print((root / 'loader-result.json').read_text())\n",
        encoding="utf-8",
    )
    config = deploy.DeployConfig(
        state_root=tmp_path / "state",
        repos=repos,
        installation=installation,
        baseline_paths=[baseline],
        baseline_names={"soliplex-docs"},
        room_paths=[other_rooms, rooms],
        room_revision="rooms-1",
        loader_command=[sys.executable, str(script)],
    )
    assert all(o.status == "accepted" for o in deploy.bootstrap(config))
    config_path = tmp_path / "deploy.json"
    _write_config(config_path, config)
    reconciler_config = reconciler.ReconcilerConfig(
        deploy_config=config_path,
        remotes={},
        backend_url="http://backend.invalid",
        secret_file=tmp_path / "secret",
    )
    reloads = []
    monkeypatch.setattr(reconciler, "_reload_skills", reloads.append)
    (tmp_path / "request.json").unlink()
    return reconciler_config, config, rooms, reloads


def _bind(env, skills, *, team="alpha", room=ROOM):
    """Set bindings through the public API with a fixed author."""
    config, _, rooms, _ = env
    return bindings.set_bindings(
        config, rooms, room, skills, team=team, **AUTHOR
    )


def _assert_refused(env, result, problems):
    """Check a refusal leaves the room clone and backend untouched."""
    _, config, rooms, reloads = env
    assert result == bindings.BindResult("refused", None, problems)
    assert _git(rooms, "rev-list", "--count", "HEAD") == "1"
    assert _git(rooms, "status", "--porcelain") == ""
    assert (rooms / ROOM / "room_config.yaml").read_text() == ROOM_TEXT
    assert reloads == []
    assert DeployState(config.state_root).read_target().generation == 1


def test_bind_own_skills_preserves_room_and_commits_once(env):
    reconciler_config, config, rooms, reloads = env
    names = ["alpha-two", "alpha-one"]
    result = _bind(env, names)
    assert result.status == "committed"
    assert result.problems == []
    assert result.commit == _git(rooms, "rev-parse", "HEAD")
    assert _git(rooms, "rev-list", "--count", "HEAD") == "2"
    assert _git(rooms, "log", "-1", "--format=%s") == f"Bind skills in {ROOM}"
    assert _git(rooms, "log", "-1", "--format=%an <%ae>") == (
        "Room operator <op@example.com>"
    )
    assert _git(rooms, "diff", "HEAD~", "--name-only") == (
        f"{ROOM}/room_config.yaml"
    )
    assert _git(rooms, "status", "--porcelain") == ""
    text = (rooms / ROOM / "room_config.yaml").read_text()
    actual = yaml.safe_load(text)
    expected = yaml.safe_load(ROOM_TEXT)
    expected["skills"]["installation_skill_names"] = [
        {"name": name, "defer_loading": False} for name in names
    ]
    assert actual == expected
    assert "# Room settings\n" in text
    assert 'name: "Room A"\n' in text
    assert "# Keep local skills\n" in text
    assert (rooms / ROOM / "guide.txt").read_text() == "Keep the guide.\n"
    assert reloads == [reconciler_config]
    request = loader.LoadRequest.from_json(
        (config.state_root.parent / "request.json").read_text()
    )
    generation = DeployState(config.state_root).read_generation(1)
    assert request.installation == config.installation
    assert request.team_paths == {
        team: config.state_root / path
        for team, path in generation["teams"].items()
    }
    assert request.allow_list == generation["allow_list"]
    assert request.baseline_paths == config.baseline_paths
    assert request.expected == {
        name: loader.ExpectedSkill(**spec)
        for name, spec in generation["loader_result"]["skills"].items()
    }
    assert request.room_paths[0] == config.room_paths[0]
    assert request.room_paths[1] != rooms
    assert not request.room_paths[1].exists()


@pytest.mark.parametrize(
    "name, team",
    [
        ("beta-public", "alpha"),
        ("beta-private", None),
        ("beta-default", None),
        ("soliplex-docs", "alpha"),
        ("soliplex-docs", None),
    ],
)
def test_public_operator_and_baseline_bindings(env, name, team):
    assert _bind(env, [name], team=team).status == "committed"


@pytest.mark.parametrize("name", ["beta-private", "beta-default"])
def test_other_team_private_skill_refused(env, name):
    _assert_refused(
        env, _bind(env, [name]), [f"Skill is private to beta: {name}"]
    )


def test_unknown_and_duplicate_skills_refused(env):
    _assert_refused(
        env,
        _bind(env, ["missing-one", "missing-two"]),
        ["Unknown skill: missing-one", "Unknown skill: missing-two"],
    )
    _assert_refused(
        env,
        _bind(env, ["alpha-one", "alpha-one", "alpha-one"]),
        ["Duplicate skill: alpha-one"],
    )


def test_no_committed_target_refused(env):
    _, config, _, _ = env
    (config.state_root / "target").unlink()
    result = _bind(env, ["alpha-one"])
    assert result == bindings.BindResult(
        "refused", None, ["No committed skill generation"]
    )
    assert _git(env[2], "rev-list", "--count", "HEAD") == "1"
    assert env[3] == []


@pytest.mark.parametrize("missing", ["room", "file"])
def test_missing_room_or_config_refused(env, missing):
    rooms = env[2]
    if missing == "room":
        result = _bind(env, [], room="missing-room")
        problems = ["Room does not exist: missing-room"]
    else:
        (rooms / ROOM / "room_config.yaml").unlink()
        result = _bind(env, [])
        problems = [f"Room has no room_config.yaml: {ROOM}"]
    assert result == bindings.BindResult("refused", None, problems)
    assert _git(rooms, "rev-list", "--count", "HEAD") == "1"
    assert env[3] == []


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "dot",
        "parent",
        "traversal",
        "slash",
        "absolute",
        "invalid-character",
        "newline",
        "room-symlink",
        "internal-room-symlink",
        "config-symlink",
        "resolved-parent",
    ],
)
def test_unsafe_room_paths_refused(env, monkeypatch, kind):
    rooms = env[2]
    outside = rooms.parent / "outside"
    outside.mkdir()
    sentinel = outside / "room_config.yaml"
    sentinel.write_text(ROOM_TEXT)
    room = {
        "empty": "",
        "dot": ".",
        "parent": "..",
        "traversal": "../outside",
        "slash": "/",
        "absolute": str(outside),
        "invalid-character": "room:a",
        "newline": "room-a\n",
    }.get(kind, ROOM)
    path = rooms / ROOM / "room_config.yaml"
    if kind in {"room-symlink", "internal-room-symlink"}:
        room = "room-link"
        target = outside if kind == "room-symlink" else rooms / ROOM
        (rooms / room).symlink_to(target, target_is_directory=True)
    elif kind == "config-symlink":
        path.unlink()
        path.symlink_to(sentinel)
    elif kind == "resolved-parent":
        original_resolve = pathlib.Path.resolve

        def resolve(path, *args, **kwargs):
            if path == rooms / ROOM:
                return outside
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "resolve", resolve)
    before = _git(rooms, "status", "--porcelain")
    result = _bind(env, ["alpha-one"], room=room)
    assert result == bindings.BindResult("refused", None, ["Unsafe room path"])
    assert sentinel.read_text() == ROOM_TEXT
    assert path.read_text() == ROOM_TEXT
    assert _git(rooms, "status", "--porcelain") == before
    assert _git(rooms, "rev-list", "--count", "HEAD") == "1"
    assert env[3] == []
    assert not (rooms.parent / "request.json").exists()


def test_room_clone_missing_from_room_paths_refused(env):
    reconciler_config, config, rooms, _ = env
    config.room_paths.pop()
    _write_config(reconciler_config.deploy_config, config)
    install_lock = config.state_root / "install.lock"
    install_lock.unlink(missing_ok=True)
    room_lock = landing_clone.lock_path(rooms)
    assert not room_lock.exists()
    _assert_refused(env, _bind(env, []), ["Room clone is not in room_paths"])
    assert not room_lock.exists()
    assert not install_lock.exists()


def test_hardlinked_room_config_refused(env):
    rooms = env[2]
    path = rooms / ROOM / "room_config.yaml"
    other = rooms.parent / "linked-room_config.yaml"
    os.link(path, other)
    _assert_refused(env, _bind(env, ["alpha-one"]), ["Unsafe room path"])
    assert other.read_text() == ROOM_TEXT


def test_candidate_preserves_other_room_symlink(env, monkeypatch):
    rooms = env[2]
    outside = rooms.parent / "outside.txt"
    outside.write_text("Outside content")
    other_room = rooms / "room-b"
    other_room.mkdir()
    (other_room / "guide.txt").symlink_to(outside)
    original_loader = deploy._run_loader

    def run_loader(config, request):
        link = request.room_paths[1] / "room-b" / "guide.txt"
        assert link.is_symlink()
        assert link.readlink() == outside
        return original_loader(config, request)

    monkeypatch.setattr(deploy, "_run_loader", run_loader)
    assert _bind(env, ["alpha-one"]).status == "committed"
    assert outside.read_text() == "Outside content"


def test_commit_failure_restores_file_and_unstages_room(env):
    rooms = env[2]
    hook = rooms / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    previous = (rooms / ROOM / "room_config.yaml").read_bytes()
    with pytest.raises(landing_clone.GitFailed, match="commit"):
        _bind(env, ["alpha-one"])
    assert (rooms / ROOM / "room_config.yaml").read_bytes() == previous
    assert _git(rooms, "status", "--porcelain") == ""
    assert _git(rooms, "rev-list", "--count", "HEAD") == "1"
    assert env[3] == []


def test_skills_yaml_alias_leaves_other_key_unchanged(env):
    path = env[2] / ROOM / "room_config.yaml"
    text = ROOM_TEXT.replace("skills:\n", "skills: &shared\n")
    text += "other: *shared\n"
    path.write_text(text)
    before = yaml.safe_load(text)
    assert _bind(env, ["alpha-one"]).status == "committed"
    actual = yaml.safe_load(path.read_text())
    assert actual["other"] == before["skills"]
    expected = yaml.safe_load(text)
    expected["skills"] = {
        **expected["skills"],
        "installation_skill_names": [
            {"name": "alpha-one", "defer_loading": False}
        ],
    }
    assert actual == expected


def test_unbind_all_and_unchanged_skip_loader_and_reload(env, monkeypatch):
    assert _bind(env, ["alpha-one"]).status == "committed"
    assert _bind(env, ["alpha-one"]) == bindings.BindResult(
        "unchanged", None, []
    )
    assert _bind(env, []).status == "committed"
    rooms, reloads = env[2:]
    text = (rooms / ROOM / "room_config.yaml").read_text()
    assert yaml.safe_load(text)["skills"]["installation_skill_names"] == []
    monkeypatch.setattr(deploy, "_run_loader", None)
    assert _bind(env, []) == bindings.BindResult("unchanged", None, [])
    assert _git(rooms, "rev-list", "--count", "HEAD") == "3"
    assert len(reloads) == 2


def test_loader_failure_preserves_clone(env):
    config = env[1]
    (config.state_root.parent / "loader-result.json").write_text(
        loader.LoadResult(False, {}, [], [], ["Room cannot load"]).to_json()
    )
    _assert_refused(env, _bind(env, ["alpha-one"]), ["Room cannot load"])


@pytest.mark.parametrize("http_status", [None, 503])
def test_reload_failure_keeps_commit_and_fixed_problem(
    env, monkeypatch, http_status
):
    def reload(config):
        assert config == env[0]
        if http_status is None:
            message = "Sensitive server response"
            raise RuntimeError(message)
        raise urllib.error.HTTPError(
            config.backend_url, http_status, "Sensitive response", {}, None
        )

    monkeypatch.setattr(reconciler, "_reload_skills", reload)
    result = _bind(env, ["alpha-one"])
    problem = "skill reload failed"
    if http_status is not None:
        problem += f" (HTTP {http_status})"
    assert result == bindings.BindResult(
        "committed", _git(env[2], "rev-parse", "HEAD"), [problem]
    )
    assert _git(env[2], "rev-list", "--count", "HEAD") == "2"


@pytest.mark.parametrize("clone_form", ["canonical", "symlink"])
def test_locks_cover_validation_commit_and_reload(
    env, monkeypatch, clone_form
):
    reconciler_config, config, rooms, _ = env
    alias = rooms.parent / "rooms-alias"
    alias.symlink_to(rooms, target_is_directory=True)
    clone = {
        "canonical": rooms,
        "symlink": alias,
    }[clone_form]
    config = dataclasses.replace(
        config,
        room_paths=[
            config.room_paths[0],
            clone,
            alias,
            rooms / ".." / rooms.name,
        ],
    )
    _write_config(reconciler_config.deploy_config, config)
    locks = [
        config.state_root / "install.lock",
        landing_clone.lock_path(rooms),
    ]
    calls = []

    def check_locks():
        for path in locks:
            with path.open("a") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    original_loader = deploy._run_loader
    original_commit = landing_clone.commit_room

    def run_loader(config, request):
        check_locks()
        calls.append("validate")
        candidate = request.room_paths[1]
        assert candidate != rooms
        assert request.room_paths[1:] == [candidate] * 3
        assert (candidate / ".git").is_dir()
        assert (candidate / ROOM / "guide.txt").read_bytes() == (
            rooms / ROOM / "guide.txt"
        ).read_bytes()
        assert (rooms / ROOM / "room_config.yaml").read_text() == ROOM_TEXT
        assert yaml.safe_load(
            (candidate / ROOM / "room_config.yaml").read_text()
        )["skills"]["installation_skill_names"] == [
            {"name": "alpha-one", "defer_loading": False}
        ]
        return original_loader(config, request)

    def commit(*args, **kwargs):
        check_locks()
        calls.append("commit")
        return original_commit(*args, **kwargs)

    def reload(config):
        assert config == reconciler_config
        check_locks()
        calls.append("reload")

    monkeypatch.setattr(deploy, "_run_loader", run_loader)
    monkeypatch.setattr(landing_clone, "commit_room", commit)
    monkeypatch.setattr(reconciler, "_reload_skills", reload)
    assert (
        bindings.set_bindings(
            reconciler_config,
            clone,
            ROOM,
            ["alpha-one"],
            team="alpha",
            **AUTHOR,
        ).status
        == "committed"
    )
    assert calls == ["validate", "commit", "reload"]
    for path in locks:
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_visibility_uses_committed_version(env):
    repo = env[1].repos["beta"]
    path = repo / "beta-private" / "SKILL.md"
    path.write_text(path.read_text().replace("private", "public"))
    _assert_refused(
        env,
        _bind(env, ["beta-private"]),
        ["Skill is private to beta: beta-private"],
    )


def test_invalid_committed_skill_cannot_grant_public_access(env):
    config = env[1]
    version = DeployState(config.state_root).read_generation(1)["teams"][
        "beta"
    ]
    path = config.state_root / version / "beta-public" / "SKILL.md"
    path.chmod(0o644)
    path.write_text("Invalid skill")
    _assert_refused(
        env,
        _bind(env, ["beta-public"]),
        ["Skill is private to beta: beta-public"],
    )


def test_room_without_skills_mapping(env):
    path = env[2] / ROOM / "room_config.yaml"
    path.write_text(f"id: {ROOM}\n")
    assert _bind(env, ["alpha-one"]).status == "committed"
    assert yaml.safe_load(path.read_text()) == {
        "id": ROOM,
        "skills": {
            "installation_skill_names": [
                {"name": "alpha-one", "defer_loading": False}
            ]
        },
    }


@pytest.mark.parametrize(
    "data",
    [
        b"id: [\n",
        b"date: 2026-02-30\n",
        b"- room-a\n",
        b"id: room-a\nskills: 5\n",
        b"id: room-a\nskills: null\n",
        b"",
        b"\xff",
    ],
)
def test_invalid_room_yaml_refused(env, data):
    rooms = env[2]
    path = rooms / ROOM / "room_config.yaml"
    path.write_bytes(data)
    before = _git(rooms, "status", "--porcelain")
    assert _bind(env, ["alpha-one"]) == bindings.BindResult(
        "refused", None, ["Invalid room_config.yaml"]
    )
    assert path.read_bytes() == data
    assert _git(rooms, "status", "--porcelain") == before
    assert _git(rooms, "rev-list", "--count", "HEAD") == "1"
    assert env[3] == []
    assert not (rooms.parent / "request.json").exists()


def test_cli_exit_codes_and_repeatable_skills(env, capsys, monkeypatch):
    config, deploy_config, rooms, _ = env
    config_path = deploy_config.state_root.parent / "reconciler.json"
    config_path.write_text(json.dumps(dataclasses.asdict(config), default=str))
    args = [
        "bind",
        "--config",
        str(config_path),
        "--rooms-clone",
        str(rooms),
        "--room",
        ROOM,
        "--author-name",
        AUTHOR["author_name"],
        "--author-email",
        AUTHOR["author_email"],
    ]
    names = ["--skill", "alpha-one", "--skill", "beta-private"]
    monkeypatch.setattr(sys, "argv", ["skills-bind", *args, *names])
    assert bindings.main() == 0
    assert capsys.readouterr().out == "committed\n"
    assert yaml.safe_load((rooms / ROOM / "room_config.yaml").read_text())[
        "skills"
    ]["installation_skill_names"] == [
        {"name": "alpha-one", "defer_loading": False},
        {"name": "beta-private", "defer_loading": False},
    ]
    assert bindings.main([*args, *names]) == 0
    assert capsys.readouterr().out == "unchanged\n"
    assert bindings.main([*args, "--skill", "unknown"]) == 1
    assert capsys.readouterr().out == "refused\nUnknown skill: unknown\n"
    assert (
        bindings.main([*args, "--team", "alpha", "--skill", "beta-private"])
        == 1
    )
    assert capsys.readouterr().out == (
        "refused\nSkill is private to beta: beta-private\n"
    )
    assert bindings.main(args) == 0
    assert capsys.readouterr().out == "committed\n"
    assert (
        yaml.safe_load((rooms / ROOM / "room_config.yaml").read_text())[
            "skills"
        ]["installation_skill_names"]
        == []
    )


def test_bind_result_is_frozen():
    result = bindings.BindResult("unchanged", None, [])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "committed"
