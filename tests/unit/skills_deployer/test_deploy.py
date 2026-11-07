"""Test deploy attempts, generation validation and command line reports."""

import dataclasses
import fcntl
import io
import json
import pathlib
import subprocess
import sys

import pytest

from soliplex_concierge.skills_deployer import deploy
from soliplex_concierge.skills_deployer import loader
from soliplex_concierge.skills_deployer import manifest
from soliplex_concierge.skills_deployer.model import Check
from soliplex_concierge.skills_deployer.model import Limits
from soliplex_concierge.skills_deployer.model import Violation
from soliplex_concierge.skills_deployer.state import DeployState
from soliplex_concierge.skills_deployer.state import Target


def _git(repo, *args):
    """Run git against a real repository and return its output."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _skill(name, body="Use this skill."):
    """Build a skill file with valid frontmatter and instructions."""
    return f"---\nname: {name}\ndescription: Test skill\n---\n{body}\n"


def _commit(work, files):
    """Commit files in a team's working repository."""
    for name, text in files.items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(work, "-c", "core.autocrlf=false", "add", "--all")
    _git(
        work,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--message=test",
    )


def _repo(root, team, files):
    """Build a bare team repository from committed files."""
    work = root / f"{team}-work"
    work.mkdir()
    _git(work, "init", "--quiet", "--initial-branch=main")
    _commit(work, files)
    bare = root / f"{team}.git"
    _git(root, "clone", "--quiet", "--bare", str(work), str(bare))
    return bare


def _push(repo, files):
    """Commit and push a new main revision to a bare team repository."""
    work = repo.with_name(f"{repo.stem}-work")
    _commit(work, files)
    _git(work, "push", "--quiet", str(repo), "HEAD:refs/heads/main")


def _fake_loader(root, *, output=None, exit_code=0, delay=0):
    """Write a loader script that records requests and emits a result."""
    if output is None:
        output = loader.LoadResult(True, {}, ["room-a"], [], []).to_json()
    script = root / "fake_loader.py"
    script.write_text(
        "import json, pathlib, sys, time\n"
        "request_path = pathlib.Path(sys.argv[1])\n"
        "request = json.loads(request_path.read_text())\n"
        "path = pathlib.Path(__file__).with_suffix('.request.json')\n"
        "path.write_text(json.dumps(request))\n"
        "path.with_suffix('.path').write_text(str(request_path))\n"
        f"time.sleep({delay!r})\n"
        f"print({output!r})\n"
        f"sys.exit({exit_code!r})\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


@pytest.fixture
def config(tmp_path):
    repo = _repo(
        tmp_path,
        "alpha",
        {
            "alpha-one/SKILL.md": _skill("alpha-one"),
            "alpha-one/resources/nested/guide.txt": "Read the guide.\n",
        },
    )
    installation = tmp_path / "installation"
    installation.mkdir()
    (installation / "installation.yaml").write_text(
        "id: test-installation\noidc_paths: []\n", encoding="utf-8"
    )
    baseline = tmp_path / "baseline"
    (baseline / "soliplex-docs").mkdir(parents=True)
    (baseline / "soliplex-docs" / "SKILL.md").write_text(
        _skill("soliplex-docs"), encoding="utf-8"
    )
    rooms = tmp_path / "rooms"
    rooms.mkdir()
    return deploy.DeployConfig(
        state_root=tmp_path / "state",
        repos={"alpha": repo},
        installation=installation,
        baseline_paths=[baseline],
        baseline_names={"soliplex-docs"},
        room_paths=[rooms],
        room_revision="rooms-1",
        loader_command=_fake_loader(tmp_path),
    )


def _config_file(config, *, defaults=False):
    """Write a configuration file for command line tests."""
    data = dataclasses.asdict(config)
    data["baseline_names"] = sorted(config.baseline_names)
    if defaults:
        del data["limits"]
        del data["loader_timeout"]
    path = config.state_root.parent / "config.json"
    path.write_text(json.dumps(data, default=str), encoding="utf-8")
    return path


def _request(config):
    """Read the fake loader's captured request and check temporary cleanup."""
    path = pathlib.Path(config.loader_command[-1]).with_suffix(".request.json")
    assert not pathlib.Path(path.with_suffix(".path").read_text()).exists()
    return json.loads(path.read_text())


def _versions(config):
    """List published version directories under deployment state."""
    return list((config.state_root / "versions").glob("*/*"))


def test_first_deploy_and_unchanged(config):
    outcome = deploy.deploy(config, "alpha")
    revision = _git(config.repos["alpha"], "rev-parse", "refs/heads/main")
    assert outcome == deploy.Outcome(
        "alpha", revision, "accepted", Target(1, 1, "rooms-1"), [], []
    )
    state = DeployState(config.state_root)
    assert state.read_target() == outcome.target
    version = f"versions/alpha/{revision}-1"
    skills = {
        "alpha-one": {
            "owner": "alpha",
            "digest": loader.instruction_digest("Use this skill."),
        }
    }
    generation = state.read_generation(1)
    assert generation == {
        "teams": {"alpha": version},
        "baseline": {str(config.baseline_paths[0]): ""},
        "allow_list": ["alpha-one", "soliplex-docs"],
        "room_revision": "rooms-1",
        "loader_result": {
            "loader_result": 1,
            "ok": True,
            "skills": skills,
            "rooms": ["room-a"],
            "shadowed": [],
            "problems": [],
        },
    }
    request = _request(config)
    assert request == {
        "installation": str(config.installation),
        "team_paths": {"alpha": str(config.state_root / version)},
        "baseline_paths": [str(p) for p in config.baseline_paths],
        "allow_list": generation["allow_list"],
        "room_paths": [str(p) for p in config.room_paths],
        "expected": skills,
    }
    assert deploy.deploy(config, "alpha") == dataclasses.replace(
        outcome, status="unchanged"
    )
    assert len(_versions(config)) == 1
    assert outcome.lines() == [f"alpha: accepted {revision}"]


@pytest.mark.parametrize("defaults", [False, True])
def test_config_round_trip(config, defaults):
    if not defaults:
        config = dataclasses.replace(
            config, limits=Limits(max_skills=3), loader_timeout=7.5
        )
    assert (
        deploy.DeployConfig.from_json(_config_file(config, defaults=defaults))
        == config
    )


def test_structure_rejection_and_cached_context(config):
    _push(config.repos["alpha"], {"README.md": "Remove this file.\n"})
    outcome = deploy.deploy(config, "alpha")
    assert outcome.status == "rejected"
    assert outcome.target is None
    assert outcome.problems == []
    assert any(v.check == Check.LAYOUT for v in outcome.violations)
    assert outcome.lines() == [
        f"alpha: rejected {outcome.revision}",
        *(
            f"alpha: {v.check} {v.path}: {v.message}"
            for v in outcome.violations
        ),
    ]
    assert _versions(config) == []
    assert deploy.deploy(config, "alpha").status == "unchanged"
    changed = dataclasses.replace(config, room_revision="rooms-2")
    assert deploy.deploy(changed, "alpha").status == "rejected"
    assert DeployState(config.state_root).rejection("alpha")["context"] == {
        "generation": 0,
        "room_revision": "rooms-2",
        "validator": 1,
    }


def test_content_rejection_leaves_versions_empty(config):
    _push(
        config.repos["alpha"],
        {"alpha-one/SKILL.md": _skill("alpha-one", body="")},
    )
    outcome = deploy.deploy(config, "alpha")
    assert outcome.status == "rejected"
    assert any(v.check == Check.FRONTMATTER for v in outcome.violations)
    assert list((config.state_root / "versions").iterdir()) == []
    assert DeployState(config.state_root).rejection("alpha")["violations"]


@pytest.mark.parametrize(
    "ok,exit_code,problems,expected",
    [
        (False, 1, ["Room cannot load"], ["Room cannot load"]),
        (False, 0, ["Bad digest"], ["Bad digest"]),
        (True, 1, [], ["Loader rejected generation (exit 1)"]),
        (False, 0, [], ["Loader rejected generation (exit 0)"]),
        (True, 0, ["Unexpected problem"], ["Unexpected problem"]),
    ],
)
def test_loader_rejections(
    config, tmp_path, ok, exit_code, problems, expected
):
    output = loader.LoadResult(ok, {}, [], [], problems).to_json()
    config = dataclasses.replace(
        config,
        loader_command=_fake_loader(
            tmp_path, output=output, exit_code=exit_code
        ),
    )
    outcome = deploy.deploy(config, "alpha")
    assert outcome.status == "rejected"
    assert outcome.problems == expected
    assert outcome.violations == []
    assert outcome.lines() == [
        f"alpha: rejected {outcome.revision}",
        *(f"alpha: {problem}" for problem in expected),
    ]
    assert _versions(config) == []
    _request(config)
    state = DeployState(config.state_root)
    assert state.read_target() is None
    assert state.rejection("alpha")["violations"] == [
        {"check": "C9", "path": "", "message": p} for p in expected
    ]
    assert deploy.deploy(config, "alpha").status == "unchanged"


def test_loader_timeout(config, tmp_path):
    config = dataclasses.replace(
        config,
        loader_command=_fake_loader(tmp_path, delay=30),
        loader_timeout=0.2,
    )
    outcome = deploy.deploy(config, "alpha")
    assert outcome.status == "rejected"
    assert outcome.problems == ["Loader timed out after 0.2 seconds"]
    assert _versions(config) == []
    _request(config)


@pytest.mark.parametrize(
    "output",
    ["not json", "{}", "[]", "null", '{"ok": true}\nextra', "\udcff"],
)
def test_malformed_loader_output(config, tmp_path, output):
    config = dataclasses.replace(
        config, loader_command=_fake_loader(tmp_path, output=output)
    )
    outcome = deploy.deploy(config, "alpha")
    assert outcome.status == "rejected"
    assert outcome.problems == [
        "Loader output must be a LoadResult JSON document"
    ]
    assert _versions(config) == []
    _request(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("loader_result", True),
        ("loader_result", 2),
        ("ok", 1),
        ("skills", []),
        ("skills", {"alpha-one": None}),
        ("rooms", "room-a"),
        ("rooms", [1]),
        ("shadowed", {}),
        ("shadowed", ["abc"]),
        ("shadowed", [["a", "b"]]),
        ("shadowed", [["a", "b", 3]]),
        ("problems", [False]),
    ],
)
def test_loader_result_schema(config, tmp_path, field, value):
    data = json.loads(loader.LoadResult(True, {}, [], [], []).to_json())
    data[field] = value
    config = dataclasses.replace(
        config, loader_command=_fake_loader(tmp_path, output=json.dumps(data))
    )
    assert deploy.deploy(config, "alpha").status == "rejected"
    assert _versions(config) == []


def test_loader_result_metadata(config, tmp_path):
    result = loader.LoadResult(
        True,
        {"alpha-one": {"owner": "alpha", "path": "from-loader"}},
        ["room-a"],
        [("room-a", "first.yaml", "second.yaml")],
        [],
    )
    config = dataclasses.replace(
        config, loader_command=_fake_loader(tmp_path, output=result.to_json())
    )
    assert deploy.deploy(config, "alpha").status == "accepted"
    saved = DeployState(config.state_root).read_generation(1)["loader_result"]
    assert saved["shadowed"] == [["room-a", "first.yaml", "second.yaml"]]
    assert saved["skills"] == {
        "alpha-one": {
            "owner": "alpha",
            "digest": loader.instruction_digest("Use this skill."),
        }
    }


def test_two_teams_preserve_skills_and_reject_collision(
    config, tmp_path, monkeypatch
):
    beta = _repo(tmp_path, "beta", {"beta-one/SKILL.md": _skill("beta-one")})
    config = dataclasses.replace(config, repos={**config.repos, "beta": beta})
    assert deploy.deploy(config, "alpha").status == "accepted"
    seen_names = []
    original = manifest.check_structure

    def check_structure(team, entries, *, other_names, limits):
        seen_names.append(other_names)
        return original(team, entries, other_names=other_names, limits=limits)

    monkeypatch.setattr(manifest, "check_structure", check_structure)
    outcome = deploy.deploy(config, "beta")
    assert outcome.status == "accepted"
    assert outcome.target == Target(2, 2, "rooms-1")
    assert seen_names == [{"alpha-one", "soliplex-docs"}]
    request = _request(config)
    assert set(request["team_paths"]) == {"alpha", "beta"}
    assert request["expected"] == {
        "alpha-one": {
            "owner": "alpha",
            "digest": loader.instruction_digest("Use this skill."),
        },
        "beta-one": {
            "owner": "beta",
            "digest": loader.instruction_digest("Use this skill."),
        },
    }
    assert request["allow_list"] == ["alpha-one", "beta-one", "soliplex-docs"]
    _push(beta, {"alpha-one/SKILL.md": _skill("alpha-one")})
    rejected = deploy.deploy(config, "beta")
    assert rejected.status == "rejected"
    assert rejected.target == outcome.target
    assert any(v.check == Check.COLLISION for v in rejected.violations)
    assert seen_names[-1] == {"alpha-one", "soliplex-docs"}
    assert DeployState(config.state_root).read_target() == outcome.target


def test_generation_context_retries_loader_rejection(config, tmp_path):
    bad = dataclasses.replace(
        config, loader_command=_fake_loader(tmp_path, output="broken")
    )
    assert deploy.deploy(bad, "alpha").status == "rejected"
    config = dataclasses.replace(config, loader_command=_fake_loader(tmp_path))
    assert deploy.deploy(config, "alpha").status == "unchanged"
    beta = _repo(tmp_path, "beta", {"beta-one/SKILL.md": _skill("beta-one")})
    config = dataclasses.replace(config, repos={**config.repos, "beta": beta})
    assert deploy.deploy(config, "beta").status == "accepted"
    outcome = deploy.deploy(config, "alpha")
    assert outcome.status == "accepted"
    assert outcome.target == Target(2, 2, "rooms-1")


def test_update_replaces_own_expected_skills(config):
    first = deploy.deploy(config, "alpha")
    _push(
        config.repos["alpha"],
        {"alpha-one/SKILL.md": _skill("alpha-one", body="New instructions.")},
    )
    second = deploy.deploy(config, "alpha")
    assert second.status == "accepted"
    assert second.target.epoch == 2
    assert second.revision != first.revision
    assert _request(config)["expected"]["alpha-one"]["digest"] == (
        loader.instruction_digest("New instructions.")
    )


def test_sequence_counts_existing_revision_directories(config):
    first = deploy.deploy(config, "alpha")
    state = DeployState(config.state_root)
    state.pin(first.target)
    _push(
        config.repos["alpha"],
        {"alpha-one/SKILL.md": _skill("alpha-one", body="New instructions.")},
    )
    assert deploy.deploy(config, "alpha").status == "accepted"
    _git(
        config.repos["alpha"], "update-ref", "refs/heads/main", first.revision
    )
    (
        config.state_root / "versions" / "alpha" / f"{first.revision}-file"
    ).touch()
    outcome = deploy.deploy(config, "alpha")
    assert outcome.status == "accepted"
    assert outcome.target == Target(3, 3, "rooms-1")
    assert state.read_generation(3)["teams"]["alpha"] == (
        f"versions/alpha/{first.revision}-2"
    )


def test_baseline_collision_and_configured_limits(config):
    baseline = dataclasses.replace(config, baseline_names={"alpha-one"})
    outcome = deploy.deploy(baseline, "alpha")
    assert any(v.check == Check.COLLISION for v in outcome.violations)
    limited = dataclasses.replace(
        config, limits=Limits(max_skills=0), room_revision="rooms-2"
    )
    outcome = deploy.deploy(limited, "alpha")
    assert any(v.check == Check.LIMITS for v in outcome.violations)


def test_lock_covers_recovery_and_attempt(config, monkeypatch):
    original = DeployState.recover
    calls = []

    def recover(state):
        with (state.root / "install.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append("recover")
        return original(state)

    monkeypatch.setattr(DeployState, "recover", recover)
    assert deploy.deploy(config, "alpha").status == "accepted"
    assert calls == ["recover"]
    with (config.state_root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_main_deploy(config, capsys):
    path = _config_file(config)
    args = ["deploy", "--config", str(path), "--team", "alpha"]
    assert deploy.main(args) == 0
    assert "alpha: accepted " in capsys.readouterr().out
    assert deploy.main(args) == 0
    assert "alpha: unchanged " in capsys.readouterr().out
    _push(config.repos["alpha"], {"README.md": "Invalid layout"})
    assert deploy.main(args) == 1
    assert "alpha: rejected " in capsys.readouterr().out


def test_main_reconcile_sorted_and_aggregate_failure(config, tmp_path, capsys):
    beta = _repo(tmp_path, "beta", {"README.md": "Invalid layout"})
    config = dataclasses.replace(
        config, repos={"beta": beta, "alpha": config.repos["alpha"]}
    )
    args = ["reconcile", "--config", str(_config_file(config))]
    assert deploy.main(args) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("alpha: accepted ")
    assert lines[1].startswith("beta: rejected ")
    assert deploy.main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("alpha: unchanged ")
    assert lines[1].startswith("beta: unchanged ")


def test_main_hook_drains_stdin(config, capsys, monkeypatch):
    stream = io.StringIO("old new refs/heads/main\nold new refs/heads/topic\n")
    monkeypatch.setattr(sys, "stdin", stream)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skills-deploy",
            "hook",
            "--config",
            str(_config_file(config)),
            "--team",
            "alpha",
        ],
    )
    assert deploy.main() == 0
    assert stream.read() == ""
    assert capsys.readouterr().out.startswith("alpha: accepted ")


def test_real_loader_in_separate_process(config):
    command = [
        sys.executable,
        "-c",
        "import sys; from soliplex_concierge.skills_deployer import loader; "
        "sys.exit(loader.main(sys.argv[1:]))",
    ]
    config = dataclasses.replace(config, loader_command=command)
    outcome = deploy.deploy(config, "alpha")
    assert outcome.status == "accepted", outcome.lines()
    assert outcome.target == Target(1, 1, "rooms-1")
    skills = DeployState(config.state_root).read_generation(1)[
        "loader_result"
    ]["skills"]
    assert skills == {
        "alpha-one": {
            "owner": "alpha",
            "digest": loader.instruction_digest("Use this skill."),
        }
    }


def test_outcome_combines_diagnostics():
    outcome = deploy.Outcome(
        "alpha",
        "revision",
        "rejected",
        None,
        [Violation(Check.NAME, "bad", "Choose an alpha- name")],
        ["Loader cannot start"],
    )
    assert outcome.lines() == [
        "alpha: rejected revision",
        "alpha: C2 bad: Choose an alpha- name",
        "alpha: Loader cannot start",
    ]


def test_bootstrap_accepts_all_teams_and_blocks_repeat(
    config, tmp_path, monkeypatch
):
    beta = _repo(tmp_path, "beta", {"beta-one/SKILL.md": _skill("beta-one")})
    config = dataclasses.replace(
        config, repos={"beta": beta, "alpha": config.repos["alpha"]}
    )
    missing = loader.LoadResult(
        False, {}, [], [], ["Room requires both teams"]
    ).to_json()
    single = dataclasses.replace(
        config,
        loader_command=_fake_loader(tmp_path, output=missing, exit_code=1),
    )
    assert deploy.deploy(single, "alpha").status == "rejected"
    assert deploy.deploy(single, "beta").status == "rejected"
    both = loader.LoadResult(True, {}, ["room-a", "room-b"], [], []).to_json()
    config = dataclasses.replace(
        config, loader_command=_fake_loader(tmp_path, output=both)
    )
    seen_names = []
    original = manifest.check_structure

    def check_structure(team, entries, *, other_names, limits):
        seen_names.append((team, other_names))
        return original(team, entries, other_names=other_names, limits=limits)

    monkeypatch.setattr(manifest, "check_structure", check_structure)
    outcomes = deploy.bootstrap(config)
    alpha_rev = _git(config.repos["alpha"], "rev-parse", "refs/heads/main")
    beta_rev = _git(beta, "rev-parse", "refs/heads/main")
    target = Target(1, 1, "rooms-1")
    assert outcomes == [
        deploy.Outcome("alpha", alpha_rev, "accepted", target, [], []),
        deploy.Outcome("beta", beta_rev, "accepted", target, [], []),
    ]
    assert seen_names == [
        ("alpha", {"beta-one", "soliplex-docs"}),
        ("beta", {"alpha-one", "soliplex-docs"}),
    ]
    state = DeployState(config.state_root)
    assert state.read_target() == target
    assert set(state.read_generation(1)["teams"]) == {"alpha", "beta"}
    assert set(_request(config)["team_paths"]) == {"alpha", "beta"}
    assert deploy.deploy(config, "alpha") == dataclasses.replace(
        outcomes[0], status="unchanged"
    )
    assert deploy.deploy(config, "beta") == dataclasses.replace(
        outcomes[1], status="unchanged"
    )
    with pytest.raises(RuntimeError, match="^already bootstrapped$"):
        deploy.bootstrap(config)


def test_bootstrap_structure_and_content_failures(config, tmp_path):
    beta = _repo(tmp_path, "beta", {"README.md": "Invalid layout"})
    config = dataclasses.replace(
        config, repos={"beta": beta, "alpha": config.repos["alpha"]}
    )
    outcomes = deploy.bootstrap(config)
    assert [o.team for o in outcomes] == ["beta"]
    assert outcomes[0].status == "rejected"
    assert any(v.check == Check.LAYOUT for v in outcomes[0].violations)
    assert _versions(config) == []
    request_file = pathlib.Path(config.loader_command[-1]).with_suffix(
        ".request.json"
    )
    assert not request_file.exists()
    state = DeployState(config.state_root)
    assert state.read_target() is None
    assert state.rejection("beta") is None
    _git(beta.with_name("beta-work"), "rm", "--quiet", "README.md")
    _push(beta, {"beta-one/SKILL.md": _skill("beta-one", body="")})
    outcomes = deploy.bootstrap(config)
    assert [o.team for o in outcomes] == ["beta"]
    assert outcomes[0].status == "rejected"
    assert any(v.check == Check.FRONTMATTER for v in outcomes[0].violations)
    assert _versions(config) == []


def test_bootstrap_loader_failure_removes_every_version(config, tmp_path):
    beta = _repo(tmp_path, "beta", {"beta-one/SKILL.md": _skill("beta-one")})
    output = loader.LoadResult(
        False, {}, [], [], ["Room cannot load"]
    ).to_json()
    config = dataclasses.replace(
        config,
        repos={"alpha": config.repos["alpha"], "beta": beta},
        loader_command=_fake_loader(tmp_path, output=output, exit_code=1),
    )
    outcomes = deploy.bootstrap(config)
    assert [o.team for o in outcomes] == ["alpha", "beta"]
    assert all(o.status == "rejected" for o in outcomes)
    assert all(o.problems == ["Room cannot load"] for o in outcomes)
    assert _versions(config) == []
    _request(config)
    state = DeployState(config.state_root)
    assert state.read_target() is None
    assert state.rejection("alpha") is None
    assert state.rejection("beta") is None


def test_main_bootstrap(config, tmp_path, capsys):
    beta = _repo(tmp_path, "beta", {"README.md": "Invalid layout"})
    config = dataclasses.replace(
        config, repos={"alpha": config.repos["alpha"], "beta": beta}
    )
    args = ["bootstrap", "--config", str(_config_file(config))]
    assert deploy.main(args) == 1
    assert "beta: rejected " in capsys.readouterr().out
    _git(beta.with_name("beta-work"), "rm", "--quiet", "README.md")
    _push(beta, {"beta-one/SKILL.md": _skill("beta-one")})
    assert deploy.main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("alpha: accepted ")
    assert lines[1].startswith("beta: accepted ")
    assert deploy.main(args) == 1
    assert capsys.readouterr().out.strip() == "already bootstrapped"
