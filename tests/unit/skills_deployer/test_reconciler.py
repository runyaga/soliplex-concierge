"""Test mirror synchronization, deploy reconciliation and backend reloads."""

import dataclasses
import io
import json
import subprocess
import sys
import urllib.error

import pytest

from soliplex_concierge.provisioner import soliplex_api
from soliplex_concierge.skills_deployer import deploy
from soliplex_concierge.skills_deployer import loader
from soliplex_concierge.skills_deployer import reconciler
from soliplex_concierge.skills_deployer.state import DeployState
from soliplex_concierge.skills_deployer.state import Target

BACKEND = "http://backend:8000"
SECRET = "test" * 8


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


def _remote(root, team, files):
    """Build a bare remote repository from committed files."""
    work = root / f"{team}-work"
    work.mkdir()
    _git(work, "init", "--quiet", "--initial-branch=main")
    _commit(work, files)
    bare = root / "remotes" / f"{team}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    _git(root, "clone", "--quiet", "--bare", str(work), str(bare))
    return bare


def _push(remote, files):
    """Commit and push a new main revision to a bare remote repository."""
    work = remote.parents[1] / f"{remote.stem}-work"
    _commit(work, files)
    _git(work, "push", "--quiet", str(remote), "HEAD:refs/heads/main")


def _fake_loader(root):
    """Write a loader script that emits an accepted LoadResult."""
    output = loader.LoadResult(True, {}, ["room-a"], [], []).to_json()
    script = root / "fake_loader.py"
    script.write_text(
        f"print({output!r})\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def _write_deploy_config(tmp_path, teams):
    """Write a DeployConfig JSON file whose repos are local mirror paths."""
    installation = tmp_path / "installation"
    installation.mkdir(exist_ok=True)
    (installation / "installation.yaml").write_text(
        "id: test-installation\noidc_paths: []\n", encoding="utf-8"
    )
    baseline = tmp_path / "baseline"
    (baseline / "soliplex-docs").mkdir(parents=True, exist_ok=True)
    (baseline / "soliplex-docs" / "SKILL.md").write_text(
        _skill("soliplex-docs"), encoding="utf-8"
    )
    rooms = tmp_path / "rooms"
    rooms.mkdir(exist_ok=True)
    config = deploy.DeployConfig(
        state_root=tmp_path / "state",
        repos={team: tmp_path / "mirrors" / f"{team}.git" for team in teams},
        installation=installation,
        baseline_paths=[baseline],
        baseline_names={"soliplex-docs"},
        room_paths=[rooms],
        room_revision="rooms-1",
        loader_command=_fake_loader(tmp_path),
    )
    data = dataclasses.asdict(config)
    data["baseline_names"] = sorted(config.baseline_names)
    path = tmp_path / "deploy.json"
    path.write_text(json.dumps(data, default=str), encoding="utf-8")
    return path, config


def _write_reconciler_config(config, *, defaults=False):
    """Write a ReconcilerConfig JSON file for CLI and round-trip tests."""
    data = {
        "deploy_config": str(config.deploy_config),
        "remotes": config.remotes,
        "backend_url": config.backend_url,
        "secret_file": str(config.secret_file),
        "service_subject": config.service_subject,
        "interval": config.interval,
        "git_timeout": config.git_timeout,
    }
    if defaults:
        del data["service_subject"]
        del data["interval"]
        del data["git_timeout"]
    path = config.deploy_config.parent / "reconciler.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def calls(monkeypatch):
    """Record backend API calls made through soliplex_api.call."""
    recorded = []

    def fake_call(base_url, method, path, *, token=None):
        recorded.append((base_url, method, path, token))
        return {}

    monkeypatch.setattr(soliplex_api, "call", fake_call)
    return recorded


@pytest.fixture
def env(tmp_path):
    """Build two remote team repositories and a reconciler configuration."""
    alpha = _remote(
        tmp_path, "alpha", {"alpha-one/SKILL.md": _skill("alpha-one")}
    )
    beta = _remote(tmp_path, "beta", {"beta-one/SKILL.md": _skill("beta-one")})
    deploy_path, deploy_config = _write_deploy_config(
        tmp_path, ("alpha", "beta")
    )
    secret_file = tmp_path / "secret"
    secret_file.write_text(f"{SECRET}\n", encoding="utf-8")
    config = reconciler.ReconcilerConfig(
        deploy_config=deploy_path,
        remotes={"alpha": str(alpha), "beta": str(beta)},
        backend_url=BACKEND,
        secret_file=secret_file,
    )
    return config, deploy_config, {"alpha": alpha, "beta": beta}


@pytest.mark.parametrize("defaults", [False, True])
def test_config_round_trip(env, defaults):
    config, _deploy_config, _remotes = env
    if not defaults:
        config = dataclasses.replace(
            config,
            service_subject="custom-subject",
            interval=15.0,
            git_timeout=30.0,
        )
    path = _write_reconciler_config(config, defaults=defaults)
    assert reconciler.ReconcilerConfig.from_json(path) == config


@pytest.mark.parametrize(
    "timeout", [None, 0, -1, True, float("inf"), float("nan"), "12"]
)
def test_invalid_git_timeout_rejected_directly_and_from_json(env, timeout):
    config, _deploy_config, _remotes = env
    with pytest.raises(ValueError, match="git_timeout"):
        dataclasses.replace(config, git_timeout=timeout)

    path = _write_reconciler_config(config)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["git_timeout"] = timeout
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="git_timeout"):
        reconciler.ReconcilerConfig.from_json(path)


def test_first_run_clones_mirrors_and_updates_fetch_new_commit(env, calls):
    config, deploy_config, remotes = env
    assert not deploy_config.repos["alpha"].exists()
    assert not deploy_config.repos["beta"].exists()

    outcomes = reconciler.run_once(config)

    alpha_rev = _git(remotes["alpha"], "rev-parse", "refs/heads/main")
    beta_rev = _git(remotes["beta"], "rev-parse", "refs/heads/main")
    assert deploy_config.repos["alpha"].is_dir()
    assert deploy_config.repos["beta"].is_dir()
    assert outcomes == [
        deploy.Outcome(
            "alpha", alpha_rev, "accepted", Target(1, 1, "rooms-1"), [], []
        ),
        deploy.Outcome(
            "beta", beta_rev, "accepted", Target(2, 2, "rooms-1"), [], []
        ),
    ]
    assert len(calls) == 1
    base_url, method, path, token = calls[0]
    assert (base_url, method, path) == (
        BACKEND,
        "POST",
        "/v1/installation/reload/skills",
    )
    claims = soliplex_api.validate_service_token(SECRET, token)
    assert claims["sub"] == "soliplex-provisioner"

    outcomes = reconciler.run_once(config)
    assert [o.status for o in outcomes] == ["unchanged", "unchanged"]
    assert len(calls) == 1

    _push(
        remotes["alpha"],
        {"alpha-one/SKILL.md": _skill("alpha-one", body="Updated.")},
    )
    new_rev = _git(remotes["alpha"], "rev-parse", "refs/heads/main")
    assert new_rev != alpha_rev

    outcomes = reconciler.run_once(config)
    assert outcomes == [
        deploy.Outcome(
            "alpha", new_rev, "accepted", Target(3, 3, "rooms-1"), [], []
        ),
        deploy.Outcome(
            "beta", beta_rev, "unchanged", Target(3, 3, "rooms-1"), [], []
        ),
    ]
    assert len(calls) == 2


def test_configured_subject_mints_token(env, calls):
    config, _deploy_config, _remotes = env
    config = dataclasses.replace(config, service_subject="skills-reconciler")

    reconciler.run_once(config)

    [(_base_url, _method, _path, token)] = calls
    claims = soliplex_api.validate_service_token(SECRET, token)
    assert claims["sub"] == "skills-reconciler"


def test_unfetchable_remote_reported_and_others_deploy(env, calls, tmp_path):
    config, deploy_config, remotes = env
    missing = tmp_path / "remotes" / "missing.git"
    config = dataclasses.replace(
        config,
        remotes={"alpha": str(missing), "beta": str(remotes["beta"])},
    )

    outcomes = reconciler.run_once(config)

    beta_rev = _git(remotes["beta"], "rev-parse", "refs/heads/main")
    assert [o.team for o in outcomes] == ["alpha", "beta"]
    assert outcomes[0].status == "rejected"
    assert outcomes[0].revision == ""
    assert outcomes[0].target is None
    assert outcomes[0].problems
    assert outcomes[0].lines()[0] == "alpha: rejected "
    assert outcomes[1] == deploy.Outcome(
        "beta", beta_rev, "accepted", Target(1, 1, "rooms-1"), [], []
    )
    assert DeployState(deploy_config.state_root).read_target() == Target(
        1, 1, "rooms-1"
    )
    assert len(calls) == 1


def test_no_accepted_outcome_skips_reload(env, calls, tmp_path):
    config, _deploy_config, remotes = env
    _push(remotes["alpha"], {"README.md": "Invalid layout\n"})
    missing = tmp_path / "remotes" / "missing.git"
    config = dataclasses.replace(
        config,
        remotes={"alpha": str(remotes["alpha"]), "beta": str(missing)},
    )

    outcomes = reconciler.run_once(config)

    assert [o.status for o in outcomes] == ["rejected", "rejected"]
    assert calls == []


def test_failing_reload_reported_without_undoing_commit(
    env, calls, monkeypatch
):
    config, deploy_config, remotes = env
    reconciler.run_once(
        dataclasses.replace(config, remotes={"beta": str(remotes["beta"])})
    )

    def fail_call(base_url, method, path, *, token=None):
        raise soliplex_api.RequestFailed(
            f"{base_url}/api{path}", "backend down"
        )

    monkeypatch.setattr(soliplex_api, "call", fail_call)

    outcomes = reconciler.run_once(config)

    alpha_rev = _git(remotes["alpha"], "rev-parse", "refs/heads/main")
    beta_rev = _git(remotes["beta"], "rev-parse", "refs/heads/main")
    problem = "skill reload failed"
    assert outcomes == [
        deploy.Outcome(
            "alpha",
            alpha_rev,
            "accepted",
            Target(2, 2, "rooms-1"),
            [],
            [problem],
        ),
        deploy.Outcome(
            "beta",
            beta_rev,
            "unchanged",
            Target(2, 2, "rooms-1"),
            [],
            [],
        ),
    ]
    assert outcomes[0].lines() == [
        f"alpha: accepted {alpha_rev}",
        f"alpha: {problem}",
    ]
    state = DeployState(deploy_config.state_root)
    assert state.read_target() == Target(2, 2, "rooms-1")
    assert state.committed_revision("alpha") == alpha_rev


def test_main_once_exit_codes(env, calls, capsys):
    config, _deploy_config, remotes = env
    path = _write_reconciler_config(config)
    args = ["once", "--config", str(path)]

    assert reconciler.main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("alpha: accepted ")
    assert lines[1].startswith("beta: accepted ")
    assert len(calls) == 1

    assert reconciler.main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("alpha: unchanged ")
    assert lines[1].startswith("beta: unchanged ")
    assert len(calls) == 1

    _push(remotes["alpha"], {"README.md": "Invalid layout\n"})
    assert reconciler.main(args) == 1
    output = capsys.readouterr().out
    assert "alpha: rejected " in output
    assert len(calls) == 1


def test_main_loop_runs_until_interrupted(env, calls, monkeypatch, capsys):
    config, _deploy_config, remotes = env
    config = dataclasses.replace(config, interval=7.5)
    path = _write_reconciler_config(config)
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            _push(
                remotes["alpha"],
                {"alpha-one/SKILL.md": _skill("alpha-one", body="Second.")},
            )
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(reconciler.time, "sleep", fake_sleep)

    assert reconciler.main(["loop", "--config", str(path)]) == 0
    assert sleeps == [7.5, 7.5]
    assert len(calls) == 2
    output = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("alpha: accepted ") for line in output) == 2


def test_sentinels_omitted_from_outcomes_and_output(
    env, monkeypatch, tmp_path, capfd
):
    config, _deploy_config, remotes = env
    token_sentinel = "sentinel-service-token-value"
    git_sentinel = "sentinel-git-stderr-url"
    missing = tmp_path / "remotes" / f"{git_sentinel}.git"
    config = dataclasses.replace(
        config,
        remotes={"alpha": str(missing), "beta": str(remotes["beta"])},
    )

    def fail_mint(_secret, _claims):
        raise RuntimeError(token_sentinel)

    monkeypatch.setattr(soliplex_api, "mint_service_token", fail_mint)

    first = reconciler.run_once(config)
    reconciler._report(first)

    assert first[0].problems == ["fetch failed for alpha (exit 128)"]
    assert first[1].problems == ["skill reload failed"]

    _push(
        remotes["beta"],
        {"beta-one/SKILL.md": _skill("beta-one", body="Second.")},
    )
    monkeypatch.undo()

    def fail_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            502,
            "Bad Gateway",
            hdrs=None,
            fp=io.BytesIO(f"Authorization: Bearer {token_sentinel}".encode()),
        )

    monkeypatch.setattr(soliplex_api.urllib.request, "urlopen", fail_urlopen)

    second = reconciler.run_once(config)
    reconciler._report(second)

    assert second[1].problems == ["skill reload failed (HTTP 502)"]
    captured = capfd.readouterr()
    lines = [
        *(line for outcome in (*first, *second) for line in outcome.lines()),
        captured.out,
        captured.err,
    ]
    for text in lines:
        assert token_sentinel not in text
        assert git_sentinel not in text


@pytest.mark.parametrize(
    "change",
    [
        ("remote.origin.url", "other.git"),
        ("remote.origin.fetch", "+refs/heads/*:refs/heads/*"),
        ("--add", "remote.origin.url", "other.git"),
        ("--add", "remote.origin.fetch", "+refs/*:refs/*"),
        ("core.bare", "false"),
    ],
    ids=["url", "refspec", "second-url", "second-refspec", "not-bare"],
)
def test_existing_mirror_configuration_verified(
    env, calls, monkeypatch, change
):
    config, deploy_config, remotes = env
    reconciler.run_once(config)
    alpha_rev = _git(remotes["alpha"], "rev-parse", "refs/heads/main")
    _push(
        remotes["alpha"],
        {"alpha-one/SKILL.md": _skill("alpha-one", body="New.")},
    )
    state = DeployState(deploy_config.state_root)
    mirror = deploy_config.repos["alpha"]

    _git(mirror, "config", *change)
    deployed = []
    real_deploy = deploy.deploy

    def record_deploy(config, team):
        deployed.append(team)
        return real_deploy(config, team)

    monkeypatch.setattr(deploy, "deploy", record_deploy)
    outcomes = reconciler.run_once(config)
    assert outcomes[0] == deploy.Outcome(
        "alpha", "", "rejected", None, [], ["mirror configuration differs"]
    )
    assert state.committed_revision("alpha") == alpha_rev
    assert deployed == ["beta"]
    assert len(calls) == 1


def test_remote_path_with_trailing_space_accepted_on_second_run(env, calls):
    config, _deploy_config, remotes = env
    remote = remotes["alpha"]
    spaced = remote.with_name(remote.name + " ")
    remote.rename(spaced)
    config = dataclasses.replace(config, remotes={"alpha": str(spaced)})

    first = reconciler.run_once(config)
    _push(
        spaced,
        {"alpha-one/SKILL.md": _skill("alpha-one", body="Updated.")},
    )
    second = reconciler.run_once(config)

    assert [outcome.status for outcome in first] == ["accepted"]
    assert [outcome.status for outcome in second] == ["accepted"]
    assert len(calls) == 2


def test_team_without_main_reported_and_other_team_reloads(env, calls):
    config, deploy_config, remotes = env
    _git(remotes["alpha"], "branch", "-m", "main", "topic")

    outcomes = reconciler.run_once(config)

    beta_rev = _git(remotes["beta"], "rev-parse", "refs/heads/main")
    assert outcomes == [
        deploy.Outcome(
            "alpha", "", "rejected", None, [], ["deploy failed for alpha"]
        ),
        deploy.Outcome(
            "beta", beta_rev, "accepted", Target(1, 1, "rooms-1"), [], []
        ),
    ]
    assert DeployState(deploy_config.state_root).read_target() == Target(
        1, 1, "rooms-1"
    )
    assert len(calls) == 1


@pytest.mark.parametrize("timeout", [12, 12.5])
def test_git_timeout_reported_like_fetch_failure(
    env, calls, monkeypatch, timeout
):
    config, deploy_config, _remotes = env
    config = dataclasses.replace(config, git_timeout=timeout)
    real_run = subprocess.run
    invocations = []
    failing_command = [
        "git",
        "-C",
        str(deploy_config.repos["alpha"]),
        "fetch",
        "--prune",
        "origin",
    ]
    fail_fetch = False

    def timed_run(command, *args, **kwargs):
        invocations.append((command, kwargs.get("timeout")))
        if fail_fetch and command == failing_command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return real_run(command, *args, **kwargs)

    def unchanged(_config, team):
        return deploy.Outcome(team, "", "unchanged", None, [], [])

    monkeypatch.setattr(deploy, "deploy", unchanged)
    monkeypatch.setattr(reconciler.subprocess, "run", timed_run)

    assert [o.status for o in reconciler.run_once(config)] == [
        "unchanged",
        "unchanged",
    ]
    assert len(invocations) == 2
    assert all(command[1] == "clone" for command, _ in invocations)
    assert [o.status for o in reconciler.run_once(config)] == [
        "unchanged",
        "unchanged",
    ]
    assert len(invocations) == 10
    assert invocations[5][0] == failing_command
    assert all(value == timeout for _command, value in invocations)

    fail_fetch = True
    outcomes = reconciler.run_once(config)

    assert len(invocations) == 18
    assert all(value == timeout for _command, value in invocations)
    assert outcomes == [
        deploy.Outcome(
            "alpha", "", "rejected", None, [], ["fetch failed for alpha"]
        ),
        deploy.Outcome("beta", "", "unchanged", None, [], []),
    ]
    assert calls == []
