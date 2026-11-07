"""Check durable deploy state and recovery from the target commit point."""

import dataclasses
import json
import os
import pathlib
import shutil
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from soliplex_concierge.skills_deployer.model import Check
from soliplex_concierge.skills_deployer.model import Violation
from soliplex_concierge.skills_deployer.state import CommitNotDurable
from soliplex_concierge.skills_deployer.state import DeployState
from soliplex_concierge.skills_deployer.state import RecoveryRequired
from soliplex_concierge.skills_deployer.state import Target


@pytest.fixture
def state(tmp_path):
    return DeployState(tmp_path / "nested" / "deploy")


def _generation(state, teams=None):
    return state.write_generation(
        {} if teams is None else teams,
        baseline={"shared": "digest"},
        allow_list=["search"],
        room_revision="room-a",
        loader_result={"accepted": True},
    )


def _version(state, team, name):
    path = state.root / "versions" / team / name
    (path / "resources").mkdir(parents=True)
    (path / "SKILL.md").write_text("instructions", encoding="utf-8")
    (path / "resources" / "data").write_bytes(b"content")
    return path.relative_to(state.root).as_posix()


def test_generation_and_commit_round_trip(state):
    assert state.read_target() is None
    assert state.committed_revision("alpha") is None
    version = _version(state, "alpha", "revision-1")
    first = _generation(state, {"alpha": version})
    assert first == 1
    expected = {
        "teams": {"alpha": "versions/alpha/revision-1"},
        "baseline": {"shared": "digest"},
        "allow_list": ["search"],
        "room_revision": "room-a",
        "loader_result": {"accepted": True},
    }
    assert state.read_generation(first) == expected
    assert state.read_target() is None
    original = (state.root / "generations" / "1.json").read_bytes()
    target = state.commit(first, "room-a")
    assert target == Target(1, first, "room-a")
    assert state.read_target() == target
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.epoch = 2
    assert state.committed_revision("alpha") == "revision"
    assert state.committed_revision("beta") is None
    second = _generation(state)
    assert second == 2
    assert (state.root / "generations" / "1.json").read_bytes() == original
    assert state.read_target() == target
    assert state.commit(second, "room-b") == Target(2, second, "room-b")
    restarted = DeployState(state.root)
    assert restarted.commit(first, "room-c") == Target(3, first, "room-c")
    assert not list(state.root.rglob(".tmp-*"))


def test_commit_missing_generation_preserves_target(state):
    with pytest.raises(FileNotFoundError):
        state.commit(1, "room-a")
    assert state.read_target() is None
    target = state.commit(_generation(state), "room-a")
    with pytest.raises(FileNotFoundError):
        state.commit(9, "room-b")
    assert state.read_target() == target


def test_concurrent_generation_writers_keep_distinct_files(state, monkeypatch):
    publishing = threading.Event()
    release = threading.Event()
    started = threading.Event()
    original_link = os.link
    other = DeployState(state.root)

    def link(source, destination):
        original_link(source, destination)
        if destination.name == "1.json":
            publishing.set()
            assert release.wait(5)

    def second_write():
        started.set()
        return _generation(other, {"alpha": "versions/alpha/second-1"})

    monkeypatch.setattr(os, "link", link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_generation, state)
        try:
            assert publishing.wait(5)
            path = state.root / "generations" / "1.json"
            original = path.read_bytes()
            second = executor.submit(second_write)
            assert started.wait(5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        assert first.result(timeout=5) == 1
        assert second.result(timeout=5) == 2
    assert path.read_bytes() == original
    assert state.read_generation(1)["teams"] == {}
    assert state.read_generation(2)["teams"] == {
        "alpha": "versions/alpha/second-1"
    }
    assert not list(state.root.rglob(".tmp-*"))


def test_generation_publication_preserves_existing_file(state, monkeypatch):
    original_link = os.link

    def link(source, destination):
        destination.write_bytes(b"committed generation")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(FileExistsError):
        _generation(state)
    assert (state.root / "generations" / "1.json").read_bytes() == (
        b"committed generation"
    )
    assert not list(state.root.rglob(".tmp-*"))


def test_concurrent_commit_waits_for_lock_and_uses_next_epoch(
    state, monkeypatch
):
    committing = threading.Event()
    release = threading.Event()
    started = threading.Event()
    original_replace = os.replace
    generation = _generation(state)
    other = DeployState(state.root)

    def replace(source, destination):
        if json.loads(pathlib.Path(source).read_text())["room_revision"] == (
            "room-a"
        ):
            committing.set()
            assert release.wait(5)
        original_replace(source, destination)

    def second_commit():
        started.set()
        return other.commit(generation, "room-b")

    monkeypatch.setattr(os, "replace", replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(state.commit, generation, "room-a")
        try:
            assert committing.wait(5)
            second = executor.submit(second_commit)
            assert started.wait(5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        assert first.result(timeout=5) == Target(1, generation, "room-a")
        assert second.result(timeout=5) == Target(2, generation, "room-b")
    assert state.read_target() == Target(2, generation, "room-b")


def test_commit_missing_version_preserves_target(state):
    target = state.commit(_generation(state), "room-a")
    version = _version(state, "alpha", "revision-1")
    generation = _generation(state, {"alpha": version})
    shutil.rmtree(state.root / version)
    original = (state.root / "target").read_bytes()
    with pytest.raises(ValueError, match=version):
        state.commit(generation, "room-b")
    assert state.read_target() == target
    assert (state.root / "target").read_bytes() == original


@pytest.mark.parametrize("room_revision", [None, 1, False, ""])
def test_commit_rejects_invalid_room_revision(state, room_revision):
    generation = _generation(state)
    target = state.commit(generation, "room-a")
    with pytest.raises(ValueError, match="room_revision"):
        state.commit(generation, room_revision)
    assert state.read_target() == target


def test_replace_failure_preserves_target(state, monkeypatch):
    first = _generation(state)
    target = state.commit(first, "room-a")
    prepared = _generation(state)
    old_bytes = (state.root / "target").read_bytes()

    def fail_replace(source, destination):
        assert pathlib.Path(source).parent == state.root
        assert pathlib.Path(source).name.startswith(".tmp-")
        assert json.loads(pathlib.Path(source).read_text()) == {
            "epoch": 2,
            "generation": prepared,
            "room_revision": "room-b",
        }
        assert destination == state.root / "target"
        msg = "simulated crash before replace"
        raise OSError(msg)

    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated crash"):
            state.commit(prepared, "room-b")
    assert (state.root / "target").read_bytes() == old_bytes
    assert state.read_target() == target
    assert not list(state.root.rglob(".tmp-*"))
    assert state.recover() == [f"Deleted generation {prepared}"]


def test_commit_directory_fsync_failure_reports_committed_target(
    state, monkeypatch
):
    first = state.commit(_generation(state), "room-a")
    prepared = _generation(state)
    expected = Target(first.epoch + 1, prepared, "room-b")
    original_fsync = os.fsync
    error = OSError("simulated directory fsync failure")

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            assert state.read_target() == expected
            raise error
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(CommitNotDurable) as raised:
        state.commit(prepared, "room-b")
    assert isinstance(raised.value, OSError)
    assert raised.value.__cause__ is error
    assert raised.value.target == expected
    assert state.read_target() == raised.value.target


def test_writes_flush_and_fsync_before_replace(state, monkeypatch):
    events = []
    original_fsync = os.fsync
    original_replace = os.replace

    def fsync(fd):
        events.append(
            "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        )
        original_fsync(fd)

    def replace(source, destination):
        assert pathlib.Path(source).read_bytes() == b"r\xc3\xa9vision\n"
        assert pathlib.Path(source).parent == destination.parent
        events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    state.set_desired("alpha", "révision")
    assert events == ["file", "replace", "directory"]
    assert state.desired("alpha") == "révision"
    assert not list(state.root.rglob(".tmp-*"))


@pytest.mark.parametrize(
    "content", [b"{", b"\xff", b"[]", b"{}", b'{"epoch": 1}']
)
def test_corrupt_target_json(state, content):
    (state.root / "target").write_bytes(content)
    with pytest.raises(ValueError, match="target"):
        state.read_target()
    with pytest.raises(ValueError, match="target"):
        state.commit(_generation(state), "room-a")


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", 0),
        ("epoch", True),
        ("epoch", "1"),
        ("generation", -1),
        ("generation", False),
        ("room_revision", None),
    ],
)
def test_corrupt_target_fields(state, field, value):
    data = {"epoch": 1, "generation": 1, "room_revision": "room-a"}
    data[field] = value
    (state.root / "target").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid target"):
        state.read_target()


@pytest.mark.parametrize("content", [b"{", b"\xff", b"[]", b"{}"])
def test_corrupt_generation_json(state, content):
    (state.root / "generations" / "1.json").write_bytes(content)
    with pytest.raises(ValueError, match="1.json"):
        state.read_generation(1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("teams", []),
        ("baseline", []),
        ("allow_list", {}),
        ("room_revision", 1),
        ("loader_result", []),
        ("baseline", {"shared": 1}),
        ("allow_list", [1]),
        ("teams", {"bad-team": "versions/bad-team/rev-1"}),
        ("teams", {"alpha": None}),
        ("teams", {"alpha": "/versions/alpha/rev-1"}),
        ("teams", {"alpha": "versions/beta/rev-1"}),
        ("teams", {"alpha": "versions/alpha"}),
        ("teams", {"alpha": "versions/alpha/rev-1/extra"}),
        ("teams", {"alpha": "versions/alpha/.."}),
    ],
)
def test_corrupt_generation_fields(state, field, value):
    number = _generation(state)
    data = state.read_generation(number)
    data[field] = value
    (state.root / "generations" / "1.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid"):
        state.read_generation(number)


@pytest.mark.parametrize("number", [0, -1, True, "../target"])
def test_invalid_generation_number(state, number):
    with pytest.raises(ValueError, match="Invalid generation number"):
        state.read_generation(number)


def test_generation_numbering_scans_existing_numbers(state):
    (state.root / "generations" / "20.json").write_text("{}")
    (state.root / "generations" / "other.json").write_text("{}")
    (state.root / "generations" / "١.json").write_text("{}")
    assert _generation(state) == 21
    (state.root / "generations" / "21.json").unlink()
    assert _generation(DeployState(state.root)) == 22


@pytest.mark.parametrize("value", ["bad", -1, True])
def test_corrupt_sequence(state, value):
    (state.root / ".generation-sequence").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Invalid generation sequence"):
        _generation(state)


def test_desired_and_rejections(state):
    assert state.desired("alpha") is None
    state.set_desired("alpha", "rev-a")
    assert (state.root / "desired" / "alpha").read_bytes() == b"rev-a\n"
    assert state.desired("alpha") == "rev-a"
    state.set_desired("alpha", "rev-b")
    assert state.desired("alpha") == "rev-b"
    assert state.rejection("alpha") is None
    assert not state.is_rejected("alpha", "rev-b", {})
    context = {"baseline": "first", "allow_list": ["search"]}
    violations = [Violation(Check.NAME, "SKILL.md", "Choose a valid name")]
    state.reject("alpha", "rev-b", context, violations)
    assert state.rejection("alpha") == {
        "revision": "rev-b",
        "context": context,
        "violations": [
            {
                "check": "C2",
                "path": "SKILL.md",
                "message": "Choose a valid name",
            },
        ],
    }
    assert state.is_rejected("alpha", "rev-b", dict(reversed(context.items())))
    assert not state.is_rejected("alpha", "rev-a", context)
    assert not state.is_rejected("alpha", "rev-b", {**context, "new": 1})
    assert not state.is_rejected("alpha", "rev-b", {"baseline": "second"})
    path = state.root / "rejected" / "alpha.json"
    assert (
        path.read_text()
        == json.dumps(
            state.rejection("alpha"), sort_keys=True, ensure_ascii=False
        )
        + "\n"
    )
    state.reject("alpha", "rev-c", {}, [])
    assert state.is_rejected("alpha", "rev-c", {})
    assert state.rejection("alpha")["violations"] == []


@pytest.mark.parametrize("revision", ["", "first\nsecond", "first\rsecond"])
def test_desired_rejects_multiple_lines(state, revision):
    with pytest.raises(ValueError, match="single line"):
        state.set_desired("alpha", revision)


@pytest.mark.parametrize("team", ["", "Alpha", "two-teams", "../bad", "a\n"])
@pytest.mark.parametrize(
    "method,args",
    [
        ("set_desired", ("rev",)),
        ("desired", ()),
        ("reject", ("rev", {}, [])),
        ("rejection", ()),
        ("is_rejected", ("rev", {})),
        ("committed_revision", ()),
    ],
)
def test_team_methods_reject_invalid_slugs(state, team, method, args):
    with pytest.raises(ValueError, match="Invalid team slug"):
        getattr(state, method)(team, *args)


def test_generation_rejects_invalid_team_before_writing(state):
    with pytest.raises(ValueError, match="Invalid team slug"):
        _generation(state, {"../bad": "versions/bad/rev-1"})
    assert list((state.root / "generations").iterdir()) == []
    assert not (state.root / ".generation-sequence").exists()


def test_pins_round_trip(state):
    assert state.pins() == []
    first = Target(2, 1, "room-a")
    second = Target(10, 3, "room-b")
    state.pin(second)
    state.pin(first)
    assert state.read_target() is None
    assert state.pins() == [first, second]
    state.pin(Target(2, 1, "room-c"))
    assert state.pins() == [Target(2, 1, "room-c"), second]
    state.unpin(first.epoch)
    state.unpin(first.epoch)
    assert state.pins() == [second]
    assert not list(state.root.rglob(".tmp-*"))


@pytest.mark.parametrize("room_revision", [None, 1, False, ""])
def test_pin_rejects_invalid_room_revision(state, room_revision):
    with pytest.raises(ValueError, match="room_revision"):
        state.pin(Target(1, 1, room_revision))
    assert state.pins() == []


@pytest.mark.parametrize("field", ["epoch", "generation"])
@pytest.mark.parametrize("value", [0, -1, True, False, "1", 1.0, None])
def test_pin_rejects_invalid_numbers(state, field, value):
    target = dataclasses.replace(Target(1, 1, "room-a"), **{field: value})
    with pytest.raises(ValueError, match=f"Invalid pinned target:.*{field}="):
        state.pin(target)
    assert state.pins() == []


def test_recovery_removes_temporary_pin_json(state):
    (state.root / "pins" / ".tmp-interrupted.json").write_text("{")
    assert state.pins() == []
    assert state.recover() == ["Deleted temporary pins/.tmp-interrupted.json"]
    assert not list(state.root.rglob(".tmp-*"))


def test_recover_prepared_and_unreferenced_state(state):
    old = _version(state, "alpha", "old-1")
    old_generation = _generation(state, {"alpha": old})
    live = _version(state, "alpha", "live-2")
    committed = _generation(state, {"alpha": live})
    target = state.commit(committed, "room-live")
    pinned = _version(state, "beta", "pinned-1")
    pinned_generation = _generation(state, {"beta": pinned})
    state.pin(Target(3, pinned_generation, "room-pinned"))
    unused = _version(state, "alpha", "unused-3")
    prepared = _generation(state, {"alpha": unused})
    orphan = _version(state, "gamma", "orphan-1")
    (state.root / unused / "resources" / "link").symlink_to("data")
    for name in (unused, orphan, old):
        path = state.root / name
        for child in path.rglob("*"):
            if not child.is_symlink():
                child.chmod(0o555 if child.is_dir() else 0o444)
        path.chmod(0o555)
    for parent in (
        state.root,
        state.root / "generations",
        state.root / live / "resources",
    ):
        (parent / ".tmp-leftover").write_text("partial")
    (state.root / ".tmp-directory").mkdir()
    (state.root / "versions" / "file").touch()
    (state.root / "versions" / "alpha" / "file").touch()
    changes = state.recover()
    assert f"Deleted generation {prepared}" in changes
    for name in (unused, orphan, old):
        assert f"Deleted version {name}" in changes
        assert not (state.root / name).exists()
    assert len([change for change in changes if "temporary" in change]) == 3
    assert len(changes) == 7
    assert (state.root / live).is_dir()
    assert (state.root / pinned).is_dir()
    assert (state.root / "generations" / f"{old_generation}.json").exists()
    assert state.read_generation(pinned_generation)["teams"] == {
        "beta": pinned
    }
    assert state.read_target() == target
    assert (state.root / "versions" / "gamma").is_dir()
    assert (state.root / ".tmp-directory").is_dir()
    assert state.recover() == []
    assert _generation(DeployState(state.root)) == prepared + 1
    state.unpin(3)
    changes = state.recover()
    assert f"Deleted generation {pinned_generation}" in changes
    assert f"Deleted version {pinned}" in changes


def test_recover_without_target_keeps_only_pins(state):
    pinned = _version(state, "alpha", "pinned-1")
    first = _generation(state, {"alpha": pinned})
    state.pin(Target(1, first, "room-a"))
    orphan = _version(state, "beta", "orphan-1")
    second = _generation(state, {"beta": orphan})
    assert state.recover() == [
        f"Deleted generation {second}",
        f"Deleted version {orphan}",
    ]
    state.unpin(1)
    assert state.recover() == [
        f"Deleted generation {first}",
        f"Deleted version {pinned}",
    ]
    assert state.read_target() is None
    assert _generation(DeployState(state.root)) == 3


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "missing-generation",
        "corrupt-generation",
        "missing-version",
        "version-is-file",
    ],
)
def test_recover_requires_intact_references(state, pinned, damage):
    version = _version(state, "alpha", "revision-1")
    number = _generation(state, {"alpha": version})
    if pinned:
        state.pin(Target(1, number, "room-a"))
    else:
        state.commit(number, "room-a")
    path = state.root / "generations" / f"{number}.json"
    if damage == "missing-generation":
        path.unlink()
        message = rf"Restore generation {number}.*{number}\.json"
    elif damage == "corrupt-generation":
        path.write_text("{}")
        message = rf"Invalid generation in .*{number}\.json"
    elif damage == "missing-version":
        shutil.rmtree(state.root / version)
        message = f"Restore version directory {version}"
    else:
        shutil.rmtree(state.root / version)
        (state.root / version).touch()
        message = f"Restore version directory {version}"
    prepared = _generation(state)
    orphan = _version(state, "beta", "orphan-1")
    with pytest.raises(RecoveryRequired, match=message):
        state.recover()
    assert (state.root / "generations" / f"{prepared}.json").exists()
    assert (state.root / orphan).is_dir()


def test_committed_revision_uses_last_sequence_suffix(state):
    version = _version(state, "alpha", "topic-with-hyphens-42")
    number = _generation(state, {"alpha": version})
    state.commit(number, "room-a")
    _generation(state, {"alpha": "versions/alpha/uncommitted-1"})
    assert state.committed_revision("alpha") == "topic-with-hyphens"
