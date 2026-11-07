"""Check durable deploy state and recovery from the target commit point."""

import dataclasses
import fcntl
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


def test_current_links_follow_committed_generation(state):
    alpha = _version(state, "alpha", "first-1")
    beta = _version(state, "beta", "first-1")
    state.commit(_generation(state, {"alpha": alpha, "beta": beta}), "r1")
    current = state.root / "current"
    for team, version in {"alpha": alpha, "beta": beta}.items():
        assert (current / team).is_symlink()
        assert (current / team).readlink() == pathlib.Path("..") / version
        assert (current / team).resolve(strict=True) == state.root / version
    assert state.update_current_links() == []
    beta_inode = (current / "beta").lstat().st_ino
    alpha = _version(state, "alpha", "second-1")
    prepared = _generation(state, {"alpha": alpha, "beta": beta})
    assert state.update_current_links() == []
    state.commit(prepared, "r2")
    assert (current / "alpha").is_symlink()
    assert (current / "alpha").readlink() == pathlib.Path("..") / alpha
    assert (current / "alpha").resolve(strict=True) == state.root / alpha
    assert (current / "beta").lstat().st_ino == beta_inode
    (current / "notes").touch()
    state.commit(_generation(state, {"beta": beta}), "r3")
    assert not (current / "alpha").is_symlink()
    assert (current / "beta").lstat().st_ino == beta_inode
    assert (current / "notes").is_file()
    state.commit(_generation(state), "r4")
    assert not (current / "beta").is_symlink()
    (current / "beta").symlink_to(pathlib.Path("..") / beta)
    assert state.update_current_links() == ["Deleted link current/beta"]
    assert state.update_current_links() == []


def test_current_links_without_target_do_nothing(state):
    assert state.update_current_links() == []
    current = state.root / "current"
    assert not current.exists()
    version = _version(state, "alpha", "prepared-1")
    number = _generation(state, {"alpha": version})
    state.pin(Target(1, number, "r1"))
    current.mkdir()
    (current / "alpha").symlink_to("../versions/alpha/old-1")
    assert state.update_current_links() == []
    assert state.recover() == []
    assert (current / "alpha").readlink() == pathlib.Path(
        "../versions/alpha/old-1"
    )


@pytest.mark.parametrize("damage", ["deleted", "wrong", "dangling"])
def test_recover_repairs_current_link(state, damage):
    alpha = _version(state, "alpha", "first-1")
    beta = _version(state, "beta", "first-1")
    state.commit(_generation(state, {"alpha": alpha, "beta": beta}), "r1")
    link = state.root / "current" / "alpha"
    link.unlink()
    if damage == "wrong":
        link.symlink_to(pathlib.Path("..") / beta)
    elif damage == "dangling":
        link.symlink_to("../versions/alpha/missing-1")
    assert state.recover() == [f"Linked current/alpha to {alpha}"]
    assert link.resolve(strict=True) == state.root / alpha
    assert state.recover() == []


def test_recover_updates_links_before_removing_versions(state, monkeypatch):
    from soliplex_concierge.skills_deployer import state as state_module

    alpha1 = _version(state, "alpha", "first-1")
    beta1 = _version(state, "beta", "first-1")
    state.commit(_generation(state, {"alpha": alpha1, "beta": beta1}), "r1")
    alpha2 = _version(state, "alpha", "second-1")
    beta2 = _version(state, "beta", "second-1")
    state.commit(_generation(state, {"alpha": alpha2, "beta": beta2}), "r2")
    state.acknowledge(2)
    current = state.root / "current"
    for team, old in {"alpha": alpha1, "beta": beta1}.items():
        (current / team).unlink()
        (current / team).symlink_to(pathlib.Path("..") / old)
    committed = {"alpha": state.root / alpha2, "beta": state.root / beta2}
    removed = []
    original_remove = state_module._remove_version

    def remove_version(path):
        for team, expected in committed.items():
            assert (current / team).resolve(strict=True) == expected
        removed.append(path)
        original_remove(path)

    monkeypatch.setattr(state_module, "_remove_version", remove_version)
    changes = state.recover()
    assert removed == [state.root / alpha1, state.root / beta1]
    assert changes == [
        f"Linked current/alpha to {alpha2}",
        f"Linked current/beta to {beta2}",
        f"Deleted version {alpha1}",
        f"Deleted version {beta1}",
    ]


def test_recover_syncs_correct_links_before_removing_versions(
    state, monkeypatch
):
    from soliplex_concierge.skills_deployer import state as state_module

    live = _version(state, "alpha", "live-1")
    state.commit(_generation(state, {"alpha": live}), "r1")
    orphan = _version(state, "alpha", "orphan-1")
    assert state.update_current_links() == []
    events = []
    original_fsync = state_module._fsync_directory
    original_remove = state_module._remove_version

    def fsync_directory(path):
        events.append(("fsync", path))
        original_fsync(path)

    def remove_version(path):
        events.append(("remove", path))
        original_remove(path)

    monkeypatch.setattr(state_module, "_fsync_directory", fsync_directory)
    monkeypatch.setattr(state_module, "_remove_version", remove_version)
    assert state.recover() == [f"Deleted version {orphan}"]
    assert events[:3] == [
        ("fsync", state.root / "current"),
        ("fsync", state.root),
        ("remove", state.root / orphan),
    ]


@pytest.mark.parametrize("directory", ["current", ""])
def test_recover_fsync_failure_preserves_versions(
    state, monkeypatch, directory
):
    from soliplex_concierge.skills_deployer import state as state_module

    live = _version(state, "alpha", "live-1")
    state.commit(_generation(state, {"alpha": live}), "r1")
    orphan = _version(state, "alpha", "orphan-1")
    assert state.update_current_links() == []
    original_fsync = state_module._fsync_directory
    error = OSError("simulated recovery fsync failure")

    def fsync_directory(path):
        if path == state.root / directory:
            raise error
        original_fsync(path)

    monkeypatch.setattr(state_module, "_fsync_directory", fsync_directory)
    with pytest.raises(
        OSError, match="simulated recovery fsync failure"
    ) as raised:
        state.recover()
    assert raised.value is error
    assert (state.root / orphan).is_dir()


def test_recover_without_target_skips_current_and_root_fsync(
    state, monkeypatch
):
    from soliplex_concierge.skills_deployer import state as state_module

    orphan = _version(state, "alpha", "orphan-1")
    synced = []
    original_fsync = state_module._fsync_directory

    def fsync_directory(path):
        synced.append(path)
        original_fsync(path)

    monkeypatch.setattr(state_module, "_fsync_directory", fsync_directory)
    assert state.recover() == [f"Deleted version {orphan}"]
    assert synced == [state.root / "versions" / "alpha"]


def test_current_directory_fsynced_on_link_create_and_remove(
    state, monkeypatch
):
    from soliplex_concierge.skills_deployer import state as state_module

    alpha = _version(state, "alpha", "first-1")
    first = _generation(state, {"alpha": alpha})
    empty = _generation(state)
    current = state.root / "current"
    synced = []
    original_fsync = state_module._fsync_directory

    def fsync_directory(path):
        if path == current:
            link = current / "alpha"
            synced.append(link.readlink() if link.is_symlink() else None)
        original_fsync(path)

    monkeypatch.setattr(state_module, "_fsync_directory", fsync_directory)
    state.commit(first, "r1")
    assert synced == [pathlib.Path("..") / alpha]
    synced.clear()
    state.commit(empty, "r2")
    assert synced == [None]


def test_current_link_replacement_is_atomic_and_synced(state, monkeypatch):
    first = _version(state, "alpha", "first-1")
    state.commit(_generation(state, {"alpha": first}), "r1")
    second = _version(state, "alpha", "second-1")
    state.commit(_generation(state, {"alpha": second}), "r2")
    current = state.root / "current"
    link = current / "alpha"
    link.unlink()
    link.symlink_to(pathlib.Path("..") / first)
    events = []
    original_replace = os.replace
    original_fsync = os.fsync

    def replace(source, destination):
        assert source.parent == current
        assert source.name.startswith(".tmp-")
        assert source.is_symlink()
        assert destination == link
        assert link.exists()
        assert link.resolve() == state.root / first
        with (state.root / ".lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_replace(source, destination)
        assert link.exists()
        assert link.resolve() == state.root / second
        events.append("replace")

    def fsync(fd):
        assert os.fstat(fd).st_ino == current.stat().st_ino
        assert link.exists()
        events.append("fsync")
        original_fsync(fd)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "fsync", fsync)
    assert state.update_current_links() == [
        f"Linked current/alpha to {second}"
    ]
    assert events == ["replace", "fsync"]
    assert list(current.iterdir()) == [link]


def test_current_link_replace_failure_preserves_link(state, monkeypatch):
    first = _version(state, "alpha", "first-1")
    state.commit(_generation(state, {"alpha": first}), "r1")
    second = _version(state, "alpha", "second-1")
    state.commit(_generation(state, {"alpha": second}), "r2")
    current = state.root / "current"
    link = current / "alpha"
    link.unlink()
    link.symlink_to(pathlib.Path("..") / first)

    def replace(source, destination):
        assert source.is_symlink()
        assert destination == link
        msg = "simulated link replace failure"
        raise OSError(msg)

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError, match="simulated link replace failure"):
        state.update_current_links()
    assert link.resolve() == state.root / first
    assert list(current.iterdir()) == [link]


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
        if destination == state.root / "target" and (
            json.loads(pathlib.Path(source).read_text())["room_revision"]
            == "room-a"
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
    first_version = _version(state, "alpha", "first-1")
    first = state.commit(
        _generation(state, {"alpha": first_version}), "room-a"
    )
    second_version = _version(state, "alpha", "second-1")
    prepared = _generation(state, {"alpha": second_version})
    expected = Target(first.epoch + 1, prepared, "room-b")
    original_fsync = os.fsync
    error = OSError("simulated directory fsync failure")

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            assert state.read_target() == expected
            raise error
        original_fsync(fd)

    with monkeypatch.context() as patched:
        patched.setattr(os, "fsync", fsync)
        with pytest.raises(CommitNotDurable) as raised:
            state.commit(prepared, "room-b")
    assert isinstance(raised.value, OSError)
    assert raised.value.__cause__ is error
    assert raised.value.target == expected
    assert state.read_target() == raised.value.target
    assert (state.root / "current" / "alpha").resolve(strict=True) == (
        state.root / first_version
    )
    assert state.recover() == [
        f"Linked current/alpha to {second_version}",
    ]
    assert (state.root / "current" / "alpha").resolve(strict=True) == (
        state.root / second_version
    )


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
    with pytest.raises(ValueError, match="Invalid generation number"):
        state.acknowledge(number)


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
    context = {"baseline": "fírst", "allow_list": ["search"]}
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
            state.rejection("alpha"), sort_keys=True, ensure_ascii=True
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
    state.acknowledge(committed)
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
    (state.root / "current" / "alpha").unlink()
    (state.root / ".tmp-directory").mkdir()
    (state.root / "versions" / "file").touch()
    (state.root / "versions" / "alpha" / "file").touch()
    changes = state.recover()
    assert f"Deleted generation {prepared}" in changes
    for name in (unused, orphan, old):
        assert f"Deleted version {name}" in changes
        assert not (state.root / name).exists()
    assert len([change for change in changes if "temporary" in change]) == 3
    assert f"Linked current/alpha to {live}" in changes
    assert len(changes) == 8
    assert (state.root / "current" / "alpha").resolve(strict=True) == (
        state.root / live
    )
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


def test_acknowledged_round_trip_and_atomic_write(state, monkeypatch):
    assert state.acknowledged() is None
    for _ in range(3):
        state.commit(_generation(state), "room-a")
    events = []
    original_replace = os.replace

    def replace(source, destination):
        assert destination == state.root / "acknowledged"
        assert pathlib.Path(source).parent == state.root
        assert pathlib.Path(source).name.startswith(".tmp-")
        with (state.root / ".lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events.append(pathlib.Path(source).read_text(encoding="utf-8"))
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace)
    state.acknowledge(1)
    assert state.acknowledged() == 1
    state.acknowledge(3)
    assert state.acknowledged() == 3
    state.acknowledge(2)
    state.acknowledge(3)
    assert state.acknowledged() == 3
    assert events == ['{"generation": 1}\n', '{"generation": 3}\n']
    assert not list(state.root.rglob(".tmp-*"))


@pytest.mark.parametrize("acknowledged", [False, True])
def test_recover_protects_generation_versions_until_acknowledged(
    state, acknowledged
):
    first = _version(state, "alpha", "first-1")
    first_gen = _generation(state, {"alpha": first})
    state.commit(first_gen, "room-a")
    if acknowledged:
        state.acknowledge(first_gen)
    second = _version(state, "alpha", "second-1")
    second_gen = _generation(state, {"alpha": second})
    state.commit(second_gen, "room-b")
    assert state.recover() == []
    assert (state.root / first).is_dir()
    assert (state.root / second).is_dir()
    state.acknowledge(second_gen)
    assert state.recover() == [f"Deleted version {first}"]
    assert not (state.root / first).exists()
    assert (state.root / second).is_dir()


@pytest.mark.parametrize(
    "content",
    [
        None,
        b"{",
        b"\xff",
        b"[]",
        b"{}",
        b'{"generation": 0}',
        b'{"generation": -1}',
        b'{"generation": true}',
        b'{"generation": "1"}',
        b'{"generation": 1.0}',
        b'{"generation": null}',
        b'{"generation": 3}',
        b'{"generation": ' + b"9" * 300 + b"}",
        b'{"generation": 1, "extra": 2}',
    ],
)
def test_missing_or_corrupt_acknowledgement_protects_versions(state, content):
    first = _version(state, "alpha", "first-1")
    first_gen = _generation(state, {"alpha": first})
    state.commit(first_gen, "room-a")
    second = _version(state, "alpha", "second-1")
    second_gen = _generation(state, {"alpha": second})
    state.commit(second_gen, "room-b")
    path = state.root / "acknowledged"
    if content is not None:
        path.write_bytes(content)
    assert state.acknowledged() is None
    assert state.recover() == []
    assert (state.root / first).is_dir()
    assert (state.root / second).is_dir()
    state.acknowledge(second_gen)
    assert state.recover() == [f"Deleted version {first}"]
    assert not (state.root / first).exists()
    assert (state.root / second).is_dir()


@pytest.mark.parametrize("damage", ["directory", "name-too-long"])
def test_unreadable_acknowledged_file_is_absent(state, damage):
    state.commit(_generation(state), "room-a")
    path = state.root / "acknowledged"
    if damage == "directory":
        path.mkdir()
    else:
        path.symlink_to("x" * 300)
    assert state.acknowledged() is None
    assert state.recover() == []


def test_acknowledged_without_committed_target_is_absent(state):
    (state.root / "acknowledged").write_text('{"generation": 1}')
    assert state.acknowledged() is None


@pytest.mark.parametrize("damage", ["missing", "corrupt-json", "corrupt-data"])
@pytest.mark.parametrize("damaged_number", [1, 2])
@pytest.mark.parametrize("acknowledged", [False, True])
def test_missing_or_invalid_acknowledged_generation_protects_nothing(
    state, damage, damaged_number, acknowledged
):
    first = _version(state, "alpha", "first-1")
    first_gen = _generation(state, {"alpha": first})
    state.commit(first_gen, "room-a")
    if acknowledged:
        state.acknowledge(first_gen)
    second = _version(state, "alpha", "second-1")
    second_gen = _generation(state, {"alpha": second})
    state.commit(second_gen, "room-b")
    third = _version(state, "alpha", "third-1")
    state.commit(_generation(state, {"alpha": third}), "room-c")
    gen_path = state.root / "generations" / f"{damaged_number}.json"
    if damage == "missing":
        gen_path.unlink()
    elif damage == "corrupt-json":
        gen_path.write_text("{", encoding="utf-8")
    else:
        gen_path.write_text("{}", encoding="utf-8")
    removed, kept = (first, second) if damaged_number == 1 else (second, first)
    assert state.recover() == [f"Deleted version {removed}"]
    assert state.recover() == []
    assert not (state.root / removed).exists()
    assert (state.root / kept).is_dir()
    assert (state.root / third).is_dir()
