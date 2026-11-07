"""Persist deploy generations and recover from the committed target."""

import dataclasses
import fcntl
import functools
import json
import os
import pathlib
import shutil
import stat
import tempfile

from soliplex_concierge.skills_deployer.model import TEAM_SLUG_RE
from soliplex_concierge.skills_deployer.model import Violation


@dataclasses.dataclass(frozen=True)
class Target:
    """Identify the generation and room revision committed at an epoch."""

    epoch: int
    generation: int
    room_revision: str


class RecoveryRequired(Exception):
    """Report committed or pinned state that needs operator repair."""


def _check_team(team: str) -> None:
    if not TEAM_SLUG_RE.fullmatch(team):
        msg = f"Invalid team slug: {team!r}"
        raise ValueError(msg)


def _fsync_directory(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir(path: pathlib.Path) -> None:
    if not path.exists():
        _mkdir(path.parent)
        path.mkdir(exist_ok=True)
        _fsync_directory(path.parent)


def _write(path: pathlib.Path, text: str, *, exclusive: bool = False) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".tmp-",
        delete=False,
    ) as stream:
        temporary = pathlib.Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            if exclusive:
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _write_json(
    path: pathlib.Path, value: object, *, exclusive: bool = False
) -> None:
    _write(
        path,
        json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n",
        exclusive=exclusive,
    )


def _locked(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return method(self, *args, **kwargs)

    return wrapped


def _check_room_revision(room_revision: str) -> None:
    if not isinstance(room_revision, str) or not room_revision:
        msg = "A room_revision must be a non-empty str"
        raise ValueError(msg)


def _read_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        msg = f"Invalid JSON in {path}"
        raise ValueError(msg) from error


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _read_target(path: pathlib.Path) -> Target:
    value = _read_json(path)
    if (
        not isinstance(value, dict)
        or set(value) != {"epoch", "generation", "room_revision"}
        or not _positive_int(value["epoch"])
        or not _positive_int(value["generation"])
        or not isinstance(value["room_revision"], str)
    ):
        msg = f"Invalid target in {path}"
        raise ValueError(msg)
    return Target(**value)


def _check_generation(value: object, path: pathlib.Path) -> None:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "teams",
            "baseline",
            "allow_list",
            "room_revision",
            "loader_result",
        }
        or not isinstance(value["teams"], dict)
        or not isinstance(value["baseline"], dict)
        or not isinstance(value["allow_list"], list)
        or not isinstance(value["room_revision"], str)
        or not isinstance(value["loader_result"], dict)
        or not all(isinstance(v, str) for v in value["baseline"].values())
        or not all(isinstance(v, str) for v in value["allow_list"])
    ):
        msg = f"Invalid generation in {path}"
        raise ValueError(msg)
    for team, version in value["teams"].items():
        _check_team(team)
        if (
            not isinstance(version, str)
            or pathlib.PurePosixPath(version).parts[:2] != ("versions", team)
            or len(pathlib.PurePosixPath(version).parts) != 3
            or ".." in pathlib.PurePosixPath(version).parts
        ):
            msg = f"Invalid version path for {team} in {path}: {version!r}"
            raise ValueError(msg)


def _remove_version(path: pathlib.Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IRWXU)
    for child in path.rglob("*"):
        if not child.is_symlink():
            child.chmod(child.stat().st_mode | stat.S_IRWXU)
    shutil.rmtree(path)
    _fsync_directory(path.parent)


class DeployState:
    """Store prepared deploys and atomically select the committed target."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        for directory in (
            "",
            "generations",
            "desired",
            "rejected",
            "pins",
            "versions",
        ):
            _mkdir(root / directory)

    def _generations(self) -> list[tuple[int, pathlib.Path]]:
        return sorted(
            (int(path.stem), path)
            for path in (self.root / "generations").glob("*.json")
            if path.stem.isascii() and path.stem.isdecimal()
        )

    @_locked
    def write_generation(
        self,
        teams: dict[str, str],
        *,
        baseline: dict[str, str],
        allow_list: list[str],
        room_revision: str,
        loader_result: dict,
    ) -> int:
        """Prepare an immutable generation with a fresh number."""
        sequence = self.root / ".generation-sequence"
        highest = max((n for n, _ in self._generations()), default=0)
        if sequence.exists():
            saved = _read_json(sequence)
            if type(saved) is not int or saved < 0:
                msg = f"Invalid generation sequence in {sequence}"
                raise ValueError(msg)
            highest = max(highest, saved)
        number = highest + 1
        path = self.root / "generations" / f"{number}.json"
        value = {
            "teams": teams,
            "baseline": baseline,
            "allow_list": allow_list,
            "room_revision": room_revision,
            "loader_result": loader_result,
        }
        _check_generation(value, path)
        _write_json(sequence, number)
        _write_json(path, value, exclusive=True)
        return number

    def read_generation(self, n: int) -> dict:
        """Read and validate a prepared generation."""
        if not _positive_int(n):
            msg = f"Invalid generation number: {n!r}"
            raise ValueError(msg)
        path = self.root / "generations" / f"{n}.json"
        value = _read_json(path)
        _check_generation(value, path)
        return value

    def read_target(self) -> Target | None:
        """Read the committed target if one exists."""
        try:
            return _read_target(self.root / "target")
        except FileNotFoundError:
            return None

    @_locked
    def commit(self, generation: int, room_revision: str) -> Target:
        """Commit a generation and room revision at the next epoch."""
        _check_room_revision(room_revision)
        value = self.read_generation(generation)
        for version in value["teams"].values():
            if not (self.root / version).is_dir():
                msg = f"Missing version directory {version}"
                raise ValueError(msg)
        current = self.read_target()
        target = Target(
            epoch=1 if current is None else current.epoch + 1,
            generation=generation,
            room_revision=room_revision,
        )
        _write_json(self.root / "target", dataclasses.asdict(target))
        return target

    @_locked
    def set_desired(self, team: str, revision: str) -> None:
        """Persist the revision a team requests."""
        _check_team(team)
        if not revision or "\n" in revision or "\r" in revision:
            msg = "A desired revision must be a nonempty single line"
            raise ValueError(msg)
        _write(self.root / "desired" / team, revision + "\n")

    def desired(self, team: str) -> str | None:
        """Read the revision a team requests."""
        _check_team(team)
        try:
            return (
                (self.root / "desired" / team)
                .read_text(encoding="utf-8")
                .removesuffix("\n")
            )
        except FileNotFoundError:
            return None

    @_locked
    def reject(
        self,
        team: str,
        revision: str,
        context: dict,
        violations: list[Violation],
    ) -> None:
        """Persist a team's rejection and the context it applies to."""
        _check_team(team)
        _write_json(
            self.root / "rejected" / f"{team}.json",
            {
                "revision": revision,
                "context": context,
                "violations": [dataclasses.asdict(v) for v in violations],
            },
        )

    def rejection(self, team: str) -> dict | None:
        """Read a team's stored rejection."""
        _check_team(team)
        try:
            return _read_json(self.root / "rejected" / f"{team}.json")
        except FileNotFoundError:
            return None

    def is_rejected(self, team: str, revision: str, context: dict) -> bool:
        """Check whether a rejection matches a revision and its context."""
        stored = self.rejection(team)
        return (
            stored is not None
            and stored["revision"] == revision
            and stored["context"] == context
        )

    @_locked
    def pin(self, target: Target) -> None:
        """Keep a target available during recovery."""
        _check_room_revision(target.room_revision)
        if not _positive_int(target.epoch) or not _positive_int(
            target.generation
        ):
            msg = f"Invalid pinned target: {target!r}"
            raise ValueError(msg)
        _write_json(
            self.root / "pins" / f"{target.epoch}.json",
            dataclasses.asdict(target),
        )

    @_locked
    def unpin(self, epoch: int) -> None:
        """Release a target's recovery pin."""
        (self.root / "pins" / f"{epoch}.json").unlink(missing_ok=True)
        _fsync_directory(self.root / "pins")

    def pins(self) -> list[Target]:
        """Read pinned targets in epoch order."""
        return sorted(
            (
                _read_target(path)
                for path in (self.root / "pins").glob("*.json")
                if not path.name.startswith(".tmp-")
            ),
            key=lambda target: target.epoch,
        )

    def committed_revision(self, team: str) -> str | None:
        """Read the revision deployed for a team at the committed target."""
        _check_team(team)
        target = self.read_target()
        if target is None:
            return None
        version = self.read_generation(target.generation)["teams"].get(team)
        if version is None:
            return None
        return pathlib.PurePosixPath(version).name.rsplit("-", 1)[0]

    @_locked
    def recover(self) -> list[str]:
        """Remove prepared state outside committed and pinned targets."""
        target = self.read_target()
        targets = self.pins()
        if target is not None:
            targets.append(target)
        protected = {item.generation for item in targets}
        versions = set()
        for number in sorted(protected):
            try:
                generation = self.read_generation(number)
            except (FileNotFoundError, ValueError) as error:
                msg = f"Restore generation {number} before recovering: {error}"
                raise RecoveryRequired(msg) from error
            for version in generation["teams"].values():
                if not (self.root / version).is_dir():
                    msg = f"Restore version directory {version} to recover"
                    raise RecoveryRequired(msg)
                versions.add(self.root / version)
        changes = []
        committed = 0 if target is None else target.generation
        for number, path in self._generations():
            if number > committed and number not in protected:
                path.unlink()
                _fsync_directory(path.parent)
                changes.append(f"Deleted generation {number}")
        for team in sorted((self.root / "versions").iterdir()):
            if team.is_dir():
                for version in sorted(team.iterdir()):
                    if version.is_dir() and version not in versions:
                        _remove_version(version)
                        changes.append(
                            f"Deleted version {version.relative_to(self.root)}"
                        )
        for path in sorted(self.root.rglob(".tmp-*")):
            if path.is_file():
                path.unlink()
                _fsync_directory(path.parent)
                changes.append(
                    f"Deleted temporary {path.relative_to(self.root)}"
                )
        return changes
