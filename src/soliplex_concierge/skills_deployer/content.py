"""Read git blobs, check skill file contents and write an immutable version."""

from __future__ import annotations

import collections.abc
import contextlib
import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile
import unicodedata

import yaml

from soliplex_concierge.skills_deployer import model

_DEFAULT_LIMITS = model.Limits()
_FORBIDDEN_SEPARATORS = (
    "\r",
    "\v",
    "\f",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
)


class ContentRejected(Exception):
    """Skill repository blobs failed one or more content checks."""

    def __init__(
        self, violations: collections.abc.Iterable[model.Violation]
    ) -> None:
        self.violations = sorted(
            violations, key=lambda v: (v.check, v.path, v.message)
        )
        summary = ", ".join(
            f"{v.check} {v.path}: {v.message}" for v in self.violations
        )
        super().__init__(f"content rejected ({summary})")


class _DuplicateKeyError(yaml.YAMLError):
    """A YAML mapping repeats a key."""

    def __init__(self, key: object) -> None:
        super().__init__(f"duplicate key {key!r}")


class _NotABlobError(ValueError):
    """A git object exists and is not a blob."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A YAML SafeLoader that rejects duplicate keys in mappings."""

    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise _DuplicateKeyError(key)
            seen.add(key)
        return mapping


def read_blob(repo: pathlib.Path, oid: str, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` of a git blob by object id."""
    with subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "blob", oid],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        data = proc.stdout.read(max_bytes + 1)
        if len(data) > max_bytes:
            proc.kill()
            msg = f"blob {oid} exceeds limit of {max_bytes} bytes"
            raise ValueError(msg)
        stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
        if proc.wait() != 0:
            kind = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "cat-file",
                    "-t",
                    "--end-of-options",
                    oid,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            msg = f"git cat-file blob {oid} failed: {stderr}"
            if kind.returncode == 0 and kind.stdout.strip() != "blob":
                raise _NotABlobError(msg)
            raise ValueError(msg)
        return data


def check_text(path: str, data: bytes) -> list[model.Violation]:
    """Check that ``data`` is strict UTF-8 without BOM, NUL or odd breaks."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [
            model.Violation(
                model.Check.ENCODING,
                path,
                f"invalid UTF-8 encoding: {exc}",
            )
        ]

    violations: list[model.Violation] = []
    if text.startswith("\ufeff"):
        violations.append(
            model.Violation(
                model.Check.ENCODING,
                path,
                "UTF-8 BOM is not allowed",
            )
        )
    if "\x00" in text:
        violations.append(
            model.Violation(
                model.Check.ENCODING,
                path,
                "NUL byte is not allowed",
            )
        )
    for sep in _FORBIDDEN_SEPARATORS:
        if sep in text:
            violations.append(
                model.Violation(
                    model.Check.ENCODING,
                    path,
                    f"forbidden line separator {sep!r}",
                )
            )
    return violations


def parse_skill_md(
    skill_dir: str, data: bytes
) -> tuple[model.Skill | None, list[model.Violation]]:
    """Parse and validate ``SKILL.md`` bytes for ``skill_dir``."""
    path = f"{skill_dir}/{model.SKILL_FILENAME}"
    violations = check_text(path, data)
    if violations:
        return None, violations

    lines = data.decode("utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None, [
            model.Violation(model.Check.FRONTMATTER, path, "no frontmatter")
        ]

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    else:
        return None, [
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "no closing frontmatter delimiter",
            )
        ]

    frontmatter = "\n".join(lines[1:end])
    try:
        for event in yaml.parse(frontmatter, Loader=yaml.SafeLoader):
            if (
                isinstance(event, yaml.AliasEvent)
                or getattr(event, "anchor", None) is not None
                or (
                    isinstance(event, yaml.ScalarEvent) and event.value == "<<"
                )
            ):
                return None, [
                    model.Violation(
                        model.Check.FRONTMATTER,
                        path,
                        "frontmatter must not use YAML anchors, aliases "
                        "or merge keys",
                    )
                ]
        raw = yaml.load(frontmatter, Loader=_UniqueKeyLoader)
    except Exception as exc:
        return None, [
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                f"invalid frontmatter YAML: {exc}",
            )
        ]

    metadata = {} if raw is None else raw
    if not isinstance(metadata, dict):
        return None, [
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "frontmatter must be a YAML mapping",
            )
        ]

    unknown = sorted(
        str(key) for key in metadata if key not in model.FRONTMATTER_KEYS
    )
    if unknown:
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                f"unknown frontmatter keys: {', '.join(unknown)}",
            )
        )

    name = metadata.get("name")
    if "name" not in metadata:
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "missing required frontmatter key 'name'",
            )
        )
    elif not isinstance(name, str):
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "frontmatter 'name' must be a string",
            )
        )
    elif name != skill_dir:
        violations.append(
            model.Violation(
                model.Check.NAME,
                path,
                f"skill name {name!r} does not match directory {skill_dir!r}",
            )
        )

    description = metadata.get("description")
    if "description" not in metadata:
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "missing required frontmatter key 'description'",
            )
        )
    elif not isinstance(description, str):
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "frontmatter 'description' must be a string",
            )
        )
    elif "\n" in description:
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "frontmatter 'description' must be a single line",
            )
        )
    elif not (1 <= len(description) <= model.DESCRIPTION_MAX):
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "frontmatter 'description' must be 1 to "
                f"{model.DESCRIPTION_MAX} characters",
            )
        )
    elif any(
        unicodedata.category(char) in ("Cc", "Cf") for char in description
    ):
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "frontmatter 'description' must not contain control or "
                "format characters",
            )
        )

    visibility = metadata.get("visibility", model.DEFAULT_VISIBILITY)
    if not isinstance(visibility, str) or visibility not in model.VISIBILITIES:
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "frontmatter 'visibility' must be one of "
                f"{sorted(model.VISIBILITIES)}",
            )
        )

    instructions = "\n".join(lines[end + 1 :]).strip()
    if not instructions:
        violations.append(
            model.Violation(
                model.Check.FRONTMATTER,
                path,
                "skill instructions must not be empty",
            )
        )

    if violations:
        return None, violations

    digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    return (
        model.Skill(
            name=name,
            description=description,
            visibility=visibility,
            instructions_digest=digest,
        ),
        [],
    )


def _fsync_dir(path: pathlib.Path) -> None:
    """Flush directory entries at ``path`` to disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_tree(root: pathlib.Path) -> None:
    """Make ``root`` writable and remove it recursively."""
    for dirpath, _dirnames, filenames in os.walk(root):
        parent = pathlib.Path(dirpath)
        parent.chmod(0o700)
        for filename in filenames:
            (parent / filename).chmod(0o600)
    shutil.rmtree(root, ignore_errors=True)


def _publish_dir(
    tmp_dir: pathlib.Path, versions_root: pathlib.Path, dest: pathlib.Path
) -> None:
    """Rename ``tmp_dir`` to ``dest`` and flush parent directories."""
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        msg = f"version directory already exists: {dest}"
        raise FileExistsError(msg)
    tmp_dir.chmod(0o755)
    os.rename(tmp_dir, dest)
    try:
        dest.chmod(0o555)
        _fsync_dir(dest.parent)
        _fsync_dir(versions_root)
    except BaseException:
        with contextlib.suppress(OSError):
            _remove_tree(dest)
        with contextlib.suppress(OSError):
            _fsync_dir(dest.parent)
        raise


def write_version(
    repo: pathlib.Path,
    team: str,
    revision: str,
    entries: collections.abc.Iterable[model.Entry],
    versions_root: pathlib.Path,
    *,
    seq: int,
    limits: model.Limits = _DEFAULT_LIMITS,
) -> model.Version:
    """Check all blobs in ``entries`` and write an immutable version."""
    violations: list[model.Violation] = []
    skills: list[model.Skill] = []
    files: list[tuple[str, bytes]] = []
    blob_digests: dict[str, str] = {}

    for entry in entries:
        skill_dir, rel = entry.path.split("/", 1)
        max_bytes = (
            limits.max_skill_md_bytes
            if rel == model.SKILL_FILENAME
            else limits.max_file_bytes
        )
        try:
            data = read_blob(repo, entry.oid, max_bytes)
        except _NotABlobError:
            violations.append(
                model.Violation(
                    model.Check.MODE,
                    entry.path,
                    f"object {entry.oid} is not a blob",
                )
            )
            continue
        if rel == model.SKILL_FILENAME:
            skill, file_violations = parse_skill_md(skill_dir, data)
            violations.extend(file_violations)
            if skill is not None:
                skills.append(skill)
        else:
            violations.extend(check_text(entry.path, data))
        files.append((entry.path, data))
        blob_digests[entry.path] = hashlib.sha256(data).hexdigest()

    if violations:
        raise ContentRejected(violations)

    versions_root.mkdir(parents=True, exist_ok=True)
    dest = versions_root / team / f"{revision}-{seq}"
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=".tmp-", dir=versions_root))
    try:
        for rel_path, data in files:
            target = tmp_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

        for dirpath, _dirnames, filenames in os.walk(tmp_dir, topdown=False):
            directory = pathlib.Path(dirpath)
            for filename in filenames:
                (directory / filename).chmod(0o444)
            _fsync_dir(directory)
            directory.chmod(0o555)

        _publish_dir(tmp_dir, versions_root, dest)
    except BaseException:
        _remove_tree(tmp_dir)
        raise

    return model.Version(
        team=team,
        revision=revision,
        path=dest,
        skills=tuple(sorted(skills, key=lambda s: s.name)),
        blob_digests=blob_digests,
    )
