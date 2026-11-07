"""Unit tests for ``soliplex_concierge.skills_deployer.content``."""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess

import pytest

from soliplex_concierge.skills_deployer import content
from soliplex_concierge.skills_deployer import model


def _entries(repo: pathlib.Path, rev: str = "HEAD") -> list[model.Entry]:
    """Read tree entries at ``rev`` using ``git ls-tree``."""
    out = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            "-l",
            rev,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    entries = []
    for record in out.rstrip("\x00").split("\x00"):
        header, path = record.split("\t", 1)
        mode, _kind, oid, size = header.split()
        entries.append(
            model.Entry(mode=mode, oid=oid, size=int(size), path=path)
        )
    return entries


def _make_repo(
    tmp_path: pathlib.Path, files: dict[str, bytes]
) -> tuple[pathlib.Path, str, list[model.Entry]]:
    """Build a git repository in ``tmp_path`` with ``files`` committed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "-C", str(repo), "init", "--quiet", "--initial-branch=main"],
        check=True,
    )
    for rel_path, data in files.items():
        target = repo / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "core.autocrlf=false", "add", "--all"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        check=True,
    )
    rev = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return repo, rev, _entries(repo, rev)


def _skill_bytes(
    name: str = "alpha",
    description: str = "Alpha skill.",
    body: str = "Follow these instructions.\n",
    extra: str = "",
) -> bytes:
    """Build valid ``SKILL.md`` bytes from frontmatter fields and body."""
    text = (
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{body}"
    )
    return text.encode("utf-8")


# ---- read_blob ----


def test_read_blob_returns_bytes(tmp_path: pathlib.Path) -> None:
    payload = b"hello world\n"
    repo, _rev, entries = _make_repo(
        tmp_path, {"alpha/resources/note.txt": payload}
    )

    result = content.read_blob(repo, entries[0].oid, max_bytes=100)

    assert result == payload


def test_read_blob_over_limit_raises(tmp_path: pathlib.Path) -> None:
    payload = b"x" * 50
    repo, _rev, entries = _make_repo(
        tmp_path, {"alpha/resources/note.txt": payload}
    )

    with pytest.raises(ValueError, match="exceeds limit"):
        content.read_blob(repo, entries[0].oid, max_bytes=10)


def test_read_blob_bad_oid_raises(tmp_path: pathlib.Path) -> None:
    repo, _rev, _entries_list = _make_repo(
        tmp_path, {"alpha/resources/note.txt": b"ok\n"}
    )

    with pytest.raises(ValueError, match="git cat-file blob"):
        content.read_blob(repo, "0" * 40, max_bytes=100)


# ---- check_text (C7) ----


def test_check_text_accepts_clean_utf8() -> None:
    violations = content.check_text("alpha/resources/a.txt", b"line 1\nline 2")

    assert violations == []


@pytest.mark.parametrize(
    "data, expected_msg",
    [
        (b"\xff\xfeinvalid", "invalid UTF-8"),
        (b"has\x00nul\n", "NUL byte"),
        (b"\xef\xbb\xbfbom start\n", "UTF-8 BOM"),
        (b"line1\r\nline2\n", "'\\r'"),
        (b"line1\rline2\n", "'\\r'"),
        (b"line1\vline2\n", "'\\x0b'"),
        (b"line1\fline2\n", "'\\x0c'"),
        (b"line1\x1cline2\n", "'\\x1c'"),
        (b"line1\x1dline2\n", "'\\x1d'"),
        (b"line1\x1eline2\n", "'\\x1e'"),
        ("line1\x85line2\n".encode(), "'\\x85'"),
        ("line1\u2028line2\n".encode(), "'\\u2028'"),
        ("line1\u2029line2\n".encode(), "'\\u2029'"),
    ],
)
def test_check_text_rejects_c7_cases(data: bytes, expected_msg: str) -> None:
    violations = content.check_text("alpha/resources/bad.txt", data)

    assert len(violations) == 1
    assert violations[0].check is model.Check.ENCODING
    assert violations[0].path == "alpha/resources/bad.txt"
    assert expected_msg in violations[0].message


def test_parse_skill_md_rejects_u2028_before_closing_delimiter() -> None:
    raw = ("---\nname: alpha\ndescription: ok\u2028---\nBody text.\n").encode()

    skill, violations = content.parse_skill_md("alpha", raw)

    assert skill is None
    assert len(violations) == 1
    assert violations[0].check is model.Check.ENCODING
    assert violations[0].path == "alpha/SKILL.md"


# ---- parse_skill_md (C4 & C2) ----


def test_parse_skill_md_valid_default_visibility() -> None:
    data = _skill_bytes(
        name="alpha",
        description="Helpful skill.",
        body="  Step one.\nStep two.  \n",
    )

    skill, violations = content.parse_skill_md("alpha", data)

    expected_digest = hashlib.sha256(b"Step one.\nStep two.").hexdigest()
    assert violations == []
    assert skill == model.Skill(
        name="alpha",
        description="Helpful skill.",
        visibility="private",
        instructions_digest=expected_digest,
    )


def test_parse_skill_md_valid_public_visibility() -> None:
    data = _skill_bytes(
        name="alpha",
        description="Public skill.",
        extra="visibility: public\n",
    )

    skill, violations = content.parse_skill_md("alpha", data)

    assert violations == []
    assert skill is not None
    assert skill.visibility == "public"


@pytest.mark.parametrize(
    "raw, expected_msg",
    [
        (b"", "no frontmatter"),
        (b"# Just a heading\nBody\n", "no frontmatter"),
        (
            b"---\nname: alpha\ndescription: ok\nBody without closing\n",
            "no closing frontmatter delimiter",
        ),
        (
            b"---\nname: [unclosed\n---\nBody\n",
            "invalid frontmatter YAML",
        ),
        (
            b"---\nname: alpha\ndescription: first\ndescription: second\n---\n"
            b"Body\n",
            "duplicate key 'description'",
        ),
        (
            b"---\n- item1\n- item2\n---\nBody\n",
            "frontmatter must be a YAML mapping",
        ),
        (
            b"---\nname: alpha\ndescription: ok\nauthor: alice\n---\nBody\n",
            "unknown frontmatter keys: author",
        ),
        (
            b"---\ndescription: ok\n---\nBody\n",
            "missing required frontmatter key 'name'",
        ),
        (
            b"---\nname: alpha\n---\nBody\n",
            "missing required frontmatter key 'description'",
        ),
        (
            b"---\n---\nBody\n",
            "missing required frontmatter key 'name'",
        ),
        (
            b"---\nname: 123\ndescription: ok\n---\nBody\n",
            "frontmatter 'name' must be a string",
        ),
        (
            b"---\nname: alpha\ndescription: 456\n---\nBody\n",
            "frontmatter 'description' must be a string",
        ),
        (
            b"---\nname: alpha\ndescription: ok\nvisibility: internal\n---\n"
            b"Body\n",
            "frontmatter 'visibility' must be one of",
        ),
        (
            b"---\nname: alpha\ndescription: ok\nvisibility: 1\n---\nBody\n",
            "frontmatter 'visibility' must be one of",
        ),
        (
            b"---\nname: alpha\ndescription: |\n  line one\n  line two\n---\n"
            b"Body\n",
            "frontmatter 'description' must be a single line",
        ),
        (
            b"---\nname: alpha\ndescription: ''\n---\nBody\n",
            "frontmatter 'description' must be 1 to 1024 characters",
        ),
        (
            (
                f"---\nname: alpha\ndescription: {'a' * 1025}\n---\nBody\n"
            ).encode(),
            "frontmatter 'description' must be 1 to 1024 characters",
        ),
        (
            b'---\nname: alpha\ndescription: "has\\ttab"\n---\nBody\n',
            "control or format characters",
        ),
        (
            (
                "---\nname: alpha\ndescription: has\u200bzwsp\n---\nBody\n"
            ).encode(),
            "control or format characters",
        ),
        (
            b"---\nname: alpha\ndescription: ok\n---\n   \n\n",
            "skill instructions must not be empty",
        ),
    ],
)
def test_parse_skill_md_rejects_c4_cases(
    raw: bytes, expected_msg: str
) -> None:
    skill, violations = content.parse_skill_md("alpha", raw)

    assert skill is None
    assert violations
    assert all(v.check is model.Check.FRONTMATTER for v in violations)
    assert all(v.path == "alpha/SKILL.md" for v in violations)
    assert any(expected_msg in v.message for v in violations)


def test_parse_skill_md_rejects_name_mismatch() -> None:
    data = _skill_bytes(name="other-skill", description="Mismatch.")

    skill, violations = content.parse_skill_md("alpha", data)

    assert skill is None
    assert len(violations) == 1
    assert violations[0].check is model.Check.NAME
    assert violations[0].path == "alpha/SKILL.md"
    assert "'other-skill'" in violations[0].message


def test_parse_skill_md_rejects_merge_keys_and_aliases() -> None:
    s = "{description: ok}"
    for i in range(25):
        s = f"{{<<: [&a{i} {s}, *a{i}]}}"
    nested = f"---\n{s}\n---\nBody\n".encode()
    assert len(nested) <= 1024

    skill, violations = content.parse_skill_md("alpha", nested)

    assert skill is None
    assert len(violations) == 1
    assert violations[0].check is model.Check.FRONTMATTER
    assert violations[0].path == "alpha/SKILL.md"

    alias_doc = b"---\nname: &a alpha\ndescription: *a\n---\nBody\n"
    skill, violations = content.parse_skill_md("alpha", alias_doc)

    assert skill is None
    assert len(violations) == 1
    assert violations[0].check is model.Check.FRONTMATTER
    assert violations[0].path == "alpha/SKILL.md"

    merge_doc = b"---\nname: alpha\n<<: {description: ok}\n---\nBody\n"
    skill, violations = content.parse_skill_md("alpha", merge_doc)

    assert skill is None
    assert len(violations) == 1
    assert violations[0].check is model.Check.FRONTMATTER
    assert violations[0].path == "alpha/SKILL.md"
    assert "merge key" in violations[0].message


@pytest.mark.parametrize(
    "description",
    [
        pytest.param("2026-99-99", id="invalid-timestamp"),
        pytest.param("9" * 5000, id="oversized-integer"),
        pytest.param("[" * 600 + "0" + "]" * 600, id="deep-nesting"),
    ],
)
def test_parse_skill_md_rejects_unusual_yaml(description: str) -> None:
    data = _skill_bytes(description=description)
    assert len(data) < model.Limits().max_skill_md_bytes

    skill, violations = content.parse_skill_md("alpha", data)

    assert skill is None
    assert len(violations) == 1
    assert violations[0].check is model.Check.FRONTMATTER
    assert violations[0].path == "alpha/SKILL.md"
    assert "invalid frontmatter YAML" in violations[0].message


# ---- write_version ----


def test_write_version_creates_immutable_tree(tmp_path: pathlib.Path) -> None:
    alpha_md = _skill_bytes(
        name="alpha",
        description="Alpha skill.",
        body="Alpha instructions.\n",
    )
    beta_md = _skill_bytes(
        name="beta",
        description="Beta skill.",
        body="Beta instructions.\n",
        extra="visibility: public\n",
    )
    resource_bytes = b"Reference data.\n"
    repo, rev, entries = _make_repo(
        tmp_path,
        {
            "beta/SKILL.md": beta_md,
            "alpha/SKILL.md": alpha_md,
            "alpha/resources/ref.txt": resource_bytes,
        },
    )
    versions_root = tmp_path / "versions"

    version = content.write_version(
        repo,
        "team1",
        rev,
        entries,
        versions_root,
        seq=1,
        limits=model.Limits(),
    )

    expected_path = versions_root / "team1" / f"{rev}-1"
    assert version.team == "team1"
    assert version.revision == rev
    assert version.path == expected_path
    assert [s.name for s in version.skills] == ["alpha", "beta"]
    assert (
        version.skills[0].instructions_digest
        == hashlib.sha256(b"Alpha instructions.").hexdigest()
    )
    assert version.blob_digests == {
        "alpha/SKILL.md": hashlib.sha256(alpha_md).hexdigest(),
        "alpha/resources/ref.txt": hashlib.sha256(resource_bytes).hexdigest(),
        "beta/SKILL.md": hashlib.sha256(beta_md).hexdigest(),
    }
    assert (expected_path / "alpha" / "SKILL.md").read_bytes() == alpha_md
    assert (
        expected_path / "alpha" / "resources" / "ref.txt"
    ).read_bytes() == resource_bytes
    assert (expected_path.stat().st_mode & 0o777) == 0o555
    assert ((expected_path / "alpha").stat().st_mode & 0o777) == 0o555
    assert (
        (expected_path / "alpha" / "resources").stat().st_mode & 0o777
    ) == 0o555
    assert (
        (expected_path / "alpha" / "SKILL.md").stat().st_mode & 0o777
    ) == 0o444
    assert (
        (expected_path / "alpha" / "resources" / "ref.txt").stat().st_mode
        & 0o777
    ) == 0o444


def test_write_version_rejects_and_leaves_nothing_on_disk(
    tmp_path: pathlib.Path,
) -> None:
    repo, rev, entries = _make_repo(
        tmp_path,
        {
            "alpha/SKILL.md": _skill_bytes(name="wrong-name"),
            "alpha/resources/bad.txt": b"bad\r\nline",
        },
    )
    versions_root = tmp_path / "versions"

    with pytest.raises(content.ContentRejected) as caught:
        content.write_version(
            repo,
            "team1",
            rev,
            entries,
            versions_root,
            seq=1,
            limits=model.Limits(),
        )

    assert [v.check for v in caught.value.violations] == [
        model.Check.NAME,
        model.Check.ENCODING,
    ]
    assert not versions_root.exists()


def test_write_version_existing_destination_raises_and_cleans_temp(
    tmp_path: pathlib.Path,
) -> None:
    repo, rev, entries = _make_repo(
        tmp_path,
        {"alpha/SKILL.md": _skill_bytes(name="alpha")},
    )
    versions_root = tmp_path / "versions"
    existing = versions_root / "team1" / f"{rev}-1"
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        content.write_version(
            repo,
            "team1",
            rev,
            entries,
            versions_root,
            seq=1,
            limits=model.Limits(),
        )

    assert list(versions_root.iterdir()) == [versions_root / "team1"]
    assert list((versions_root / "team1").iterdir()) == [existing]


def test_write_version_cleans_up_when_fsync_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, rev, entries = _make_repo(
        tmp_path,
        {"alpha/SKILL.md": _skill_bytes(name="alpha")},
    )
    versions_root = tmp_path / "versions"

    def fail_fsync(_fd: int) -> None:
        msg = "simulated disk error"
        raise OSError(msg)

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated disk error"):
        content.write_version(
            repo,
            "team1",
            rev,
            entries,
            versions_root,
            seq=1,
            limits=model.Limits(),
        )

    assert list(versions_root.iterdir()) == []


def test_write_version_cleans_up_destination_when_parent_fsync_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, rev, entries = _make_repo(
        tmp_path,
        {"alpha/SKILL.md": _skill_bytes(name="alpha")},
    )
    versions_root = tmp_path / "versions"
    dest = versions_root / "team1" / f"{rev}-1"
    real_fsync_dir = content._fsync_dir
    calls = 0

    def fail_parent_fsync(path: pathlib.Path) -> None:
        nonlocal calls
        if path == dest.parent:
            calls += 1
            msg = f"simulated parent fsync error {calls}"
            raise OSError(msg)
        real_fsync_dir(path)

    monkeypatch.setattr(content, "_fsync_dir", fail_parent_fsync)

    with pytest.raises(OSError, match="simulated parent fsync error 1"):
        content.write_version(
            repo,
            "team1",
            rev,
            entries,
            versions_root,
            seq=1,
            limits=model.Limits(),
        )

    assert calls == 2
    assert not dest.exists()
