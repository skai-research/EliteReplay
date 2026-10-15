"""Immutable artifact writers and stage manifests for EliteReplay."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import canonical_json, content_hash


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable_text(path: str | Path, text: str) -> Path:
    """Create *path*, or verify that an existing artifact is byte-identical."""

    destination = Path(path)
    encoded = text.encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError(
                f"refusing to change immutable artifact: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # The pipeline is single-writer. A concurrent writer is still checked
        # after replacement by all downstream content hashes.
        os.replace(temporary, destination)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return destination


def write_immutable_json(path: str | Path, value: Any) -> Path:
    return write_immutable_text(path, canonical_json(value) + "\n")


def write_immutable_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    return write_immutable_text(
        path, "".join(canonical_json(row) + "\n" for row in rows)
    )


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def runtime_provenance(
    *, project_root: str | Path | None = None, trinity_root: str | Path | None = None
) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in (
        "evorl",
        "torch",
        "transformers",
        "vllm",
        "ray",
        "trinity-rft",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "project_git_commit": (
            None if project_root is None else _git_revision(Path(project_root))
        ),
        "trinity_git_commit": (
            None if trinity_root is None else _git_revision(Path(trinity_root))
        ),
    }


def write_stage_manifest(
    path: str | Path,
    *,
    stage: str,
    config_hash: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, str | Path],
    models: Mapping[str, str],
    counts: Mapping[str, int],
    provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Write a completed-stage manifest and validate it on resumed calls."""

    destination = Path(path)
    output_records = {
        name: {"path": str(Path(output)), "sha256": sha256_file(output)}
        for name, output in outputs.items()
    }
    identity = {
        "schema_version": "EliteReplay.stage_manifest.v1",
        "stage": stage,
        "config_hash": config_hash,
        "inputs": dict(inputs),
        "outputs": output_records,
        "models": dict(models),
        "counts": dict(counts),
    }
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        existing_identity = {
            key: existing.get(key)
            for key in (
                "schema_version",
                "stage",
                "config_hash",
                "inputs",
                "outputs",
                "models",
                "counts",
            )
        }
        if canonical_json(existing_identity) != canonical_json(identity):
            raise FileExistsError(
                f"completed stage manifest conflicts with current run: {destination}"
            )
        return existing

    manifest = {
        **identity,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provenance": dict(provenance or {}),
    }
    manifest["manifest_content_hash"] = content_hash(
        {key: value for key, value in manifest.items() if key != "completed_at"}
    )
    write_immutable_json(destination, manifest)
    return manifest


def validate_output_hashes(manifest: Mapping[str, Any]) -> None:
    for name, record in manifest.get("outputs", {}).items():
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifest output {name!r} is missing: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"manifest output {name!r} changed after completion")
