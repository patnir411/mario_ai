"""Deterministic state and artifact provenance for emulator-backed experiments.

The GBA adapters expose the complete emulator savestate as bytes.  For those
backends ``snapshot_digest`` is an exact byte-level identity check.  Other
structured snapshots may still receive a deterministic digest, but the returned
``exact`` flag stays false unless the raw emulator payload is byte-addressable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Any, Mapping


class SnapshotEncodingError(TypeError):
    """Raised when a value cannot be encoded without repr/pickle ambiguity."""


class StateAliasError(RuntimeError):
    """One symbolic state was assigned two distinct physical snapshots."""

    def __init__(self, collision: dict):
        self.collision = collision
        state = collision.get("state")
        retained = (
            collision.get("retained", {}).get("digest", {}).get("full_sha256")
        )
        rejected = (
            collision.get("rejected", {}).get("digest", {}).get("full_sha256")
        )
        super().__init__(
            f"physical-state alias for MetaState {state}: "
            f"{retained} != {rejected}"
        )


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of the exact artifact bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _framed(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">Q", len(payload)) + payload


def stable_encode(value: Any) -> bytes:
    """Encode ordinary snapshot metadata deterministically.

    This deliberately rejects unknown objects instead of falling back to
    ``repr`` or pickle, neither of which is a portable scientific identity.
    """
    if value is None:
        return _framed(b"n", b"")
    if isinstance(value, bool):
        return _framed(b"b", b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return _framed(b"i", str(value).encode("ascii"))
    if isinstance(value, float):
        return _framed(b"f", struct.pack(">d", value))
    if isinstance(value, str):
        return _framed(b"s", value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _framed(b"y", bytes(value))
    if isinstance(value, Path):
        return _framed(b"p", str(value).encode("utf-8"))
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            encoded_key = stable_encode(key)
            items.append((encoded_key, stable_encode(item)))
        items.sort(key=lambda pair: pair[0])
        return _framed(
            b"d",
            b"".join(_framed(b"k", key) + _framed(b"v", item)
                     for key, item in items),
        )
    if isinstance(value, tuple):
        return _framed(b"t", b"".join(stable_encode(item) for item in value))
    if isinstance(value, list):
        return _framed(b"l", b"".join(stable_encode(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return _framed(
            b"e",
            b"".join(sorted(stable_encode(item) for item in value)),
        )

    # NumPy arrays/scalars are optional dependencies, so use their protocol
    # without importing NumPy in the core package.
    if all(hasattr(value, attr) for attr in ("dtype", "shape", "tobytes")):
        payload = (
            stable_encode(str(value.dtype))
            + stable_encode(tuple(int(v) for v in value.shape))
            + stable_encode(value.tobytes(order="C"))
        )
        return _framed(b"a", payload)
    if hasattr(value, "item"):
        scalar = value.item()
        if scalar is not value:
            return _framed(b"q", stable_encode(scalar))
    raise SnapshotEncodingError(
        f"no deterministic snapshot encoding for {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


@dataclass(frozen=True)
class SnapshotDigest:
    """Identity of an in-memory snapshot plus its future-determining context."""

    full_sha256: str
    emulator_sha256: str
    metadata_sha256: str
    adapter_context_sha256: str
    emulator_bytes: int | None
    exact: bool
    backend: str | None = None
    rom_sha1: str | None = None

    def to_json(self) -> dict:
        return {
            "full_sha256": self.full_sha256,
            "emulator_sha256": self.emulator_sha256,
            "metadata_sha256": self.metadata_sha256,
            "adapter_context_sha256": self.adapter_context_sha256,
            "emulator_bytes": self.emulator_bytes,
            "exact": self.exact,
            "backend": self.backend,
            "rom_sha1": self.rom_sha1,
        }


def snapshot_digest(
    snapshot: Any,
    *,
    adapter_context: Mapping[str, Any] | None = None,
    backend: str | None = None,
    rom_sha1: str | None = None,
) -> SnapshotDigest:
    """Hash a snapshot with domain-separated emulator and wrapper identities."""
    if (
        isinstance(snapshot, tuple)
        and len(snapshot) == 2
        and isinstance(snapshot[0], (bytes, bytearray, memoryview))
    ):
        emulator_payload = bytes(snapshot[0])
        metadata = snapshot[1]
        exact = True
        emulator_bytes = len(emulator_payload)
        emulator_encoded = _framed(b"raw-emulator-bytes", emulator_payload)
        emulator_sha = hashlib.sha256(emulator_payload).hexdigest()
    else:
        emulator_encoded = stable_encode(snapshot)
        metadata = None
        exact = False
        emulator_bytes = None
        emulator_sha = hashlib.sha256(emulator_encoded).hexdigest()

    metadata_encoded = stable_encode(metadata)
    context = dict(adapter_context or {})
    context_encoded = stable_encode(context)
    metadata_sha = hashlib.sha256(metadata_encoded).hexdigest()
    context_sha = hashlib.sha256(context_encoded).hexdigest()
    identity = (
        _framed(b"domain", b"mario-ai.snapshot.v1")
        + _framed(b"emulator", emulator_encoded)
        + _framed(b"metadata", metadata_encoded)
        + _framed(b"adapter-context", context_encoded)
        + _framed(b"backend", stable_encode(backend))
        + _framed(b"rom-sha1", stable_encode(rom_sha1))
    )
    return SnapshotDigest(
        full_sha256=hashlib.sha256(identity).hexdigest(),
        emulator_sha256=emulator_sha,
        metadata_sha256=metadata_sha,
        adapter_context_sha256=context_sha,
        emulator_bytes=emulator_bytes,
        exact=exact,
        backend=backend,
        rom_sha1=rom_sha1,
    )
