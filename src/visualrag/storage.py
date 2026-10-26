"""Pluggable storage backend abstractions for visualrag."""

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for persisting or caching slides, screen frames and deictic images."""

    def save(self, image_bytes: bytes, filename: str) -> str | None:
        """Save image bytes under a specified filename or identifier.

        Args:
            image_bytes: Raw compressed image bytes (e.g. PNG).
            filename: Target relative filename or unique identifier.

        Returns:
            The persistent URI or local file path string or None.
        """


class LocalStorage:
    """Persists images to a directory on the local filesystem."""

    def __init__(
        self,
        output_dir: Path | str = "slides",
        format: str = "png",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.format = format
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image_bytes: bytes, filename: str) -> str:
        dest = self.output_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(image_bytes)
        return str(dest)


class MemoryStorage:
    """Holds images purely in-memory with zero disk I/O."""

    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}

    @staticmethod
    def _normalize_key(key_or_uri: str) -> str:
        if key_or_uri.startswith("memory://"):
            return key_or_uri[9:]
        return key_or_uri

    def save(self, image_bytes: bytes, filename: str) -> str:
        key = self._normalize_key(filename)
        self._storage[key] = image_bytes
        return f"memory://{key}"

    def get(self, key_or_uri: str) -> bytes | None:
        return self._storage.get(self._normalize_key(key_or_uri))

    def clear(self) -> None:
        self._storage.clear()

    def __len__(self) -> int:
        return len(self._storage)

    def __contains__(self, key_or_uri: str) -> bool:
        return self._normalize_key(key_or_uri) in self._storage


class CallbackStorage:
    """Wraps a callable callback function into a StorageBackend."""

    def __init__(self, callback: Callable[[bytes, str], str | None]) -> None:
        self._callback = callback

    def save(self, image_bytes: bytes, filename: str) -> str | None:
        return self._callback(image_bytes, filename)
