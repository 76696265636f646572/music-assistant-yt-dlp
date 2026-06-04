"""On-disk file cache for YouTube VOD audio."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SIDECAR_NAME = "metadata.json"
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(frozen=True)
class CachedStreamInfo:
    """Validated cache hit with paths and format metadata."""

    video_id: str
    audio_path: Path
    metadata: dict[str, Any]


class FileCache:
    """File cache keyed by YouTube video ID with JSON sidecar metadata."""

    def __init__(self, cache_dir: Path, max_size_bytes: int | None = None) -> None:
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_bytes

    def ensure_ready(self) -> None:
        """Create the cache root directory if needed."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_video_id(video_id: str) -> str:
        """Return a safe cache key for a YouTube video ID."""
        if not _VIDEO_ID_RE.match(video_id):
            msg = f"Invalid YouTube video ID: {video_id!r}"
            raise ValueError(msg)
        return video_id

    def entry_dir(self, video_id: str) -> Path:
        """Directory for one cached video."""
        return self.cache_dir / self.sanitize_video_id(video_id)

    def sidecar_path(self, video_id: str) -> Path:
        """Path to the JSON sidecar for a video."""
        return self.entry_dir(video_id) / _SIDECAR_NAME

    def get_hit(self, video_id: str) -> CachedStreamInfo | None:
        """Return cache info if a valid cached file exists, else None."""
        entry_dir = self.entry_dir(video_id)
        sidecar = entry_dir / _SIDECAR_NAME
        if not sidecar.is_file():
            return None
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        audio_name = metadata.get("audio_file")
        if not audio_name:
            return None
        audio_path = entry_dir / audio_name
        if not self._is_valid_audio_file(audio_path):
            return None
        metadata["last_accessed"] = time.time()
        metadata["size"] = audio_path.stat().st_size
        try:
            sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except OSError:
            pass
        return CachedStreamInfo(video_id=video_id, audio_path=audio_path, metadata=metadata)

    def commit(
        self,
        video_id: str,
        temp_audio_path: Path,
        metadata: dict[str, Any],
    ) -> Path:
        """Atomically move a completed download into the cache."""
        entry_dir = self.entry_dir(video_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        ext = metadata.get("audio_ext") or temp_audio_path.suffix.lstrip(".") or "bin"
        final_name = f"audio.{ext}"
        final_path = entry_dir / final_name
        if temp_audio_path.resolve() != final_path.resolve():
            if final_path.exists():
                final_path.unlink()
            os.replace(temp_audio_path, final_path)
        sidecar_data = {
            **metadata,
            "video_id": video_id,
            "audio_file": final_name,
            "cached_at": time.time(),
            "last_accessed": time.time(),
            "size": final_path.stat().st_size,
        }
        sidecar = entry_dir / _SIDECAR_NAME
        sidecar.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")
        if self.max_size_bytes:
            self.enforce_size_limit()
        return final_path

    def temp_download_stem(self, video_id: str) -> Path:
        """Return a unique path stem (no extension) for an in-progress download."""
        entry_dir = self.entry_dir(video_id)
        entry_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="dl-", dir=entry_dir)
        os.close(fd)
        temp_path = Path(path)
        temp_path.unlink(missing_ok=True)
        return temp_path

    def clear_all(self) -> int:
        """Remove all cached entries. Returns number of video directories removed."""
        if not self.cache_dir.is_dir():
            return 0
        removed = 0
        for child in self.cache_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
            elif child.is_file():
                child.unlink(missing_ok=True)
        return removed

    def enforce_size_limit(self) -> None:
        """Evict least-recently-used entries until total size is within the limit."""
        if not self.max_size_bytes or not self.cache_dir.is_dir():
            return
        entries: list[tuple[float, Path, int]] = []
        total = 0
        for entry_dir in self.cache_dir.iterdir():
            if not entry_dir.is_dir():
                continue
            sidecar = entry_dir / _SIDECAR_NAME
            if not sidecar.is_file():
                shutil.rmtree(entry_dir, ignore_errors=True)
                continue
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                shutil.rmtree(entry_dir, ignore_errors=True)
                continue
            audio_name = metadata.get("audio_file")
            if not audio_name:
                shutil.rmtree(entry_dir, ignore_errors=True)
                continue
            audio_path = entry_dir / audio_name
            if not self._is_valid_audio_file(audio_path):
                shutil.rmtree(entry_dir, ignore_errors=True)
                continue
            size = audio_path.stat().st_size
            last_accessed = float(metadata.get("last_accessed", metadata.get("cached_at", 0)))
            entries.append((last_accessed, entry_dir, size))
            total += size
        if total <= self.max_size_bytes:
            return
        entries.sort(key=lambda item: item[0])
        for _, entry_dir, size in entries:
            if total <= self.max_size_bytes:
                break
            shutil.rmtree(entry_dir, ignore_errors=True)
            total -= size

    @staticmethod
    def _is_valid_audio_file(path: Path) -> bool:
        """Return True if path exists and has non-zero size."""
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False
