#  Copyright 2025 by REDxEYE.
#  All rights reserved.
from collections.abc import Iterable

from x4.dat import DatArchive
from x4.file_utils import Buffer
from x4.tiny_path import TinyPath


class ArchiveManager:
    def __init__(self, game_root: TinyPath):
        self._archives: list[DatArchive] = []
        for archive_id in range(100):
            archive_path = game_root / f"{archive_id:02}.dat"
            if archive_path.exists():
                self._archives.append(DatArchive(archive_path))

    def get(self, file_path: TinyPath) -> Buffer | None:
        for archive in self._archives:
            if buffer := archive.get(TinyPath(file_path)):
                return buffer
        return None

    def glob(self, pattern: str) -> Iterable[tuple[TinyPath,Buffer]]:
        for archive in self._archives:
            yield from archive.glob(pattern)
