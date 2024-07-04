#  Copyright 2024 by REDxEYE.
#  All rights reserved.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

from x4.file_utils import FileBuffer, Buffer


@dataclass
class CatEntry:
    path: str
    offset: int
    size: int
    flags: int
    hash: int


class DatArchive:
    def __init__(self, path: Path):
        self.entries: Dict[str, CatEntry] = {}
        self.path = path
        self.buffer = FileBuffer(path)
        if self.buffer.size() < 1024 * 1024 * 1024:
            self.buffer = self.buffer.slice()
        if self.path.with_suffix(".cat").exists():
            running_offset = 0
            for line in self.path.with_suffix(".cat").open("r", encoding="utf-8"):
                path, size, flags, shash = line.rsplit(" ", 3)
                self.entries[path] = CatEntry(path, running_offset, int(size), int(flags), int(shash, 16))
                running_offset += int(size)

    def get(self, filename: str) -> Buffer | None:
        if entry := self.entries.get(filename, None):
            return self.buffer.slice(entry.offset, entry.size)
        return None

    def __iter__(self) -> Iterator[tuple[str, Buffer]]:
        for entry in self.entries.values():
            yield entry.path, self.buffer.slice(entry.offset, entry.size)
