#  Copyright 2024 by REDxEYE.
#  All rights reserved.

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from x4.file_utils import Buffer
from zlib import decompress


class AttributeType(IntEnum):
    Invalid = -1
    Float2 = 1
    Float3 = 2
    Byte4 = 4


class AttributeUsage(IntEnum):
    POS = 0
    NORMAL = 3
    TANGENT = 6
    UV = 5
    COLOR = 10


@dataclass
class Attribute:
    type: AttributeType
    usage: AttributeUsage
    index: int
    unk0: int
    unk1: int

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Attribute':
        return cls(AttributeType(buffer.read_int32()), AttributeUsage(buffer.read_uint8()), *buffer.read_fmt("3b"))


@dataclass
class CompressedBuffer:
    unk0: int
    unk1: int
    c_offset: int
    unk3: int
    unk4: float
    unk5: int
    c_size: int
    item_count: int
    item_size: int
    unk9: int
    unk10: int
    unk11: int
    unk12: float
    attributes: list[Attribute]

    # struct{int a,b;}unk14[16];

    data: bytes = field(init=False, default=b"", repr=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        (unk0, unk1, c_offset, unk3, unk4, unk5, c_size, unk7, unk8, unk9, unk10, unk11, unk12,
         used_attrib_count,) = buffer.read_fmt("4if7idi")
        attributes = [Attribute.from_buffer(buffer) for _ in range(16)]
        return cls(unk0, unk1, c_offset, unk3, unk4, unk5, c_size, unk7, unk8, unk9, unk10, unk11, unk12,
                   attributes[:used_attrib_count])

    def get_data(self) -> np.ndarray:
        items = []
        for attribute in self.attributes:
            attr_name = f"{attribute.usage.name.lower()}{attribute.index}"
            if attribute.type == AttributeType.Float2:
                items.append((attr_name, np.float32, (2,)))
            elif attribute.type == AttributeType.Float3:
                items.append((attr_name, np.float32, (3,)))
            elif attribute.type == AttributeType.Byte4:
                items.append((attr_name, np.uint8, (4,)))
        dtype = np.dtype(items)
        return np.frombuffer(self.data, dtype)


@dataclass
class MaterialStrip:
    start_index: int
    count: int
    name: str

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_uint32(), buffer.read_uint32(), buffer.read_ascii_string(128))


@dataclass
class StaticModel:
    buffers: list[CompressedBuffer]
    strips: list[MaterialStrip]
    vertex_count: int
    index_count: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        ident = buffer.read_fourcc()
        if ident != "XUMF":
            return None
        (version_maj, endian, data_start, unk7,
         buffer_count, buffer_info_size, strip_count, strip_info_size,
         unkc, unkd) = buffer.read_fmt("10B")
        if version_maj != 3:
            return None
        if endian:
            buffer.set_big_endian()
        (vertex_count, index_count, unk16, unk1a, unk1e, unk26, unk2a, unk32,) = buffer.read_fmt("8I")

        buffer.seek(data_start)
        compressed_buffers = [CompressedBuffer.from_buffer(buffer) for _ in range(buffer_count)]
        strips = [MaterialStrip.from_buffer(buffer) for _ in range(strip_count)]

        compressed_offset = data_start + buffer_count * buffer_info_size + strip_count * strip_info_size
        for compressed_buffer in compressed_buffers:
            buffer.seek(compressed_offset + compressed_buffer.c_offset)
            compressed_buffer.data = decompress(buffer.read(compressed_buffer.c_size))

        return cls(compressed_buffers, strips, vertex_count, index_count)
