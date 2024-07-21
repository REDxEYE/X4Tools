#  Copyright 2024 by REDxEYE.
#  All rights reserved.

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from x4.file_utils import Buffer, WritableMemoryBuffer
from zlib import decompress, compress, Z_BEST_COMPRESSION


class PrimitiveType(IntEnum):
    POINTLIST = 1,
    LINELIST = 2,
    LINESTRIP = 3,
    TRIANGLELIST = 4,
    TRIANGLESTRIP = 5,
    TRIANGLEFAN = 6,


class D3DAttributeType(IntEnum):
    INVALID = -1
    FLOAT1 = 0,
    FLOAT2 = 1,
    FLOAT3 = 2,
    FLOAT4 = 3,
    D3DCOLOR = 4,
    UBYTE4 = 5,
    SHORT2 = 6,
    SHORT4 = 7,
    UBYTE4N = 8,
    SHORT2N = 9,
    SHORT4N = 10,
    USHORT2N = 11,
    USHORT4N = 12,
    UDEC3 = 13,
    DEC3N = 14,
    FLOAT16_2 = 15,
    FLOAT16_4 = 16,
    UNUSED = 17


class AttributeUsage(IntEnum):
    INVALID = -1
    POSITION = 0,
    BLENDWEIGHT = 1,
    BLENDINDICES = 2,
    NORMAL = 3,
    PSIZE = 4,
    TEXCOORD = 5,
    TANGENT = 6,
    BINORMAL = 7,
    TESSFACTOR = 8,
    POSITIONT = 9,
    COLOR = 10,
    FOG = 11,
    DEPTH = 12,
    SAMPLE = 13


@dataclass
class Attribute:
    type: D3DAttributeType
    usage: AttributeUsage
    index: int
    pad0: int = 0
    pad1: int = 0

    @classmethod
    def from_buffer(cls, buffer: Buffer) -> 'Attribute':
        return cls(D3DAttributeType(buffer.read_int32()), AttributeUsage(buffer.read_uint8()), *buffer.read_fmt("3b"))

    def to_buffer(self, buffer: Buffer):
        buffer.write_fmt("i4b", self.type, self.usage, self.index, self.pad0, self.pad1)


@dataclass
class CompressedBuffer:
    type: int
    usage_id: int
    c_offset: int
    compressed: int
    unk4: float
    format: int
    c_size: int
    item_count: int
    item_size: int
    section_count: int
    unk10: int = 0
    unk11: int = 0.0
    unk12: float = 0
    attributes: list[Attribute] = field(default_factory=list)

    data: bytes = field(init=False, default=b"", repr=False)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        (type_id, usage_id, c_offset, compressed, unk4, format_id, c_size, item_count, item_size, section_count, unk10,
         unk11, unk12,
         used_attrib_count,) = buffer.read_fmt("4if7idi")
        attributes = [Attribute.from_buffer(buffer) for _ in range(16)]
        return cls(type_id, usage_id, c_offset, compressed, unk4, format_id, c_size, item_count, item_size,
                   section_count, unk10, unk11, unk12, attributes[:used_attrib_count])

    def get_data(self) -> np.ndarray:
        items = []
        for attribute in self.attributes:
            attr_name = f"{attribute.usage.name.lower()}{attribute.index}"
            match attribute.type:
                case D3DAttributeType.FLOAT1:
                    items.append((attr_name, np.float32, (1,)))
                case D3DAttributeType.FLOAT2:
                    items.append((attr_name, np.float32, (2,)))
                case D3DAttributeType.FLOAT3:
                    items.append((attr_name, np.float32, (3,)))
                case D3DAttributeType.FLOAT4:
                    items.append((attr_name, np.float32, (4,)))
                case D3DAttributeType.D3DCOLOR | D3DAttributeType.UBYTE4 | D3DAttributeType.UBYTE4N:
                    items.append((attr_name, np.uint8, (4,)))
                case D3DAttributeType.SHORT2 | D3DAttributeType.SHORT2N:
                    items.append((attr_name, np.int16, (2,)))
                case D3DAttributeType.SHORT4 | D3DAttributeType.SHORT4N:
                    items.append((attr_name, np.int16, (4,)))
                case D3DAttributeType.USHORT2N:
                    items.append((attr_name, np.uint16, (2,)))
                case D3DAttributeType.USHORT4N:
                    items.append((attr_name, np.uint16, (2,)))
                case D3DAttributeType.FLOAT16_2:
                    items.append((attr_name, np.float16, (2,)))
                case D3DAttributeType.FLOAT16_4:
                    items.append((attr_name, np.float16, (4,)))
                case _:
                    raise ValueError(f"Unknown attribute type {attribute.type!r}")
        dtype = np.dtype(items)
        return np.frombuffer(self.data, dtype)

    def to_buffer(self, buffer: Buffer):
        buffer.write_fmt("4if7id", self.type, self.usage_id, self.c_offset, self.compressed, self.unk4, self.format,
                         self.c_size, self.item_count, self.item_size, self.section_count, self.unk10, self.unk11,
                         self.unk12)
        buffer.write_uint32(len(self.attributes))
        for attribute in self.attributes:
            attribute.to_buffer(buffer)
        for _ in range(16 - len(self.attributes)):
            buffer.write_fmt("i4b", -1, 0, 0, 0, 0)


@dataclass
class MaterialStrip:
    start_index: int
    count: int
    name: str

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        return cls(buffer.read_uint32(), buffer.read_uint32(), buffer.read_ascii_string(128))

    def to_buffer(self, buffer: Buffer):
        buffer.write_fmt("2I128s", self.start_index, self.count, self.name.encode("ascii"))


@dataclass
class StaticModel:
    buffers: list[CompressedBuffer]
    strips: list[MaterialStrip]
    vertex_count: int
    index_count: int
    primitive_type: PrimitiveType
    bbox_center: tuple[float, float, float]
    bbox_size: tuple[float, float, float]

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        ident = buffer.read_fourcc()
        if ident != "XUMF":
            return None
        (version_maj, endian, header_size, unused,
         buffer_count, buffer_info_size, strip_count, strip_info_size,
         cw_culling, right_hand) = buffer.read_fmt("10B")
        if version_maj != 3:
            return None
        if endian:
            buffer.set_big_endian()
        (vertex_count, index_count, prim_type, mesh_optimizations) = buffer.read_fmt("4I")
        bbox_center = buffer.read_fmt("3f")
        bbox_size = buffer.read_fmt("3f")

        buffer.seek(header_size)
        compressed_buffers = [CompressedBuffer.from_buffer(buffer) for _ in range(buffer_count)]
        strips = [MaterialStrip.from_buffer(buffer) for _ in range(strip_count)]

        compressed_offset = header_size + buffer_count * buffer_info_size + strip_count * strip_info_size
        for compressed_buffer in compressed_buffers:
            buffer.seek(compressed_offset + compressed_buffer.c_offset)
            data = buffer.read(compressed_buffer.c_size)
            if compressed_buffer.compressed:
                compressed_buffer.data = decompress(data)
            else:
                compressed_buffer.data = data

        return cls(compressed_buffers, strips, vertex_count, index_count, PrimitiveType(prim_type), bbox_center,
                   bbox_size)

    def to_buffer(self, buffer: Buffer):
        buffer.write_fmt("4s10B4I6f", b"XUMF", 3, False, 0x40, 0,
                         len(self.buffers), 188,
                         len(self.strips), 136,
                         False, False,
                         self.vertex_count, self.index_count,
                         self.primitive_type, 0,
                         *self.bbox_center, *self.bbox_size
                         )
        buffer.seek(0x40)
        compressed_data = WritableMemoryBuffer()
        for compressed_buffer in self.buffers:
            if compressed_buffer.compressed:
                compressed = compress(compressed_buffer.data, Z_BEST_COMPRESSION)
            else:
                compressed = compressed_data.data
            compressed_buffer.c_size = len(compressed)
            compressed_buffer.c_offset = compressed_data.tell()
            compressed_data.write(compressed)

            compressed_buffer.to_buffer(buffer)
        for strip in self.strips:
            strip.to_buffer(buffer)

        buffer.write(compressed_data.data)
