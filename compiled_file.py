#  Copyright 2024 by REDxEYE.
#  All rights reserved.


import warnings
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Any, Optional, TypeVar

import numpy as np

from x4.file_utils import Buffer, WritableMemoryBuffer


class ActorChunk(ABC):

    def to_buffer(self, buffer: Buffer):
        pass


class ActorChunkId(IntEnum):
    NODE = 0
    MESH = 1
    SKINNINGINFO = 2
    STDMATERIAL = 3
    STDMATERIALLAYER = 4
    FXMATERIAL = 5
    LIMIT = 6
    INFO = 7
    MESHLODLEVELS = 8
    STDPROGMORPHTARGET = 9
    NODEGROUPS = 10
    NODES = 11  # Actor_Nodes
    STDPMORPHTARGETS = 12  # Actor_MorphTargets
    MATERIALINFO = 13  # Actor_MaterialInfo
    NODEMOTIONSOURCES = 14  # Actor_NodeMotionSources
    ATTACHMENTNODES = 15  # Actor_AttachmentNodes
    MATERIALATTRIBUTESET = 16
    GENERICMATERIAL = 17  # Actor_GenericMaterial
    PHYSICSSETUP = 18
    SIMULATEDOBJECTSETUP = 19


class LayerId(IntEnum):
    UNKNOWN = 0  # unknown layer
    AMBIENT = 1  # ambient layer
    DIFFUSE = 2  # a diffuse layer
    SPECULAR = 3  # specular layer
    OPACITY = 4  # opacity layer
    BUMP = 5  # bump layer
    SELFILLUM = 6  # self illumination layer
    SHINE = 7  # shininess (for specular)
    SHINESTRENGTH = 8  # shine strength (for specular)
    FILTERCOLOR = 9  # filter color layer
    REFLECT = 10  # reflection layer
    REFRACT = 11  # refraction layer
    ENVIRONMENT = 12  # environment map layer
    DISPLACEMENT = 13  # displacement map layer


@dataclass
class CompiledHeader:
    ident: str
    version: tuple[int, int]
    endian: int
    multiply_order: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        ident = buffer.read_fourcc()
        version_maj, version_min, unk0, unk1 = buffer.read_fmt("4B")
        return cls(ident, (version_maj, version_min), unk0, unk1)

    def to_buffer(self, buffer: Buffer):
        buffer.write_fmt("4s4b", self.ident.encode("ascii"), *self.version, self.endian, self.multiply_order)


@dataclass
class MetaData(ActorChunk):
    reposition_mask: int
    repositioning_node: int
    exporter_high_version: int
    exporter_low_version: int
    retarget_root_offset: float
    exporter: str
    source_file: str
    export_date: str
    actor_name: str

    @classmethod
    def from_buffer_v2(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'MetaData':
        (reposition_mask,
         trajectory_node_index,
         exporter_high_version, exporter_low_version, *pad, retarget_root_offset) = buffer.read_fmt("Ii4bf")
        exporter = buffer.read_sized_string()
        source_file = buffer.read_sized_string()
        export_date = buffer.read_sized_string()
        actor_name = buffer.read_sized_string()
        return cls(reposition_mask, trajectory_node_index, exporter_high_version, exporter_low_version,
                   retarget_root_offset,
                   exporter, source_file, export_date, actor_name)

    def to_buffer_v2(self, buffer: Buffer):
        buffer.write_fmt("Ii4b",
                         self.reposition_mask,
                         self.repositioning_node,
                         self.exporter_high_version, self.exporter_low_version, 0, 0)
        buffer.write_float(self.retarget_root_offset)
        buffer.write_sized_string(self.exporter)
        buffer.write_sized_string(self.source_file)
        buffer.write_sized_string(self.export_date)
        buffer.write_sized_string(self.actor_name)

    def to_buffer(self, buffer: Buffer):
        chunk_buffer = WritableMemoryBuffer()
        self.to_buffer_v2(chunk_buffer)
        buffer.write_fmt("3I", ActorChunkId.INFO, chunk_buffer.size(), 2)
        buffer.write(chunk_buffer.data)


@dataclass
class Node:
    name: str
    quat: tuple[float, ...]
    scale_rot: tuple[float, ...]
    pos: tuple[float, ...]
    scale: tuple[float, ...]
    pad_: tuple[float, ...]
    unk_index0: int
    unk_index1: int
    parent_id: int
    child_count: int
    use_in_bounds_calc: int
    oobb: np.ndarray[np.float32]
    importance_factor: float

    id: int = field(init=False, default=0)
    children: list['Node'] = field(init=False, default_factory=list)
    parent: Optional['Node'] = field(init=False, default=None)

    @classmethod
    def from_buffer_v1(cls, buffer: Buffer) -> 'Node':
        quat = tuple(buffer.read_fmt("4f"))
        scale_rot = tuple(buffer.read_fmt("4f"))
        pos = tuple(buffer.read_fmt("3f"))
        scale = tuple(buffer.read_fmt("3f"))
        pad_ = tuple(buffer.read_fmt("3f"))
        unk_index0, unk_index1, parent_id, child_count, use_in_bounds_calc = buffer.read_fmt("5i")
        oobb = np.frombuffer(buffer.read(64), np.float32).reshape(4, 4)
        importance_factor = buffer.read_float()
        name = buffer.read_sized_string()
        return cls(name, quat, scale_rot, pos, scale, pad_, unk_index0, unk_index1,
                   parent_id,
                   child_count, use_in_bounds_calc, oobb, importance_factor)

    def to_buffer_v1(self, buffer: Buffer):
        buffer.write_fmt("4f", *self.quat)
        buffer.write_fmt("4f", *self.scale_rot)
        buffer.write_fmt("3f", *self.pos)
        buffer.write_fmt("3f", *self.scale)
        buffer.write_fmt("3f", *self.pad_)
        buffer.write_fmt("5i", self.unk_index0, self.unk_index1, self.parent_id,
                         self.child_count, self.use_in_bounds_calc)
        buffer.write(self.oobb.tobytes())
        buffer.write_float(self.importance_factor)
        buffer.write_sized_string(self.name)


@dataclass
class Nodes(ActorChunk):
    roots: list[Node]
    flat_list: list[Node]

    @classmethod
    def from_buffer_v1(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'Nodes':
        node_count, root_node_count = buffer.read_fmt("2I")
        roots = []
        nodes = []
        for i in range(node_count):
            node = Node.from_buffer_v1(buffer)
            node.id = i
            if node.parent_id == -1:
                roots.append(node)
            else:
                parent = nodes[node.parent_id]
                parent.children.append(node)
                node.parent = parent
            nodes.append(node)

        return cls(roots, nodes)

    def to_buffer_v1(self, buffer: Buffer):
        buffer.write_fmt("2I", len(self.flat_list), len(self.roots))
        for node in self.flat_list:
            node.to_buffer_v1(buffer)

    def to_buffer(self, buffer: Buffer):
        chunk_buffer = WritableMemoryBuffer()
        self.to_buffer_v1(chunk_buffer)
        buffer.write_fmt("3I", ActorChunkId.NODES, chunk_buffer.size(), 1)
        buffer.write(chunk_buffer.data)


@dataclass
class Layer:
    amount: float
    uv_offset: tuple[float, ...]
    uv_tiling: tuple[float, ...]
    rotation: float
    material_id: int
    layer_id: LayerId
    name: str

    @classmethod
    def from_buffer_v2(cls, buffer: Buffer) -> 'Layer':
        amount = buffer.read_float()
        uv_offset = buffer.read_fmt("2f")
        uv_tiling = buffer.read_fmt("2f")
        rotation = buffer.read_float()
        material_id, layer_id = buffer.read_fmt("2H")
        name = buffer.read_sized_string()
        return cls(amount, uv_offset, uv_tiling, rotation, material_id, LayerId(layer_id), name)

    def to_buffer_v2(self, buffer: Buffer):
        buffer.write_fmt("6f2H", self.amount, *self.uv_offset, *self.uv_tiling,
                         self.rotation, self.material_id, self.layer_id)
        buffer.write_sized_string(self.name)


@dataclass
class StdMaterial(ActorChunk):
    name: str
    unk1: int
    ambient: tuple[float, ...]
    diffuse: tuple[float, ...]
    specular: tuple[float, ...]
    emissive: tuple[float, ...]
    unk: tuple[float, ...]
    shine: float
    shine_strength: float
    opacity: float
    ior: float
    double_sided: float
    wire_frame: float
    transparency_type: float
    layers: list[Layer]

    @classmethod
    def from_buffer_v2(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'StdMaterial':
        unk1 = buffer.read_uint32()
        unk = buffer.read_fmt("3f")
        ambient = buffer.read_fmt("3f")
        diffuse = buffer.read_fmt("3f")
        specular = buffer.read_fmt("3f")
        emissive = buffer.read_fmt("3f")
        shine, shine_strength, opacity, ior, double_sided, wire_frame, transparency_type, layer_count = buffer.read_fmt(
            "4f2BcB")
        name = buffer.read_sized_string()
        layers = []
        for _ in range(layer_count):
            layers.append(Layer.from_buffer_v2(buffer))
        return cls(name, unk1, ambient, diffuse, specular, emissive, unk, shine, shine_strength, opacity, ior,
                   double_sided, wire_frame, transparency_type, layers)

    def to_buffer_v2(self, buffer: Buffer):
        chunk_buffer = WritableMemoryBuffer()

        chunk_buffer.write_fmt("I15f", self.unk1,
                               *self.unk,
                               *self.ambient,
                               *self.diffuse,
                               *self.specular,
                               *self.emissive
                               )
        chunk_buffer.write_fmt("4f2BcB", self.shine, self.shine_strength, self.opacity, self.ior,
                               self.double_sided, self.wire_frame, self.transparency_type, len(self.layers))
        chunk_buffer.write_sized_string(self.name)

        buffer.write_fmt("3I", ActorChunkId.STDMATERIAL, chunk_buffer.size(), 2)
        buffer.write(chunk_buffer.data)

        for layer in self.layers:
            layer.to_buffer_v2(buffer)

    def to_buffer(self, buffer: Buffer):
        self.to_buffer_v2(buffer)


class VertexAttributeType(IntEnum):
    POSITION = 0
    NORMAL = 1
    TANGENT = 2
    UV = 3
    RGBA8 = 4
    ORIGINAL_VERT_ID = 5
    RGBA32F = 6
    BITANGENT = 7
    CLOTH = 8


@dataclass
class SubMesh:
    vert_count: int
    material_id: int
    indices: np.ndarray
    bone_ids: np.ndarray

    @classmethod
    def from_buffer_v1(cls, buffer: Buffer) -> 'SubMesh':
        index_count, vert_count, material_id, bone_count = buffer.read_fmt("4I")
        indices = np.frombuffer(buffer.read(index_count * 4), np.uint32)
        bone_ids = np.frombuffer(buffer.read(bone_count * 4), np.uint32)
        return cls(vert_count, material_id, indices, bone_ids)

    def to_buffer_v1(self, buffer: Buffer):
        buffer.write_fmt("4I", self.indices.size, self.vert_count, self.material_id, self.bone_ids.size)
        buffer.write(self.indices.tobytes())
        buffer.write(self.bone_ids.tobytes())


@dataclass
class Attribute:
    type: VertexAttributeType
    size: int
    keep_originals: bool
    is_scale_factor: bool
    pad: int
    data: np.ndarray = field(init=False, default=None)

    @classmethod
    def from_buffer_v1(cls, buffer: Buffer) -> 'Attribute':
        a_type, a_size, keep_originals, is_scale_factor, pad = buffer.read_fmt("2I2BH")
        return cls(VertexAttributeType(a_type), a_size, keep_originals == 1, is_scale_factor == 1, pad)

    def to_buffer_v1(self, buffer: Buffer):
        buffer.write_fmt("2I2BH", self.type.value, self.size, self.keep_originals, self.is_scale_factor, self.pad)
        buffer.write(self.data.tobytes())

    def read_data(self, vertex_count: int, buffer: Buffer) -> np.ndarray:
        if self.data is not None:
            return self.data
        a_data = buffer.read(self.size * vertex_count)

        if self.type == VertexAttributeType.POSITION:
            a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 3)
        elif self.type == VertexAttributeType.NORMAL:
            a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 3)
        elif self.type == VertexAttributeType.TANGENT:
            a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 4)
        elif self.type == VertexAttributeType.UV:
            a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 2)
        elif self.type == VertexAttributeType.RGBA8:
            a_data = np.frombuffer(a_data, dtype=np.uint8).reshape(-1, 4)
        elif self.type == VertexAttributeType.RGBA32F:
            a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 4)
        elif self.type == VertexAttributeType.ORIGINAL_VERT_ID:
            a_data = np.frombuffer(a_data, dtype=np.uint32)
        else:
            raise NotImplementedError(self.type)
        self.data = a_data
        return a_data


@dataclass
class Mesh(ActorChunk):
    node_id: int
    infl_ranges: int
    vert_count: int
    index_count: int
    is_collision: int
    pad: tuple[int, ...]

    vertex_attributes: dict[VertexAttributeType, list[Attribute]]
    submeshes: list[SubMesh]

    @classmethod
    def from_buffer_v1(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'Mesh':
        (
            node_id, infl_ranges, vert_count, index_count, sub_mesh_count,
            vertex_attributes_count, is_collision, *pad
        ) = buffer.read_fmt("6i4B")
        attributes: dict[VertexAttributeType, list[Attribute]] = defaultdict(list)
        for _ in range(vertex_attributes_count):
            attribute = Attribute.from_buffer_v1(buffer)
            attribute.read_data(vert_count, buffer)
            attributes[attribute.type].append(attribute)
        submeshes = [SubMesh.from_buffer_v1(buffer) for _ in range(sub_mesh_count)]
        return cls(
            node_id, infl_ranges, vert_count, index_count, is_collision, pad, attributes,
            submeshes
        )

    def to_buffer_v1(self, buffer: Buffer):
        buffer.write_fmt("6i4B", self.node_id, self.infl_ranges, self.vert_count, self.index_count,
                         len(self.submeshes), len(self.vertex_attributes), self.is_collision, 0, 0, 0)
        for attributes in self.vertex_attributes.values():
            for attribute in attributes:
                attribute.to_buffer_v1(buffer)
        for submesh in self.submeshes:
            submesh.to_buffer_v1(buffer)

    def to_buffer(self, buffer: Buffer):
        chunk_buffer = WritableMemoryBuffer()
        self.to_buffer_v1(chunk_buffer)
        buffer.write_fmt("3I", ActorChunkId.MESH, chunk_buffer.size(), 1)
        buffer.write(chunk_buffer.data)


@dataclass
class Deformation:
    node_id: int
    pos_delta: np.ndarray[np.float32]
    normal_delta: np.ndarray[np.float32]
    tangent_delta: np.ndarray[np.float32]
    vertex_ids: np.ndarray[np.uint32]

    @classmethod
    def from_buffer_v1(cls, buffer: Buffer) -> 'Deformation':
        node_id, range_min, range_max, count = buffer.read_fmt("I2fI")
        pos_delta = np.frombuffer(buffer.read(count * 6), np.uint16).reshape(-1, 3)
        normal_delta = np.frombuffer(buffer.read(count * 3), np.uint8).reshape(-1, 3)
        tangent_delta = np.frombuffer(buffer.read(count * 3), np.uint8).reshape(-1, 3)
        vertex_ids = np.frombuffer(buffer.read(count * 4), np.uint32)
        scale = range_max - range_min
        pos_delta = (pos_delta.astype(np.float32) / 65535 * scale) + range_min
        normal_delta = (normal_delta.astype(np.float32) / 127.5 - 1)
        tangent_delta = (tangent_delta.astype(np.float32) / 127.5 - 1)
        return cls(node_id, pos_delta, normal_delta, tangent_delta, vertex_ids)

    def to_buffer_v1(self, buffer: Buffer):
        delta_min = self.pos_delta.min()
        delta_max = self.pos_delta.max()
        buffer.write_fmt("I2fI", self.node_id, delta_min, delta_max, len(self.pos_delta))
        scale = delta_max - delta_min
        buffer.write(np.ceil((self.pos_delta - delta_min) / scale * 65535).astype(np.uint16).tobytes())
        buffer.write(np.ceil((self.normal_delta + 1) * 127.5).astype(np.uint8).tobytes())
        buffer.write(np.ceil((self.tangent_delta + 1) * 127.5).astype(np.uint8).tobytes())
        buffer.write(self.vertex_ids.tobytes())


@dataclass
class MorphTarget:
    name: str
    min_range: float
    max_range: float
    target_lod_level: int
    use_transformations: int
    phoneme_set_bitmask: int
    deformations: list[Deformation]

    @classmethod
    def from_buffer_v1(cls, buffer: Buffer) -> 'MorphTarget':
        (min_range, max_range, target_lod_level, deformation_count,
         use_transformations, phoneme_set_bitmask) = buffer.read_fmt("2f4i")
        name = buffer.read_sized_string()
        deformations = []
        for _ in range(deformation_count):
            deformations.append(Deformation.from_buffer_v1(buffer))
        return cls(name, min_range, max_range, target_lod_level,
                   use_transformations, phoneme_set_bitmask, deformations
                   )

    def to_buffer_v1(self, buffer: Buffer):
        buffer.write_fmt("2f4i", self.min_range, self.max_range,
                         self.target_lod_level, len(self.deformations), self.use_transformations,
                         self.phoneme_set_bitmask)
        buffer.write_sized_string(self.name)
        for deformation in self.deformations:
            deformation.to_buffer_v1(buffer)


@dataclass
class StdMorphTargets(ActorChunk):
    target_lod_index: int
    targets: list[MorphTarget]

    @classmethod
    def from_buffer_v1(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'StdMorphTargets':
        target_count, target_lod_index = buffer.read_fmt("2I")
        targets = []
        for _ in range(target_count):
            targets.append(MorphTarget.from_buffer_v1(buffer))

        return cls(target_lod_index, targets)

    def to_buffer_v1(self, buffer: Buffer):
        buffer.write_fmt("2I", len(self.targets), self.target_lod_index)
        for target in self.targets:
            target.to_buffer_v1(buffer)

    def to_buffer(self, buffer: Buffer):
        chunk_buffer = WritableMemoryBuffer()
        self.to_buffer_v1(chunk_buffer)
        buffer.write_fmt("3I", ActorChunkId.STDPMORPHTARGETS, chunk_buffer.size(), 1)
        buffer.write(chunk_buffer.data)


@dataclass
class SkinningInfo(ActorChunk):
    node_id: int
    local_bone_count: int
    used_in_collision: int
    pad: tuple[int, ...]
    influences: np.ndarray
    groups: np.ndarray

    INFL_DTYPE = np.dtype([
        ("weight", np.float32, (1,)),
        ("bone_id", np.uint32, (1,))
    ])

    @classmethod
    def from_buffer_v3(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'SkinningInfo':
        node_id, local_bone_count, infl_count, used_in_collision, *pad = buffer.read_fmt("3I4B")

        influcences = np.frombuffer(buffer.read(infl_count * 8), cls.INFL_DTYPE)

        mesh_chunk = next(filter(lambda mesh: mesh.node_id == node_id, compiled_file.get_chunks(ActorChunkId.MESH)),
                          None)
        if mesh_chunk is None:
            raise ValueError(f"No mesh chunk with node id({node_id}) found")
        remap = np.frombuffer(buffer.read(mesh_chunk.infl_ranges * 8), np.uint32).reshape(-1, 2)
        return cls(node_id, local_bone_count, used_in_collision, pad, influcences, remap)

    def to_buffer_v3(self, buffer: Buffer):
        buffer.write_fmt("3I4B", self.node_id, self.local_bone_count, len(self.influences), self.used_in_collision, 0,
                         0, 0)
        buffer.write(self.influences.tobytes())
        buffer.write(self.groups.tobytes())

    def to_buffer(self, buffer: Buffer):
        chunk_buffer = WritableMemoryBuffer()
        self.to_buffer_v3(chunk_buffer)
        buffer.write_fmt("3I", ActorChunkId.SKINNINGINFO, chunk_buffer.size(), 3)
        buffer.write(chunk_buffer.data)


@dataclass
class MeshLodLevels(ActorChunk):
    lod_level: int
    lods: list[int]

    @classmethod
    def from_buffer_v1(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'MeshLodLevels':
        lod_level, lod_count = buffer.read_fmt("2I")
        lods = list(buffer.read(lod_count))
        return cls(lod_level, lods)


@dataclass
class MaterialTotal(ActorChunk):
    total_materials: int
    std_material_count: int
    fx_material_count: int

    @classmethod
    def from_buffer_v1(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'MaterialTotal':
        total_materials, std_material_count, fx_material_count = buffer.read_fmt("3I")
        return cls(total_materials, std_material_count, fx_material_count)

    def to_buffer_v1(self, buffer: 'Buffer'):
        buffer.write_fmt("3I", self.total_materials, self.std_material_count, self.fx_material_count)

    def to_buffer(self, buffer: Buffer):
        chunk_buffer = WritableMemoryBuffer()
        self.to_buffer_v1(chunk_buffer)
        buffer.write_fmt("3I", ActorChunkId.MATERIALINFO, chunk_buffer.size(), 1)
        buffer.write(chunk_buffer.data)


CHUNK_READERS: dict[tuple[int, int], Callable[['CompiledFile', Buffer], Any]] = {
    (ActorChunkId.INFO, 2): MetaData.from_buffer_v2,
    (ActorChunkId.NODES, 1): Nodes.from_buffer_v1,
    (ActorChunkId.MATERIALINFO, 1): MaterialTotal.from_buffer_v1,
    (ActorChunkId.STDMATERIAL, 2): StdMaterial.from_buffer_v2,
    (ActorChunkId.MESH, 1): Mesh.from_buffer_v1,
    (ActorChunkId.SKINNINGINFO, 3): SkinningInfo.from_buffer_v3,
    (ActorChunkId.STDPMORPHTARGETS, 1): StdMorphTargets.from_buffer_v1,
    (ActorChunkId.MESHLODLEVELS, 1): MeshLodLevels.from_buffer_v1,
}

ChunkType = TypeVar('ChunkType', bound=ActorChunk)


class CompiledFile:

    def __init__(self):
        self.chunks: dict[ActorChunkId, list[ActorChunk]] = defaultdict(list)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = CompiledHeader.from_buffer(buffer)
        if not cls.is_valid_header(header):
            return None
        return cls._read_file(header, buffer)

    def to_buffer(self, buffer: Buffer):
        CompiledHeader("XAC ", (1, 0), 0, 1).to_buffer(buffer)
        for chunk_id, chunk_group in self.chunks.items():
            for chunk in chunk_group:
                chunk.to_buffer(buffer)

    @classmethod
    def is_valid_header(cls, header: CompiledHeader):
        warnings.warn(f"Implement this method for {cls.__name__}")
        return False

    @classmethod
    def _read_file(cls, header: CompiledHeader, buffer: Buffer) -> 'CompiledFile':
        file = cls()
        while buffer:
            chunk_id, static_chunk_size, chunk_version = buffer.read_fmt("3I")
            actor_chunk_id = ActorChunkId(chunk_id)
            chunk_key = (actor_chunk_id, chunk_version)

            if chunk_key not in CHUNK_READERS:
                raise NotImplementedError(f"Reader for chunk {actor_chunk_id!r} v{chunk_version} not implemented")
            chunk = CHUNK_READERS[chunk_key](file, buffer)
            file.chunks[actor_chunk_id].append(chunk)

        return file

    def get_chunks(self, c_type: ActorChunkId) -> list[ChunkType]:
        return self.chunks[c_type]

    def get_chunk(self, c_type: ActorChunkId) -> ChunkType:
        chunk_group = self.chunks[c_type]
        if len(chunk_group) > 1:
            raise ValueError(f"More than one chunk in {c_type} group")
        return chunk_group[0]

    def add_chunk(self, chunk_type: ActorChunkId, chunk: ChunkType):
        self.chunks[chunk_type].append(chunk)
        return chunk
