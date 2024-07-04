import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Any, Optional

import numpy as np

from x4.file_utils import Buffer


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
    unk0: int
    unk1: int

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        ident = buffer.read_fourcc()
        version_maj, version_min, unk0, unk1 = buffer.read_fmt("4B")
        return cls(ident, (version_maj, version_min), unk0, unk1)


@dataclass
class MetaData:
    lod_count: int
    trajectory_node_index: int
    retarget_root_offset: int
    unit_type: int
    exporter_high_version: int
    exporter_low_version: int
    exporter: str
    source_file: str
    export_date: str
    actor_name: str

    @classmethod
    def from_buffer_v2(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'MetaData':
        (lod_count,
         motion_extraction_node_index,
         retarget_root_offset, unit_type,
         exporter_high_version, exporter_low_version, pad) = buffer.read_fmt("3i4b")
        exporter = buffer.read_sized_string()
        source_file = buffer.read_sized_string()
        export_date = buffer.read_sized_string()
        actor_name = buffer.read_sized_string()
        return cls(lod_count, motion_extraction_node_index, retarget_root_offset,
                   unit_type, exporter_high_version, exporter_low_version,
                   exporter, source_file, export_date, actor_name)


@dataclass
class Node:
    name: str
    quat: tuple[float, ...]
    scale_rot: tuple[float, ...]
    pos: tuple[float, ...]
    scale: tuple[float, ...]
    field_38: tuple[float, ...]
    field_44: int
    field_48: int
    parent_id: int
    child_count: int
    field_54: int
    oobb: np.ndarray[np.float32]
    field_98: int

    id: int = field(init=False, default=0)
    children: list['Node'] = field(init=False, default_factory=list)
    parent: Optional['Node'] = field(init=False, default=None)

    @classmethod
    def from_buffer_v1(cls, buffer: Buffer) -> 'Node':
        quat = tuple(buffer.read_fmt("4f"))
        scale_rot = tuple(buffer.read_fmt("4f"))
        pos = tuple(buffer.read_fmt("3f"))
        scale = tuple(buffer.read_fmt("3f"))
        field_38 = tuple(buffer.read_fmt("3f"))
        field_44, field_48, parent_id, child_coint, field_54 = buffer.read_fmt("5i")
        oobb = np.frombuffer(buffer.read(64), np.float32).reshape(4, 4)
        field_98 = buffer.read_float()
        name = buffer.read_sized_string()
        return cls(name, quat, scale_rot, pos, scale, field_38, field_44, field_48,
                   parent_id,
                   child_coint, field_54, oobb, field_98)


@dataclass
class Nodes:
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


@dataclass
class Texture:
    unk0: tuple[float, ...]
    unk1: tuple[float, ...]
    unk2: int
    layer_id: LayerId
    name: str

    @classmethod
    def from_buffer_v2(cls, buffer: Buffer) -> 'Texture':
        unk0 = buffer.read_fmt("3f")
        unk1 = buffer.read_fmt("3f")
        unk2, layerid = buffer.read_fmt("2H")
        name = buffer.read_sized_string()
        return cls(unk0, unk1, unk2, LayerId(layerid), name)


@dataclass
class StdMaterial:
    name: str
    lod_id: int
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
    textures: list[Texture]

    @classmethod
    def from_buffer_v2(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'StdMaterial':
        lod_id = buffer.read_uint32()
        unk = buffer.read_fmt("3f")
        ambient = buffer.read_fmt("3f")
        diffuse = buffer.read_fmt("3f")
        specular = buffer.read_fmt("3f")
        emissive = buffer.read_fmt("3f")
        shine, shine_strength, opacity, ior, double_sided, wire_frame, transparency_type, layer_count = buffer.read_fmt(
            "4f2BcB")
        shader = buffer.read_sized_string()
        sub_materials = []
        for _ in range(layer_count):
            sub_materials.append(Texture.from_buffer_v2(buffer))
        return cls(shader, lod_id, ambient, diffuse, specular, emissive, unk, shine, shine_strength, opacity, ior,
                   double_sided, wire_frame, transparency_type, sub_materials)


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
    indices: np.ndarray[np.uint32]
    bone_ids: np.ndarray[np.uint32]

    @classmethod
    def from_buffer_v1(cls, buffer: Buffer) -> 'SubMesh':
        index_count, vert_count, material_id, bone_count = buffer.read_fmt("4I")
        indices = np.frombuffer(buffer.read(index_count * 4), np.uint32)
        bone_ids = np.frombuffer(buffer.read(bone_count * 4), np.uint32)
        return cls(vert_count, material_id, indices, bone_ids)


@dataclass
class Attribute:
    flags: int
    unk: int
    data: np.ndarray


@dataclass
class Mesh:
    node_id: int
    infl_ranges: int
    vert_count: int
    index_count: int
    sub_mesh_count: int
    is_collision: int
    u1: int
    u2: int
    u3: int

    vertex_attributes: dict[VertexAttributeType, list[[tuple[int, int, np.ndarray]]]]
    submeshes: list[SubMesh]

    @classmethod
    def from_buffer_v1(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'Mesh':
        (
            node_id, infl_ranges, vert_count, index_count, sub_mesh_count,
            vertex_attributes_count, is_collision, u1, u2, u3
        ) = buffer.read_fmt("6i4B")
        attributes = defaultdict(list)
        for _ in range(vertex_attributes_count):
            a_type, a_size, a_flags, a_unk = buffer.read_fmt("2I2H")
            a_data = buffer.read(a_size * vert_count)

            if a_type == VertexAttributeType.POSITION:
                a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 3)
            elif a_type == VertexAttributeType.NORMAL:
                a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 3)
            elif a_type == VertexAttributeType.TANGENT:
                a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 4)
            elif a_type == VertexAttributeType.UV:
                a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 2)
            elif a_type == VertexAttributeType.RGBA8:
                a_data = np.frombuffer(a_data, dtype=np.uint8).reshape(-1, 4)
            elif a_type == VertexAttributeType.RGBA32F:
                a_data = np.frombuffer(a_data, dtype=np.float32).reshape(-1, 4)
            elif a_type == VertexAttributeType.ORIGINAL_VERT_ID:
                a_data = np.frombuffer(a_data, dtype=np.uint32)
            else:
                raise NotImplementedError(a_type)
            attributes[VertexAttributeType(a_type)].append(Attribute(a_flags, a_unk, a_data))
        submeshes = [SubMesh.from_buffer_v1(buffer) for _ in range(sub_mesh_count)]
        return cls(
            node_id, infl_ranges, vert_count, index_count, sub_mesh_count, is_collision, u1, u2, u3, attributes,
            submeshes
        )


@dataclass
class Deformation:
    node_id: int
    pos_delta: np.ndarray[np.float32]
    normal_delta: np.ndarray[np.float32]
    tangent_delta: np.ndarray[np.float32]
    vertex_ids: np.ndarray[np.uint32]

    @classmethod
    def from_buffer_v3(cls, buffer: Buffer) -> 'Deformation':
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


@dataclass
class MorphTarget:
    name: str
    field_0: int
    field_4: float
    target_lod_level: int
    deformation_count: int
    use_transformations: int
    field_14: int
    deformations: list[Deformation]

    @classmethod
    def from_buffer_v3(cls, buffer: Buffer) -> 'MorphTarget':
        (field_0, field_4, target_lod_level, deformation_count,
         use_transformations, field_14) = buffer.read_fmt("if4i")
        name = buffer.read_sized_string()
        deformations = []
        for _ in range(deformation_count):
            deformations.append(Deformation.from_buffer_v3(buffer))
        return cls(name, field_0, field_4, target_lod_level, deformation_count,
                   use_transformations, field_14, deformations
                   )


class StdMorphTargets(list[MorphTarget]):
    @classmethod
    def from_buffer_v1(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'StdMorphTargets':
        target_count, target_lod_index = buffer.read_fmt("2I")
        self = cls()
        for _ in range(target_count):
            self.append(MorphTarget.from_buffer_v3(buffer))

        return self


@dataclass
class SkinningInfo:
    node_id: int
    local_bone_count: int
    field_C: int
    field_D: int
    field_E: int
    field_F: int
    influences: np.ndarray[np.float32]
    groups: np.ndarray[np.uint8]

    INFL_DTYPE = np.dtype([
        ("weight", np.float32, (1,)),
        ("bone_id", np.uint32, (1,))
    ])

    @classmethod
    def from_buffer_v3(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'SkinningInfo':
        node_id, local_bone_count, infl_count, field_C, field_D, field_E, field_F = buffer.read_fmt("3I4B")

        influcences = np.frombuffer(buffer.read(infl_count * 8), cls.INFL_DTYPE)

        mesh_chunk = next(filter(lambda mesh: mesh.node_id == node_id, compiled_file.get_chunks(1)), None)
        if mesh_chunk is None:
            raise ValueError(f"No mesh chunk with node id({node_id}) found")
        remap = np.frombuffer(buffer.read(mesh_chunk.infl_ranges * 8), np.uint32).reshape(-1, 2)
        return cls(node_id, local_bone_count, field_C, field_D, field_E, field_F, influcences, remap)


@dataclass
class MeshLodLevels:
    lod_level: int
    lods: list[int]

    @classmethod
    def from_buffer_v1(cls, compiled_file: 'CompiledFile', buffer: Buffer) -> 'MeshLodLevels':
        lod_level, lod_count = buffer.read_fmt("2I")
        lods = list(buffer.read(lod_count))
        return cls(lod_level, lods)


CHUNK_READERS: dict[tuple[int, int], Callable[['CompiledFile', Buffer], Any]] = {
    (ActorChunkId.INFO, 2): MetaData.from_buffer_v2,
    (ActorChunkId.NODES, 1): Nodes.from_buffer_v1,
    (ActorChunkId.MATERIALINFO, 1): lambda _, buffer: buffer.read_fmt("3I"),
    (ActorChunkId.STDMATERIAL, 2): StdMaterial.from_buffer_v2,
    (ActorChunkId.MESH, 1): Mesh.from_buffer_v1,
    (ActorChunkId.SKINNINGINFO, 3): SkinningInfo.from_buffer_v3,
    (ActorChunkId.STDPMORPHTARGETS, 1): StdMorphTargets.from_buffer_v1,
    (ActorChunkId.MESHLODLEVELS, 1): MeshLodLevels.from_buffer_v1,
}


class CompiledFile:

    def __init__(self):
        self.chunks: dict[int, list[Any]] = defaultdict(list)

    @classmethod
    def from_buffer(cls, buffer: Buffer):
        header = CompiledHeader.from_buffer(buffer)
        if not cls.is_valid_header(header):
            return None
        return cls._read_file(header, buffer)

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

    def get_chunks(self, c_type: ActorChunkId) -> list[Any]:
        return self.chunks[c_type]

    def get_chunk(self, c_type: ActorChunkId) -> Any:
        chunk_group = self.chunks[c_type]
        if len(chunk_group) > 1:
            raise ValueError(f"More than one chunk in {c_type} group")
        return chunk_group[0]
