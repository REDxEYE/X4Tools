#  Copyright 2025 by REDxEYE.
#  All rights reserved.
import numpy as np

from mesh_data_extractor import process_mesh, MeshRequest, merge_mesh_results, group_indices_by_material
from x4 import StaticModel
from x4.static_model import CompressedBuffer, Attribute, AttributeType, AttributeUsage, MaterialStrip, PrimitiveType

def export_xmf(context):
    selected_objects = context.selected_objects.copy()
    selected_mesh_objects = [ob for ob in selected_objects if ob.type == 'MESH']

    export_normals = True
    export_tangents = True
    export_uvs = True
    export_colors = True
    export_positions = True

    request = MeshRequest(normals=export_normals,
                          tangents=export_tangents,
                          uv_names=export_uvs,
                          colors=export_colors,
                          positions=export_positions
                          )
    meshes_data = [process_mesh(obj, context.evaluated_depsgraph_get(), request) for obj in selected_mesh_objects]
    mesh_data = merge_mesh_results(meshes_data)

    new_indices, material_ranges = group_indices_by_material(mesh_data.indices, mesh_data.material_ids)

    vertex_format = [
        ("position0", np.float32, (3,)),
    ]
    export_vertex_attributes = [
        Attribute(AttributeType.FLOAT3, AttributeUsage.POSITION, 0, 0, 0)
    ]

    if export_normals:
        vertex_format.append(("normal0", np.uint8, (4,)))
        export_vertex_attributes.append(Attribute(AttributeType.D3DCOLOR, AttributeUsage.NORMAL, 0, 0, 0))
    if export_tangents:
        vertex_format.append(("tangent0", np.uint8, (4,)))
        export_vertex_attributes.append(Attribute(AttributeType.D3DCOLOR, AttributeUsage.TANGENT, 0, 0, 0))
    for i in range(len(mesh_data.uvs)):
        vertex_format.append((f"texcoord{i}", np.float32, (2,)))
        export_vertex_attributes.append(Attribute(AttributeType.FLOAT2, AttributeUsage.TEXCOORD, i, 0, 0), )
    for i in range(len(mesh_data.colors)):
        vertex_format.append((f"color{i}", np.uint8, (4,)))
        export_vertex_attributes.append(Attribute(AttributeType.D3DCOLOR, AttributeUsage.COLOR, i, 0, 0), )

    export_vertex_array = np.empty((len(mesh_data.positions)), np.dtype(vertex_format))

    processed_positions = mesh_data.positions.copy()
    processed_positions[:, [2, 1]] = processed_positions[:, [1, 2]]
    export_vertex_array["position0"] = processed_positions
    if export_normals:
        processed_normals = ((mesh_data.normals.copy() + 1) / 2 * 255).astype(np.uint8)
        processed_normals[:, [2, 0, 1]] = processed_normals[:, [0, 1, 2]]
        export_vertex_array["normal0"][:, :3] = processed_normals
    if export_tangents:
        processed_tangents = ((mesh_data.tangents.copy() + 1) / 2 * 255).astype(np.uint8)
        processed_tangents[:, [2, 0, 1]] = processed_tangents[:, [0, 1, 2]]
        export_vertex_array["tangent0"][:, :3] = processed_tangents
    for i, uv in enumerate(mesh_data.uvs):
        uv_copy = uv.copy()
        uv_copy[:, 1] = 1 - uv_copy[:, 1]
        export_vertex_array[f"texcoord{i}"] = uv_copy
    for i, color in enumerate(mesh_data.colors):
        color_copy = (color.copy() * 255).astype(np.uint8)
        export_vertex_array[f"color{i}"] = color_copy

    index_buffer = CompressedBuffer(30, 0, 0, 1, 0, 31, 0, new_indices.size, new_indices.itemsize, 1, 0.0, 0.0, [])
    vertex_buffer = CompressedBuffer(0, 0, 0, 1, 0, 32, 0, export_vertex_array.size, export_vertex_array.itemsize, 1,
                                     0.0, 0.0, export_vertex_attributes)

    index_buffer.data = new_indices.tobytes()
    vertex_buffer.data = export_vertex_array.tobytes()

    strips = []
    offset = 0
    for mat_id, face_start, face_count in material_ranges:
        strips.append(
            MaterialStrip(offset, face_count, mesh_data.materials[mat_id]))
        offset += face_count

    min_coords = np.min(export_vertex_array["position0"], axis=0)
    max_coords = np.max(export_vertex_array["position0"], axis=0)
    bbox_center = (min_coords + max_coords) / 2.0
    bbox_size = max_coords - min_coords

    static_model = StaticModel(3, [vertex_buffer, index_buffer], strips, export_vertex_array.size, new_indices.size * 3,
                               PrimitiveType.TRIANGLELIST, bbox_center, bbox_size)
    return static_model
