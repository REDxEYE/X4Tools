#  Copyright 2024 by REDxEYE.
#  All rights reserved.
import bpy
import bmesh
import numpy as np
from mathutils import Vector

from x4 import StaticModel
from x4.static_model import CompressedBuffer, Attribute, D3DAttributeType, AttributeUsage, MaterialStrip, PrimitiveType

ONE = Vector((1, 1, 1))


def export_xmf(context):
    selected_objects = context.selected_objects.copy()
    selected_mesh_objects = [ob for ob in selected_objects if ob.type == 'MESH']

    attributes = [
        Attribute(D3DAttributeType.FLOAT3, AttributeUsage.POSITION, 0, 0, 0),
        Attribute(D3DAttributeType.D3DCOLOR, AttributeUsage.NORMAL, 0, 0, 0),
        Attribute(D3DAttributeType.D3DCOLOR, AttributeUsage.TANGENT, 0, 0, 0),
    ]
    indices_array, material_ranges, vertex_array, material_names = prepare_mesh(selected_mesh_objects, attributes)

    vertex_array["position0"][:, [2, 1]] = vertex_array["position0"][:, [1, 2]]
    vertex_array["normal0"][:, [2, 0, 1]] = vertex_array["normal0"][:, [0, 1, 2]]
    indices_array = indices_array[np.argsort(indices_array[:, 3])][:, :3]
    index_buffer = CompressedBuffer(30, 0, 0, 1, 0, 31, 0, indices_array.size, indices_array.itemsize, 1, 0, 0,
                                    float(vertex_array.size - 1))
    vertex_buffer = CompressedBuffer(0, 0, 0, 1, 0, 32, 0, vertex_array.size, vertex_array.itemsize, 1, 0, 0, 0,
                                     attributes)

    index_buffer.data = indices_array.tobytes()
    vertex_buffer.data = vertex_array.tobytes()

    strips = []
    for i, material_range in enumerate(material_ranges[:-1]):
        strips.append(
            MaterialStrip(material_range * 3, (material_ranges[i + 1] - material_range) * 3, material_names[i]))

    min_coords = np.min(vertex_array["position0"], axis=0)
    max_coords = np.max(vertex_array["position0"], axis=0)
    bbox_center = (min_coords + max_coords) / 2.0
    bbox_size = max_coords - min_coords

    static_model = StaticModel([vertex_buffer, index_buffer], strips, vertex_array.size, indices_array.size * 3,
                               PrimitiveType.TRIANGLELIST, bbox_center, bbox_size)
    return static_model


def pre_process_mesh(source: bpy.types.Object) -> bpy.types.Mesh:
    copied = source.copy()
    copied.modifiers.new("Triangulate", "TRIANGULATE")

    dg = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(copied.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    mesh.name = source.name + "_baked"
    return mesh


def prepare_mesh(selected_mesh_objects, attributes):
    max_uv = 0
    max_vcol = 0
    total_face_count = 0
    total_loops_count = 0
    processed_meshes = [pre_process_mesh(obj) for obj in selected_mesh_objects]

    for mesh in processed_meshes:
        max_uv = max(max_uv, len(mesh.uv_layers))
        max_vcol = max(max_vcol, len(mesh.vertex_colors))
        mesh.calc_tangents(uvmap=mesh.uv_layers[0].name)
        total_face_count += len(mesh.polygons)
        total_loops_count += len(mesh.loops) // 3

    # material_names = [mat.name for mat in tmp_mesh.materials]

    vertices = []
    vertices_map = {}

    vertex_items = [
        ("position0", np.float32, (3,)),
        ("normal0", np.uint8, (4,)),
        ("tangent0", np.uint8, (4,)),
    ]
    for i in range(max_uv):
        vertex_items.append((f"texcoord{i}", np.float32, (2,)))
        attributes.append(Attribute(D3DAttributeType.FLOAT2, AttributeUsage.TEXCOORD, i, 0, 0), )
    for i in range(max_vcol):
        vertex_items.append((f"color{i}", np.uint8, (4,)))
        attributes.append(Attribute(D3DAttributeType.D3DCOLOR, AttributeUsage.COLOR, i, 0, 0), )

    vertex_dtype = np.dtype(vertex_items)
    del vertex_items
    vertex_array: np.ndarray = np.zeros(total_loops_count * 2, vertex_dtype)
    indices_array: np.ndarray = np.zeros((total_loops_count, 4), np.uint32)
    material_ranges = [0]
    curr_material_name = ""

    index_offset = 0
    index_write_offset = 0
    global_material_map = {}
    for mesh in processed_meshes:
        mesh_materials = [mat.name for mat in mesh.materials]
        local_indices_array: np.ndarray = np.zeros((len(mesh.loops) // 3, 4), np.uint32)
        for face_id, face in enumerate(mesh.polygons):
            for loop_index, loop_id in enumerate(face.loop_indices):
                loop = mesh.loops[loop_id]
                bpy_vertex = mesh.vertices[loop.vertex_index]
                all_uvs = []
                for uv_layer in mesh.uv_layers:
                    all_uvs.append(tuple(np.round(uv_layer.uv[loop.vertex_index].vector, 5)))
                normal = loop.normal
                # normal = mesh.vertex_normals[loop.index].vector
                pos = bpy_vertex.co
                tangent = loop.tangent

                vertex = (
                    loop.vertex_index,
                    tuple(np.round(normal, 5)),
                    tuple(all_uvs),
                )
                vertex_index = vertices_map.get(vertex, None)
                if vertex_index is None:
                    vertices_map[vertex] = vertex_index = len(vertices)
                    vertices.append(vertex)
                    vertex_array[vertex_index]["position0"] = pos
                    vertex_array[vertex_index]["normal0"][:3] = np.round(((normal + ONE) / 2) * 255)
                    vertex_array[vertex_index]["tangent0"][:3] = np.round(((tangent + ONE) / 2) * 255)

                    for uv_index, uv_layer in enumerate(mesh.uv_layers):
                        uv = uv_layer.uv[loop.vertex_index].vector
                        vertex_array[vertex_index][f"texcoord{uv_index}"] = uv[0], 1 - uv[1]

                    for color_index, color_layer in enumerate(mesh.vertex_colors):
                        vertex_array[vertex_index][f"color{color_index}"] = np.asarray(
                            color_layer.data[loop.vertex_index].color, np.float32) * 255

                local_indices_array[face_id, loop_index] = vertex_index
                local_indices_array[face_id, 3] = face.material_index
            # if face.material_index != curr_material_index:
            #     curr_material_index = face.material_index
            #     material_ranges.append(face_id)
        indices_array[index_write_offset:index_write_offset + len(local_indices_array)] = local_indices_array
        index_write_offset += len(local_indices_array)
    # material_ranges.append(len(tmp_mesh.polygons))

    # bpy.data.meshes.remove(tmp_mesh)
    # del tmp_mesh
    vertex_array = vertex_array[:len(vertices)]
    del vertices
    del vertices_map
    material_names = []
    return indices_array, material_ranges, vertex_array, material_names
