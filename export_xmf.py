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


def prepare_mesh(selected_mesh_objects, attributes):
    material_dict = {}
    tmp_mesh = bpy.data.meshes.new('tmp_mesh')
    bm = bmesh.new()
    tmp_bm = bmesh.new()
    for obj in selected_mesh_objects:
        mesh: bpy.types.Mesh = obj.data
        for mat in mesh.materials:
            if mat.name not in material_dict:
                material_dict[mat.name] = len(material_dict)
        tmp_bm.clear()
        tmp_bm.from_mesh(mesh)
        tmp_bm.normal_update()
        tmp_bm.transform(obj.matrix_world)
        bmesh.ops.triangulate(tmp_bm, faces=bm.faces[:])
        tmp_bm.to_mesh(tmp_mesh)
        bm.from_mesh(tmp_mesh)
        bm.normal_update()
    del mesh
    a = bpy.data.objects.new('tmp', tmp_mesh.copy())
    bpy.context.collection.objects.link(a)
    tmp_mesh.clear_geometry()
    tmp_mesh.materials.clear()
    bm.to_mesh(tmp_mesh)
    bm.clear()
    bm.free()
    del bm
    for mat_name in sorted(material_dict.keys(), key=lambda x: material_dict[x]):
        tmp_mesh.materials.append(bpy.data.materials[mat_name])

    material_names = [mat.name for mat in tmp_mesh.materials]

    vertices = []
    vertices_map = {}

    tmp_mesh.calc_tangents(uvmap=tmp_mesh.uv_layers[0].name)
    vertex_items = [
        ("position0", np.float32, (3,)),
        ("normal0", np.uint8, (4,)),
        ("tangent0", np.uint8, (4,)),
    ]
    for i in range(len(tmp_mesh.uv_layers)):
        vertex_items.append((f"texcoord{i}", np.float32, (2,)))
        attributes.append(Attribute(D3DAttributeType.FLOAT2, AttributeUsage.TEXCOORD, i, 0, 0), )
    for i in range(len(tmp_mesh.vertex_colors)):
        vertex_items.append((f"color{i}", np.uint8, (4,)))
        attributes.append(Attribute(D3DAttributeType.D3DCOLOR, AttributeUsage.COLOR, i, 0, 0), )
    vertex_dtype = np.dtype(vertex_items)
    del vertex_items
    vertex_array: np.ndarray = np.zeros(len(tmp_mesh.loops), vertex_dtype)
    indices_array: np.ndarray = np.zeros((len(tmp_mesh.polygons), 4), np.uint32)
    material_ranges = [0]
    curr_material_index = tmp_mesh.polygons[0].material_index
    for face_id, face in enumerate(tmp_mesh.polygons):
        for loop_index, loop_id in enumerate(face.loop_indices):
            loop = tmp_mesh.loops[loop_id]
            bpy_vertex = tmp_mesh.vertices[loop.vertex_index]
            all_uvs = []
            for uv_layer in tmp_mesh.uv_layers:
                all_uvs.append(uv_layer.uv[loop.vertex_index].vector.to_tuple())
            normal = loop.normal
            # normal = tmp_mesh.corner_normals[loop.index].vector
            pos = bpy_vertex.co
            tangent = loop.tangent

            vertex = (
                loop.vertex_index,
                pos.to_tuple(),
                normal.to_tuple(),
                tangent.to_tuple(),
                tuple(all_uvs),
            )
            vertex_index = vertices_map.get(vertex, None)
            if vertex_index is None:
                vertices_map[vertex] = vertex_index = len(vertices)
                vertices.append(vertex)
                vertex_array[vertex_index]["position0"] = pos
                vertex_array[vertex_index]["normal0"][:3] = ((normal + ONE) / 2) * 255 + (ONE / 2)
                vertex_array[vertex_index]["tangent0"][:3] = ((tangent + ONE) / 2) * 255 + (ONE / 2)
                for uv_index, uv in enumerate(all_uvs):
                    vertex_array[vertex_index][f"texcoord{uv_index}"] = uv[0], 1 - uv[1]

                for color_index, color_layer in enumerate(tmp_mesh.vertex_colors):
                    vertex_array[vertex_index][f"color{color_index}"] = np.asarray(
                        color_layer.data[loop.vertex_index].color, np.float32) * 255

            indices_array[face_id, loop_index] = vertex_index
            indices_array[face_id, 3] = face.material_index
        if face.material_index != curr_material_index:
            curr_material_index = face.material_index
            material_ranges.append(face_id)
    material_ranges.append(len(tmp_mesh.polygons))

    bpy.data.meshes.remove(tmp_mesh)
    del tmp_mesh
    vertex_array = vertex_array[:len(vertices)]
    del vertices
    del vertices_map

    return indices_array, material_ranges, vertex_array, material_names
