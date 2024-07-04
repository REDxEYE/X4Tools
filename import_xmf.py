#  Copyright 2024 by REDxEYE.
#  All rights reserved.


import bpy
import numpy as np

from x4.bpy_utils import add_custom_normals, add_uv_layer, add_vertex_color_layer
from x4.material_utils import create_material, add_material


def import_xmf(model_path, sm):
    model_collection = bpy.data.collections.new(model_path.stem)
    bpy.context.scene.collection.children.link(model_collection)
    mesh_data = bpy.data.meshes.new(model_path.stem)
    mesh_obj = bpy.data.objects.new(model_path.stem, mesh_data)
    model_collection.objects.link(mesh_obj)
    vertex_buffer = sm.buffers[0]
    index_buffer = sm.buffers[1]
    vertices = vertex_buffer.get_data()
    indices = np.frombuffer(index_buffer.data, np.uint16).reshape(-1, 3)
    position = vertices["pos0"].copy()
    position[:, [1, 2]] = position[:, [2, 1]]
    mesh_data.from_pydata(position, [], indices)
    normals = (vertices["normal0"][:, :3].astype(np.float32) / 255 * 2) - 1
    normals[:, [0, 1, 2]] = normals[:, [2, 0, 1]]
    add_custom_normals(normals, mesh_data)
    for i in range(8):
        uv_name = f"uv{i}"
        if uv_name in vertices.dtype.names:
            add_uv_layer(uv_name, vertices[uv_name], mesh_data)
    for i in range(8):
        color_name = f"color{i}"
        if color_name in vertices.dtype.names:
            add_vertex_color_layer(color_name, vertices[color_name], mesh_data)
    material_indices = np.zeros(sm.index_count // 3, np.uint32)
    for strip in sm.strips:
        mat = create_material(strip.name)

        material_indices[strip.start_index // 3:strip.start_index // 3 + strip.count // 3] = add_material(mat,
                                                                                                          mesh_obj)
    mesh_data.polygons.foreach_set('material_index', material_indices)
