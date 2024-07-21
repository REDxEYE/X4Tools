#  Copyright 2024 by REDxEYE.
#  All rights reserved.

from collections import defaultdict
from enum import Enum
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector, Matrix

from x4.bpy_utils import add_custom_normals, add_uv_layer
from x4.compiled_file import Node, Mesh, Nodes, ActorChunkId, SkinningInfo, StdMorphTargets, StdMaterial, \
    VertexAttributeType
from x4.material_utils import create_material, add_material


def build_matrix(node: Node):
    quat = Quaternion([node.quat[3], node.quat[0], -node.quat[2], node.quat[1]])
    pos = Vector((node.pos[0], -node.pos[2], node.pos[1]))
    scale = Vector((node.scale[0], node.scale[2], node.scale[1]))
    return Matrix.LocRotScale(pos, quat, scale)


def mark_nodes(influences, meshes, roots):
    node_types: dict[int, NodeType] = defaultdict(lambda: NodeType.Node)
    node_types[0] = NodeType.Bone
    node_to_mesh: dict[int, Mesh] = {}
    for mesh in meshes:
        node_to_mesh[mesh.node_id] = mesh
        node_types[mesh.node_id] = NodeType.Mesh
        for submesh in mesh.submeshes:
            for bone in submesh.bone_ids:
                node_types[bone] = NodeType.Bone
    for influence in influences:
        node_types[influence.node_id] = NodeType.Mesh
        for bone in np.unique(influence.influences["bone_id"]):
            node_types[bone] = NodeType.Bone
    flood_fill_bones(node_types, roots)
    return node_to_mesh, node_types


class NodeType(Enum):
    Node = "Node"
    Bone = "Bone"
    Mesh = "Mesh"


def flood_fill_bones(node_types: dict[int, NodeType], nodes: list[Node]):
    def flood_fill_down(node: Node):
        while node.parent is not None:
            node_types[node.id] = NodeType.Bone
            node = node.parent
        node_types[node.id] = NodeType.Bone

    def flood_fill_up(node: Node):
        for child in node.children:
            node_types[child.id] = NodeType.Bone
            flood_fill_up(child)

    def visit(node: Node):
        if node_types.get(node.id, NodeType.Node) == NodeType.Bone:
            flood_fill_up(node)
            flood_fill_down(node)
        for child in node.children:
            visit(child)

    for root in nodes:
        visit(root)


def create_skeleton(model_collection, name, node_types, nodes):
    armature = bpy.data.armatures.new(f"{name}_ARM_DATA")
    armature_obj = bpy.data.objects.new(f"{name}_ARM", armature)
    armature_obj.show_in_front = True

    model_collection.objects.link(armature_obj)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    matrices = {}

    def create_bone(node: Node, parent: bpy.types.EditBone | None):
        bl_bone = armature.edit_bones.new(node.name)
        matrix = build_matrix(node)
        if parent is not None:
            bl_bone.parent = parent
            matrix = matrices[node.parent.name] @ matrix
        bl_bone.matrix = matrix
        # bl_bone.tail = bl_bone.head + Vector((0, 0, 3))
        bl_bone.tail = bl_bone.head + (matrix.to_3x3() @ Vector((5, 0, 0)))

        matrices[node.name] = matrix

        for child in node.children:
            create_bone(child, bl_bone)

    def pose_bone(node: Node):
        bone = armature_obj.pose.bones[node.name]
        bone.matrix = build_matrix(node)
        # bone.matrix = matrices[node.name]

        for child in node.children:
            pose_bone(child)

    for root in nodes:
        if node_types[root.id] != NodeType.Bone:
            continue
        create_bone(root, None)
    # bpy.ops.object.mode_set(mode='POSE')
    # for root in nodes:
    #     if node_types[root.id] != NodeType.Bone:
    #         continue
    #     pose_bone(root)
    # bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj


def import_actor(cf, path):
    nodes_chunk: Nodes = cf.get_chunk(ActorChunkId.NODES)
    nodes = nodes_chunk.flat_list
    roots = nodes_chunk.roots
    meshes: list[Mesh] = cf.get_chunks(ActorChunkId.MESH)
    influences: list[SkinningInfo] = cf.get_chunks(ActorChunkId.SKINNINGINFO)
    morph_targets: list[StdMorphTargets] = cf.get_chunks(ActorChunkId.STDPMORPHTARGETS)
    materials: list[StdMaterial] = cf.get_chunks(ActorChunkId.STDMATERIAL)

    node_to_mesh, node_types = mark_nodes(influences, meshes, roots)

    matrices = {}
    objects = {}

    name = Path(path).stem
    model_collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(model_collection)

    arm_obj = create_skeleton(model_collection, name, node_types, roots)
    arm_name = arm_obj.name

    def process_mesh(node: Node):
        mesh = node_to_mesh[node.id]
        mesh_node = nodes[mesh.node_id]
        mesh_data = bpy.data.meshes.new(mesh_node.name)
        mesh_obj = bpy.data.objects.new(mesh_node.name, mesh_data)
        model_collection.objects.link(mesh_obj)
        indices = np.zeros(mesh.index_count, np.uint32)
        index_offset = 0
        vertex_offset = 0
        material_ids = np.zeros(mesh.index_count // 3, np.uint32)
        used_nodes: set[str] = set()
        for submesh in mesh.submeshes:
            indices[index_offset:index_offset + submesh.indices.shape[0]] = submesh.indices + vertex_offset
            material_ids[index_offset // 3:index_offset // 3 + submesh.indices.shape[0] // 3] = submesh.material_id
            index_offset += submesh.indices.shape[0]
            vertex_offset += submesh.vert_count
            for bone_id in np.unique(submesh.bone_ids):
                used_nodes.add(nodes[bone_id].name)
        position = mesh.vertex_attributes[VertexAttributeType.POSITION][0].data.copy()
        # position[:, [1, 2]] = position[:, [2, 1]]
        position[:, 1] *= -1
        position[:, 2] *= -1
        mesh_data.from_pydata(position, [], indices.reshape(-1, 3))
        used_materials = np.unique(material_ids)
        for mat_id in used_materials:
            mat_info = materials[mat_id]
            mat = create_material(mat_info.name)
            add_material(mat, mesh_obj)

        indices = np.searchsorted(used_materials, material_ids)
        mesh_data.polygons.foreach_set('material_index', indices)
        normal = mesh.vertex_attributes[VertexAttributeType.NORMAL][0].data.copy()
        # normal[:, [1, 2]] = normal[:, [2, 1]]
        normal[:, 1] *= -1
        normal[:, 2] *= -1
        # normal[:, 0] *= -1
        add_custom_normals(normal, mesh_data)

        for i, uv in enumerate(mesh.vertex_attributes[VertexAttributeType.UV]):
            uv = uv.data.copy()
            add_uv_layer(f"{i}UV", uv, mesh_data)

        influence: SkinningInfo = next(filter(lambda a: a.node_id == mesh.node_id, influences), None)
        if influence is not None:
            mod = mesh_obj.modifiers.new('Armature', type='ARMATURE')
            mod.object = arm_obj
            mesh_obj.parent = arm_obj

            ovid = mesh.vertex_attributes[VertexAttributeType.ORIGINAL_VERT_ID][0].data

            weight_groups = {name: mesh_obj.vertex_groups.new(name=name) for name in used_nodes}
            vertex_offset = 0
            for submesh in mesh.submeshes:
                original_ids = ovid[vertex_offset:vertex_offset + submesh.vert_count]
                for v_id, infl_id in enumerate(original_ids):
                    offset, bone_count = influence.groups[infl_id]
                    weight_and_bone = influence.influences[offset:offset + bone_count]
                    for (weight,), (bone_ids,) in weight_and_bone:
                        if weight > 0:
                            weight_groups[nodes[bone_ids].name].add([v_id + vertex_offset], weight, 'REPLACE')
                vertex_offset += submesh.vert_count
            if morph_targets:
                mesh_obj.shape_key_add(name='base')
                for morph_target in morph_targets:
                    for morph in morph_target.targets:
                        for deform in morph.deformations:
                            if deform.node_id != mesh.node_id:
                                continue
                            deform_vertices = position.copy()
                            delta = deform.pos_delta.copy()
                            delta[:, [1, 2]] = delta[:, [2, 1]]
                            delta[:, 1] *= -1
                            deform_vertices[deform.vertex_ids] += delta
                            shape_key = mesh_obj.shape_key_add(name=morph.name)
                            shape_key.data.foreach_set("co", deform_vertices.ravel())
        return mesh_obj

    def process_node(node: Node, parent: bpy.types.Object | None):
        if node_types[node.id] == NodeType.Mesh:
            obj = process_mesh(node)
        else:
            obj = bpy.data.objects.new(node.name, None)
            model_collection.objects.link(obj)
        objects[node.id] = obj
        matrix = build_matrix(node)

        if node.parent_id != -1:
            obj.parent = parent
            matrix = matrices[node.parent_id] @ matrix
        if node_types[node.id] != NodeType.Mesh:
            obj.matrix_basis = matrix
        matrices[node.id] = matrix

        for child in node.children:
            process_node(child, obj)

    for root in roots:
        if node_types[root.id] == NodeType.Bone:
            continue
        process_node(root, None)
    return bpy.data.objects[arm_name]
