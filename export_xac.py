#  Copyright 2024 by REDxEYE.
#  All rights reserved.
import math
from datetime import datetime

import bpy.types
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler

from x4 import ActorFile
from x4.compiled_file import MetaData, ActorChunkId, Nodes, Node
from x4.version import version


def get_parent_collection_names(collection):
    if collection.name in bpy.context.scene.collection.children.keys():
        return collection
    for parent_collection in bpy.data.collections:
        if collection.name in parent_collection.children.keys():
            if parent_collection.name in bpy.context.scene.collection.children.keys():
                return parent_collection
            return get_parent_collection_names(parent_collection)
    return bpy.context.scene.collection


R_x = Matrix([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])
R_y = Matrix([
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

# Combine the rotations
R_combined = R_x @ R_y


def rotate_all_bones_keep_children(armature):
    # Get the armature object
    armature_data = armature.data.copy()
    armature = armature.copy()
    armature.data = armature_data
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    # Iterate through all bones in the armature
    for bone_name in armature.data.edit_bones.keys():
        edit_bone = armature.data.edit_bones[bone_name]

        # Store the world matrix of each child bone
        child_world_matrices = {child.name: armature.matrix_world @ armature.data.bones[child.name].matrix_local.copy()
                                for child in edit_bone.children}

        # Rotate the bone's head and tail using the combined rotation matrix
        edit_bone.transform(R_combined)

        # Reapply the world matrix to each child bone to keep them in the same place
        for child_name, world_matrix in child_world_matrices.items():
            child_bone = armature.data.edit_bones[child_name]
            child_bone.matrix = armature.matrix_world.inverted() @ world_matrix

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature


def compute_bone_rotation(bone):
    direction = (bone.tail - bone.head).normalized()  # Direction from head to tail, normalized
    up = Vector((0, 0, 1))

    roll_matrix = Matrix.Rotation(bone.roll, 4, direction)

    u = up.cross(direction).normalized()
    if u.magnitude == 0:
        u = direction.orthogonal().normalized()
    v = direction.cross(u)

    rotation_matrix = Matrix([u, v, direction]).transposed()
    rotation_matrix = roll_matrix @ rotation_matrix  # Apply roll

    return rotation_matrix

def export_actor(context):
    object = context.active_object
    if object.type != "ARMATURE":
        raise Exception("Object is not an ARMATURE object")
    actor = ActorFile()
    top_collection = get_parent_collection_names(object.users_collection[0])
    info_chunk = MetaData(0, -1, version[0], version[1], 0.0, "Blender X4 Tools", bpy.data.filepath,
                          datetime.now().strftime("%B %d %Y"), top_collection.name or "")
    actor.add_chunk(ActorChunkId.INFO, info_chunk)
    roots = []
    all_nodes = []
    bone_ids = {}
    matrices = {}

    object.select_set(True)
    bpy.context.view_layer.objects.active = object
    bpy.ops.object.mode_set(mode='EDIT')

    for bone in object.data.edit_bones:
        matrix = bone.matrix
        pos, quat, scl = matrix.decompose()
        quat = quat @ Euler((0, 0, -math.radians(90))).to_quaternion()
        quat = quat @ Euler((bone.roll, 0, 0)).to_quaternion()
        matrices[bone.name] = Matrix.LocRotScale(pos, quat, scl)

    bpy.ops.object.mode_set(mode='OBJECT')

    for bone in object.data.bones:
        bone: bpy.types.Bone
        if bone.parent:
            matrix = matrices[bone.parent.name].inverted() @ matrices[bone.name]
        else:
            matrix = matrices[bone.name]

        matrix = Euler((math.pi, 0, 0)).to_matrix().to_4x4() @ matrix
        pos, quat, scl = matrix.decompose()
        quat: Quaternion
        quat = Quaternion((quat.x, quat.z, -quat.y, quat.w))
        quat.rotate(Euler((0, 0, math.pi)))
        node = Node(bone.name,
                    quat,
                    (0.0, 0.0, 0.0, 1.0),
                    (-pos[0], -pos[2], pos[1]),
                    (scl[0], scl[2], scl[1]),
                    (-1.0, -1.0, -1.0),
                    -1, -1, bone_ids[bone.parent.name] if bone.parent else -1,
                    len(bone.children),
                    True,
                    np.asarray(Matrix.Identity(4), np.float32),
                    1.0
                    )

        bone_ids[node.name] = len(all_nodes)
        if bone.parent is None:
            roots.append(node)
        all_nodes.append(node)
        print(bone)

    for obj in top_collection.all_objects:
        if obj == object:
            continue
        print(obj)
    actor.add_chunk(ActorChunkId.NODES, Nodes(roots, all_nodes))
    return actor
