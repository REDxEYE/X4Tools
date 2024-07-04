import random
from pathlib import Path
from typing import cast

import bpy


def create_material(mat_name: str):
    mat = bpy.data.materials.get(mat_name, None)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
        mat.diffuse_color = [random.uniform(.4, 1) for _ in range(3)] + [1.0]
    return mat


def add_material(mat: bpy.types.Material, model_obj: bpy.types.Object):
    model_data: bpy.types.Mesh = cast(bpy.types.Mesh, model_obj.data)
    if model_data.materials.get(mat.name, None) is not None:
        return list(model_data.materials).index(mat)
    else:
        model_data.materials.append(mat)
        return len(model_data.materials) - 1


class Nodes:
    ShaderNodeAddShader = 'ShaderNodeAddShader'
    ShaderNodeAmbientOcclusion = 'ShaderNodeAmbientOcclusion'
    ShaderNodeAttribute = 'ShaderNodeAttribute'
    ShaderNodeBackground = 'ShaderNodeBackground'
    ShaderNodeBevel = 'ShaderNodeBevel'
    ShaderNodeBlackbody = 'ShaderNodeBlackbody'
    ShaderNodeBrightContrast = 'ShaderNodeBrightContrast'
    ShaderNodeBsdfAnisotropic = 'ShaderNodeBsdfAnisotropic'
    ShaderNodeBsdfDiffuse = 'ShaderNodeBsdfDiffuse'
    ShaderNodeBsdfGlass = 'ShaderNodeBsdfGlass'
    ShaderNodeBsdfGlossy = 'ShaderNodeBsdfGlossy'
    ShaderNodeBsdfHair = 'ShaderNodeBsdfHair'
    ShaderNodeBsdfHairPrincipled = 'ShaderNodeBsdfHairPrincipled'
    ShaderNodeBsdfPrincipled = 'ShaderNodeBsdfPrincipled'
    ShaderNodeBsdfRefraction = 'ShaderNodeBsdfRefraction'
    ShaderNodeBsdfToon = 'ShaderNodeBsdfToon'
    ShaderNodeBsdfTranslucent = 'ShaderNodeBsdfTranslucent'
    ShaderNodeBsdfTransparent = 'ShaderNodeBsdfTransparent'
    ShaderNodeBsdfVelvet = 'ShaderNodeBsdfVelvet'
    ShaderNodeBump = 'ShaderNodeBump'
    ShaderNodeCameraData = 'ShaderNodeCameraData'
    ShaderNodeClamp = 'ShaderNodeClamp'
    ShaderNodeCombineHSV = 'ShaderNodeCombineHSV'
    ShaderNodeCombineRGB = 'ShaderNodeCombineRGB'
    ShaderNodeCombineXYZ = 'ShaderNodeCombineXYZ'
    ShaderNodeCustomGroup = 'ShaderNodeCustomGroup'
    ShaderNodeDisplacement = 'ShaderNodeDisplacement'
    ShaderNodeEeveeSpecular = 'ShaderNodeEeveeSpecular'
    ShaderNodeEmission = 'ShaderNodeEmission'
    ShaderNodeFresnel = 'ShaderNodeFresnel'
    ShaderNodeGamma = 'ShaderNodeGamma'
    ShaderNodeGroup = 'ShaderNodeGroup'
    ShaderNodeHairInfo = 'ShaderNodeHairInfo'
    ShaderNodeHoldout = 'ShaderNodeHoldout'
    ShaderNodeHueSaturation = 'ShaderNodeHueSaturation'
    ShaderNodeInvert = 'ShaderNodeInvert'
    ShaderNodeLayerWeight = 'ShaderNodeLayerWeight'
    ShaderNodeLightFalloff = 'ShaderNodeLightFalloff'
    ShaderNodeLightPath = 'ShaderNodeLightPath'
    ShaderNodeMapRange = 'ShaderNodeMapRange'
    ShaderNodeMapping = 'ShaderNodeMapping'
    ShaderNodeMath = 'ShaderNodeMath'
    ShaderNodeMixRGB = 'ShaderNodeMixRGB'
    ShaderNodeMix = 'ShaderNodeMix'
    ShaderNodeMixShader = 'ShaderNodeMixShader'
    ShaderNodeNewGeometry = 'ShaderNodeNewGeometry'
    ShaderNodeNormal = 'ShaderNodeNormal'
    ShaderNodeNormalMap = 'ShaderNodeNormalMap'
    ShaderNodeObjectInfo = 'ShaderNodeObjectInfo'
    ShaderNodeOutputAOV = 'ShaderNodeOutputAOV'
    ShaderNodeOutputLight = 'ShaderNodeOutputLight'
    ShaderNodeOutputLineStyle = 'ShaderNodeOutputLineStyle'
    ShaderNodeOutputMaterial = 'ShaderNodeOutputMaterial'
    ShaderNodeOutputWorld = 'ShaderNodeOutputWorld'
    ShaderNodeParticleInfo = 'ShaderNodeParticleInfo'
    ShaderNodeRGB = 'ShaderNodeRGB'
    ShaderNodeRGBCurve = 'ShaderNodeRGBCurve'
    ShaderNodeRGBToBW = 'ShaderNodeRGBToBW'
    ShaderNodeScript = 'ShaderNodeScript'
    ShaderNodeSeparateHSV = 'ShaderNodeSeparateHSV'
    ShaderNodeSeparateRGB = 'ShaderNodeSeparateRGB'
    ShaderNodeSeparateXYZ = 'ShaderNodeSeparateXYZ'
    ShaderNodeShaderToRGB = 'ShaderNodeShaderToRGB'
    ShaderNodeSqueeze = 'ShaderNodeSqueeze'
    ShaderNodeSubsurfaceScattering = 'ShaderNodeSubsurfaceScattering'
    ShaderNodeTangent = 'ShaderNodeTangent'
    ShaderNodeTexBrick = 'ShaderNodeTexBrick'
    ShaderNodeTexChecker = 'ShaderNodeTexChecker'
    ShaderNodeTexCoord = 'ShaderNodeTexCoord'
    ShaderNodeTexEnvironment = 'ShaderNodeTexEnvironment'
    ShaderNodeTexGradient = 'ShaderNodeTexGradient'
    ShaderNodeTexIES = 'ShaderNodeTexIES'
    ShaderNodeTexImage = 'ShaderNodeTexImage'
    ShaderNodeTexMagic = 'ShaderNodeTexMagic'
    ShaderNodeTexMusgrave = 'ShaderNodeTexMusgrave'
    ShaderNodeTexNoise = 'ShaderNodeTexNoise'
    ShaderNodeTexPointDensity = 'ShaderNodeTexPointDensity'
    ShaderNodeTexSky = 'ShaderNodeTexSky'
    ShaderNodeTexVoronoi = 'ShaderNodeTexVoronoi'
    ShaderNodeTexWave = 'ShaderNodeTexWave'
    ShaderNodeTexWhiteNoise = 'ShaderNodeTexWhiteNoise'
    ShaderNodeUVAlongStroke = 'ShaderNodeUVAlongStroke'
    ShaderNodeUVMap = 'ShaderNodeUVMap'
    ShaderNodeValToRGB = 'ShaderNodeValToRGB'
    ShaderNodeValue = 'ShaderNodeValue'
    ShaderNodeVectorCurve = 'ShaderNodeVectorCurve'
    ShaderNodeVectorDisplacement = 'ShaderNodeVectorDisplacement'
    ShaderNodeVectorMath = 'ShaderNodeVectorMath'
    ShaderNodeVectorRotate = 'ShaderNodeVectorRotate'
    ShaderNodeVectorTransform = 'ShaderNodeVectorTransform'
    ShaderNodeVertexColor = 'ShaderNodeVertexColor'
    ShaderNodeVolumeAbsorption = 'ShaderNodeVolumeAbsorption'
    ShaderNodeVolumeInfo = 'ShaderNodeVolumeInfo'
    ShaderNodeVolumePrincipled = 'ShaderNodeVolumePrincipled'
    ShaderNodeVolumeScatter = 'ShaderNodeVolumeScatter'
    ShaderNodeWavelength = 'ShaderNodeWavelength'
    ShaderNodeWireframe = 'ShaderNodeWireframe'
    ShaderNodeReroute = 'NodeReroute'
    ShaderNodeFrame = 'NodeFrame'


def load_image_from_path(texture_path: Path):
    return bpy.data.images.load(texture_path.as_posix())


def clear_nodes(material: bpy.types.Material):
    for node in material.node_tree.nodes:
        material.node_tree.nodes.remove(node)


def create_node(material: bpy.types.Material, node_type: str, name: str = None, location=None):
    node = material.node_tree.nodes.new(node_type)
    if name:
        node.name = name
        node.label = name
    if location is not None:
        node.location = location
    return node


def create_texture_node(material: bpy.types.Material, texture: bpy.types.Image, name=None, location=None):
    texture_node = create_node(material, Nodes.ShaderNodeTexImage, name)
    if texture is not None:
        texture_node.image = texture
    if location is not None:
        texture_node.location = location
    return texture_node


def connect_nodes(material: bpy.types.Material, output_socket: bpy.types.NodeSocket,
                  input_socket: bpy.types.NodeSocket):
    material.node_tree.links.new(output_socket, input_socket)


def connect_nodes_group(group: bpy.types.NodeTree, output_socket: bpy.types.NodeSocket,
                        input_socket: bpy.types.NodeSocket):
    group.links.new(output_socket, input_socket)


def create_node_group(material: bpy.types.Material, group_name, location=None, *, name=None):
    group_node = create_node(material, Nodes.ShaderNodeGroup, name or group_name)
    group_node.node_tree = bpy.data.node_groups.get(group_name)
    group_node.width = group_node.bl_width_max
    if location is not None:
        group_node.location = location
    return group_node
