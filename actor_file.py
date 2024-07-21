#  Copyright 2024 by REDxEYE.
#  All rights reserved.

from typing import List

from x4.compiled_file import CompiledFile, CompiledHeader, ActorChunkId, Nodes, Mesh, SkinningInfo, StdMaterial, Node


class ActorFile(CompiledFile):

    @classmethod
    def is_valid_header(cls, header: CompiledHeader):
        return header.ident == "XAC " and header.version == (1, 0)

    def get_nodes(self) -> Nodes:
        return self.get_chunk(ActorChunkId.NODES)

    def get_meshes(self) -> List[Mesh]:
        return self.get_chunks(ActorChunkId.MESH)

    def get_skinning_infos(self) -> List[SkinningInfo]:
        return self.get_chunks(ActorChunkId.SKINNINGINFO)

    def get_materials(self) -> List[StdMaterial]:
        return self.get_chunks(ActorChunkId.STDMATERIAL)

    def get_mesh_by_node(self, node: Node) -> Mesh | None:
        meshes = self.get_meshes()
        for mesh in meshes:
            if mesh.node_id == node.id:
                return mesh

    def get_skinning_info_by_node(self, node: Node) -> SkinningInfo | None:
        skinning_infos = self.get_skinning_infos()
        for skinning_info in skinning_infos:
            if skinning_info.node_id == node.id:
                return skinning_info
