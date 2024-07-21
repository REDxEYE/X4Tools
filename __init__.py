#  Copyright 2024 by REDxEYE.
#  All rights reserved.


from pathlib import Path

import bpy
from bpy.props import StringProperty, CollectionProperty

from x4.actor_file import ActorFile
from x4.file_utils import FileBuffer
from x4.import_xac import import_actor
from x4.import_xmf import import_xmf
from x4.static_model import StaticModel

bl_info = {
    "name": "X4 tools",
    "author": "RED_EYE",
    "version": (0, 0, 1),
    "blender": (3, 6, 0),
    "location": "File > Import > X4 xac",
    "description": "X4 import/export tools",
    "category": "Import-Export"
}


def is_blender_4():
    return bpy.app.version >= (4, 0, 0)


def is_blender_4_1():
    return bpy.app.version >= (4, 1, 0)


class ImportOperatorHelper(bpy.types.Operator):
    need_popup = True
    if is_blender_4_1():
        directory: StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE', 'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH', )
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    def invoke_popup(self, context, confirm_text=""):
        if self.properties.is_property_set("filepath"):
            title = self.filepath
            if len(self.files) > 1:
                title = f"Import {len(self.files)} files"

            if not confirm_text:
                confirm_text = self.bl_label
            return context.window_manager.invoke_props_dialog(self, confirm_text=confirm_text, title=title,
                                                              translate=False)

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if is_blender_4_1() and self.directory and self.files:
            if self.need_popup:
                return self.invoke_popup(context)
            else:
                return self.execute(context)
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def get_directory(self):
        if is_blender_4_1():
            return Path(self.directory)
        else:
            filepath = Path(self.filepath)
            if filepath.is_file():
                return filepath.parent.absolute()
            else:
                return filepath.absolute()


class X4_OT_XACImport(ImportOperatorHelper):
    """Load X4 xac models"""
    bl_idname = "x4.xac"
    bl_label = "Import X4 xac models"
    bl_options = {'UNDO'}

    filter_glob: StringProperty(default="*.xac", options={'HIDDEN'})

    def execute(self, context):
        directory = self.get_directory()

        for file in self.files:
            filepath = directory / file.name
            with FileBuffer(filepath) as f:
                cf = ActorFile.from_buffer(f)
                import_actor(cf, filepath)
        return {'FINISHED'}


class X4_OT_XMFImport(ImportOperatorHelper):
    """Load X4 xmf models"""
    bl_idname = "x4.xmf"
    bl_label = "Import X4 xmf models"
    bl_options = {'UNDO'}

    filter_glob: StringProperty(default="*.xmf", options={'HIDDEN'})

    def execute(self, context):
        directory = self.get_directory()

        for file in self.files:
            filepath = directory / file.name
            with FileBuffer(filepath) as f:
                cf = StaticModel.from_buffer(f)
                import_xmf(filepath, cf)
        return {'FINISHED'}


classes = [X4_OT_XACImport, X4_OT_XMFImport]

register_, unregister_ = bpy.utils.register_classes_factory(classes)


def menu_import(self, context):
    self.layout.operator(X4_OT_XACImport.bl_idname)
    self.layout.operator(X4_OT_XMFImport.bl_idname)


def register():
    register_()
    bpy.types.TOPBAR_MT_file_import.append(menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    unregister_()
