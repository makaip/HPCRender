# https://docs.blender.org/manual/en/latest/advanced/scripting/addon_tutorial.html

import bpy
import subprocess
import os
import re
from pathlib import Path, PurePosixPath

bl_info = {
    "name": "HPCRender",
}

class RenderFrame(bpy.types.Operator):
    """Render a single frame on your HPC cluster"""     # tooltip for menu items and buttons
    bl_idname = "hpc.render_frame"                      # UID for buttons and menu items to reference
    bl_label = "Render Frame on HPC cluster"            # display name in interface

    # execute() is called when running the operator.
    def execute(self, context):
        scene = context.scene

        blend_file = bpy.data.filepath

        if not blend_file:
            self.report(
                {'ERROR'}, "Please save your .blend file before running the operator.")
            return {'CANCELLED'}

        blend_path = Path(blend_file)

        # do the rest later

        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(RenderFrame.bl_idname)


def register():
    bpy.utils.register_class(RenderFrame)
    bpy.types.VIEW3D_MT_object.append(menu_func)  # add operator to menu


def unregister():
    bpy.utils.unregister_class(RenderFrame)


if __name__ == "__main__":
    register()
