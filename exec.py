import bpy
import subprocess

from .helpers import _download_remote_renders, _execute_render, _get_local_output_dir
from .prefs import get_prefs

class HPC_OT_RenderFrame(bpy.types.Operator):
    """Render current frame on HPC cluster"""
    bl_idname = "hpc.render_frame"
    bl_label = "Render Frame on HPC"

    def execute(self, context):
        frame = context.scene.frame_current
        self.report({'INFO'}, f"Rendering frame {frame}...")
        return _execute_render(self, context, frame=frame)


class HPC_OT_RenderAnimation(bpy.types.Operator):
    """Render animation on HPC cluster"""
    bl_idname = "hpc.render_animation"
    bl_label = "Render Animation on HPC"

    def execute(self, context):
        self.report({'INFO'}, "Rendering full animation...")
        return _execute_render(self, context, frame=None)


class HPC_OT_DownloadRenders(bpy.types.Operator):
    """Download renders from cluster into local project folder (in case you loose connection)"""
    bl_idname = "hpc.download_renders"
    bl_label = "Download Renders from HPC"

    def execute(self, context):
        prefs = get_prefs(context)
        blend_file = bpy.data.filepath

        if not blend_file:
            self.report({'ERROR'}, "Please save your .blend file first.")
            return {'CANCELLED'}

        local_dir = _get_local_output_dir(context)
        local_dir.mkdir(parents=True, exist_ok=True)

        self.report({'INFO'}, f"Downloading renders to {local_dir}...")
        ok, error = _download_remote_renders(prefs.host, prefs.remote_dir, local_dir)

        if not ok:
            self.report({'ERROR'}, f"Download failed:\n{error}")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Renders saved to {local_dir}")
        return {'FINISHED'}
