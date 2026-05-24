# https://docs.blender.org/manual/en/latest/advanced/scripting/addon_tutorial.html

import bpy

from .exec import HPC_OT_DownloadRenders, HPC_OT_RenderAnimation, HPC_OT_RenderFrame
from .prefs import HPCRenderPreferences, get_prefs
from .ui import HPC_PT_RenderPanel

bl_info = {
    "name": "HPCRender",
    "author": "Makai P.",
    "version": (0, 0, 1),
    "blender": (5, 0, 0),
    "location": "Render menu > HPC Render  |  Properties > Render > HPC Render",
    "description": "Render frames and animations on a SLURM HPC cluster via SSH/SCP",
    "category": "Render",
}

CLASSES = [
    HPCRenderPreferences,
    HPC_OT_RenderFrame,
    HPC_OT_RenderAnimation,
    HPC_OT_DownloadRenders,
    HPC_PT_RenderPanel,
]


def menu_func(self, context):
    scene = context.scene
    self.layout.separator()
    if getattr(scene, "hpcrender_nodes", 1) <= 1:
        self.layout.operator("hpc.render_frame")
    self.layout.operator("hpc.render_animation")


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.hpcrender_gpus = bpy.props.IntProperty(
        name="GPUs", default=4, min=1, max=9999
    )
    bpy.types.Scene.hpcrender_nodes = bpy.props.IntProperty(
        name="Nodes", default=1, min=1, max=9999
    )
    bpy.types.Scene.hpcrender_batch_size = bpy.props.IntProperty(
        name="Batch size", default=8, min=1, max=9999
    )
    bpy.types.Scene.hpcrender_gpu_limit = bpy.props.IntProperty(
        name="GPU limit", default=0, min=0, max=9999
    )
    bpy.types.Scene.hpcrender_ntasks = bpy.props.IntProperty(
        name="Tasks", default=1, min=1
    )
    bpy.types.Scene.hpcrender_cpus = bpy.props.IntProperty(
        name="CPU cores per task", default=32, min=1
    )
    bpy.types.Scene.hpcrender_mem = bpy.props.IntProperty(
        name="Memory (GB)", default=64, min=1
    )
    bpy.types.Scene.hpcrender_time_limit = bpy.props.IntProperty(
        name="Time limit (seconds)", default=21600, min=1
    )
    bpy.types.TOPBAR_MT_render.append(menu_func)
    bpy.types.VIEW3D_MT_object.append(menu_func)

# note: unregister in the opposite order of register
def unregister():
    bpy.types.TOPBAR_MT_render.remove(menu_func)
    bpy.types.VIEW3D_MT_object.remove(menu_func)

    # remove scene-level properties
    try:
        del bpy.types.Scene.hpcrender_gpus
        del bpy.types.Scene.hpcrender_nodes
        del bpy.types.Scene.hpcrender_batch_size
        del bpy.types.Scene.hpcrender_gpu_limit
        del bpy.types.Scene.hpcrender_cpus
        del bpy.types.Scene.hpcrender_ntasks
        del bpy.types.Scene.hpcrender_mem
        del bpy.types.Scene.hpcrender_time_limit
    except Exception:
        pass
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
