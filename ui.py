import bpy
from .prefs import get_prefs
from .slurm import get_latest_queue_info

# https://docs.blender.org/api/current/bpy.types.Panel.html
class HPC_PT_RenderPanel(bpy.types.Panel):
    bl_label = "HPC Render"
    bl_idname = "HPC_PT_render_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        prefs = get_prefs(context)
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.label(text=f"Host: {prefs.host}", icon="WORLD")
        col.label(text=f"Partition: {prefs.partition}", icon="DRIVER")
        col.prop(scene, "hpcrender_gpus")
        col.prop(scene, "hpcrender_ntasks")
        col.prop(scene, "hpcrender_cpus")
        col.prop(scene, "hpcrender_mem")
        col.prop(scene, "hpcrender_time_limit")
        col.label(
            text=f"GPU render: {'on' if prefs.use_gpu else 'off'}")

        layout.separator()

        row = layout.row(align=True)
        row.operator("hpc.render_frame", icon="RENDER_STILL")
        row.operator("hpc.render_animation", icon="RENDER_ANIMATION")

        layout.operator("hpc.download_renders", icon="IMPORT")

        queue_info = get_latest_queue_info()
        if queue_info["job_id"]:
            layout.label(
                text=(
                    f"Status: {queue_info['status']}. "
                    f"Elapsed: {queue_info['elapsed']}. "
                    f"Job ID: {queue_info['job_id']}"
                )
            )
        else:
            layout.label(text="Status: Idle. Elapsed: 0s. Job ID: -")

