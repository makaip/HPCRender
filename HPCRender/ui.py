import bpy
from .helpers import _is_video_output
from .prefs import get_prefs
from .slurm.state import get_latest_queue_info

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
        nodes = getattr(scene, "hpcrender_nodes", 1)

        col = layout.column(align=True)
        col.label(text=f"Host: {prefs.host}", icon="WORLD")
        col.label(text=f"Partition: {prefs.partition}", icon="DRIVER")
        col.prop(scene, "hpcrender_nodes")
        if nodes > 1:
            col.prop(scene, "hpcrender_batch_size")
        col.prop(scene, "hpcrender_gpus")
        col.prop(scene, "hpcrender_ntasks")
        col.prop(scene, "hpcrender_cpus")
        col.prop(scene, "hpcrender_mem")
        col.prop(scene, "hpcrender_time_limit")
        col.label(
            text=f"GPU render: {'on' if prefs.use_gpu else 'off'}")
        
        layout.separator()

        if nodes > 1:
            layout.label(
                text="Multi-node mode renders animations only.",
                icon="INFO",
            )
            if _is_video_output(scene):
                layout.label(
                    text="Multi-node animation requires an image sequence output.",
                    icon="ERROR",
                )
            layout.operator("hpc.render_animation", icon="RENDER_ANIMATION")
        else:
            row = layout.row(align=True)
            row.operator("hpc.render_frame", icon="RENDER_STILL")
            row.operator("hpc.render_animation", icon="RENDER_ANIMATION")

        layout.operator("hpc.download_renders", icon="IMPORT")

        queue_info = get_latest_queue_info()
        render_status = queue_info.get("render_status", "").strip()
        frame = queue_info.get("frame", "").strip()
        sample = queue_info.get("sample", "").strip()
        sample_total = queue_info.get("sample_total", "").strip()

        if queue_info["job_id"]:
            layout.label(
                text=(
                    f"Status: {queue_info['status']}. "
                    f"Elapsed: {queue_info['elapsed']}. "
                    f"Job ID: {queue_info['job_id']}"
                )
            )
            if render_status:
                layout.label(text=f"Render: {render_status}")
            if frame or sample:
                progress_text = []
                if frame:
                    progress_text.append(f"Frame {frame}")
                if sample:
                    if sample_total:
                        progress_text.append(f"Sample {sample}/{sample_total}")
                    else:
                        progress_text.append(f"Sample {sample}")
                layout.label(text="Progress: " + " | ".join(progress_text))
        else:
            layout.label(text="Status: Idle. Elapsed: 0s. Job ID: -")

