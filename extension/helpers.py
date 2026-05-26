import bpy
import re
import subprocess
import time
from pathlib import Path, PurePosixPath

from .prefs import get_prefs

CUDA_PYTHON_EXPR = (
    "import bpy; "
    "pref = bpy.context.preferences.addons['cycles'].preferences; "
    "pref.get_devices(); "
    "pref.compute_device_type = 'CUDA'; "
    "[setattr(d, 'use', True) for d in pref.devices if d.type == 'CUDA']; "
    "bpy.context.scene.cycles.device = 'GPU'"
)

VIDEO_FILE_FORMATS = {
    "FFMPEG",
    "AVI_JPEG",
    "AVI_RAW",
}


# https://docs.blender.org/api/current/bpy.types.WindowManager.html

def _notify_user(title: str, lines, icon: str = "INFO"):
    def _show_popup():
        wm = bpy.context.window_manager

        def _draw(self, _context):
            for line in lines:
                self.layout.label(text=line)

        if wm is not None:
            wm.popup_menu(_draw, title=title, icon=icon)
        for line in lines:
            print(f"[HPCRender] {line}")
        return None

    bpy.app.timers.register(_show_popup, first_interval=0.0)


def _format_duration(total_seconds: float) -> str:
    seconds = int(round(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"

    return f"{secs}s"


def _read_remote_log_tail(host: str, remote_log_path: str, max_lines: int = 80):
    cmd = ["ssh", host,
           f"tail -n {max_lines} {remote_log_path} 2>/dev/null || true"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return ""

    return result.stdout.strip()


def _scp_upload(prefs, local_path: Path, operator):
    remote_dir_str = str(PurePosixPath(prefs.remote_dir))
    
    # add -p to ensure remote dir actually exists
    mkdir_cmd = ["ssh", prefs.host, f"mkdir -p {remote_dir_str}"]
    operator.report({'INFO'}, f"Creating remote directory: {' '.join(mkdir_cmd)}")
    mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True)

    if mkdir_result.returncode != 0:
        operator.report({'ERROR'}, f"Failed to create remote directory:\n{mkdir_result.stderr}")
        return False

    remote = f"{prefs.host}:{remote_dir_str}/{local_path.name}"
    cmd = ["scp", str(local_path), remote]
    operator.report({'INFO'}, f"Uploading: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        operator.report({'ERROR'}, f"scp failed:\n{result.stderr}")
        return False

    return True


def _download_remote_renders(host: str, remote_dir: str, local_dir: Path):
    remote_dir_str = str(PurePosixPath(remote_dir))
    remote_renders = f"{host}:{remote_dir_str}/renders/."
    cmd = ["scp", "-r", remote_renders, str(local_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.returncode == 0, result.stderr


def _build_blender_cmd(prefs, blend_remote: str, output_remote: str, frame=None):
    # frame=None -> render animation (-a); frame=int -> single frame (-f N).

    parts = [prefs.remote_blender, "-b", blend_remote]

    if prefs.use_gpu:
        parts += ["--python-expr", f'"{CUDA_PYTHON_EXPR}"']

    parts += ["-o", output_remote, "-x", "1"]

    if frame is None:
        parts.append("-a")
    else:
        parts += ["-f", str(frame)]

    return " ".join(parts)


def _build_blender_base_cmd(prefs, blend_remote: str, output_remote: str):
    return " ".join([
        prefs.remote_blender,
        "-b",
        blend_remote,
        "-o",
        output_remote,
        "-x",
        "1",
    ])


def _build_distributed_animation_cmd(prefs, blend_remote: str, output_remote: str):
    return _build_blender_base_cmd(prefs, blend_remote, output_remote)


def _is_video_output(scene):
    render = getattr(scene, "render", None)
    image_settings = getattr(render, "image_settings", None)
    file_format = getattr(image_settings, "file_format", "")
    return file_format in VIDEO_FILE_FORMATS


def _execute_render(self, context, frame=None):
    from .slurm.monitor import _start_async_monitor
    from .slurm.monitor import _submit_job
    from .slurm.scripts import _build_distributed_animation_slurm_script
    from .slurm.scripts import _build_slurm_script

    prefs = get_prefs(context)
    scene = context.scene
    nodes = max(1, int(getattr(scene, "hpcrender_nodes", 1)))

    if frame is not None and nodes > 1:
        self.report(
            {'ERROR'},
            "Single-frame renders are disabled when Nodes is greater than 1.",
        )
        return {'CANCELLED'}

    if frame is None and nodes > 1:
        frame_start = int(getattr(scene, "frame_start", 1))
        frame_end = int(getattr(scene, "frame_end", frame_start))
        if frame_end < frame_start:
            self.report(
                {'ERROR'},
                "Frame end must be greater than or equal to frame start for distributed animation renders.",
            )
            return {'CANCELLED'}

        if _is_video_output(scene):
            self.report(
                {'ERROR'},
                "Multinode animation rendering has to use an image output. Change Render Properties > Output > File Format to an image sequence format.",
            )
            return {'CANCELLED'}

    blend_file = bpy.data.filepath
    if not blend_file:
        self.report({'ERROR'}, "Please save your .blend file first.")
        return {'CANCELLED'}

    blend_path = Path(blend_file)
    blend_name = blend_path.name
    remote_blend = str(PurePosixPath(prefs.remote_dir) / blend_name)

    stem = blend_path.stem
    if frame is None:
        out_name = f"{stem}_anim####"
    else:
        out_name = f"{stem}_frame####"
    remote_out = str(PurePosixPath(prefs.remote_dir) / "renders" / out_name)

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_mainfile()

    if not _scp_upload(prefs, blend_path, self):
        return {'CANCELLED'}

    if frame is None and nodes > 1:
        blender_cmd = _build_distributed_animation_cmd(
            prefs, remote_blend, remote_out)
        script = _build_distributed_animation_slurm_script(
            prefs, blender_cmd, context, nodes)
    else:
        blender_cmd = _build_blender_cmd(
            prefs, remote_blend, remote_out, frame=frame)
        script = _build_slurm_script(prefs, blender_cmd, context)

    job_id = _submit_job(prefs, script, self)
    if job_id is None:
        return {'CANCELLED'}

    local_dir = _get_local_output_dir(context)
    local_dir.mkdir(parents=True, exist_ok=True)
    _start_async_monitor(
        prefs.host,
        prefs.remote_dir,
        job_id,
        local_dir,
        prefs.poll_interval_seconds,
    )
    self.report(
        {'INFO'}, "Results will automatically download when job finishes.")

    return {'FINISHED'}


def _get_local_output_dir(context):
    raw_output_path = context.scene.render.filepath
    output_path = Path(bpy.path.abspath(raw_output_path))

    if raw_output_path.endswith(("/", "\\")):
        return output_path

    return output_path.parent
