from pathlib import PurePosixPath


def _format_slurm_time(time_limit):
    if isinstance(time_limit, str):
        if ":" in time_limit:
            return time_limit
        time_limit = int(time_limit)

    hours, remainder = divmod(int(time_limit), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _module_load_snippet(cuda_module: str):
    cuda_module = (cuda_module or "").strip()
    if not cuda_module:
        return ""

    return f"""
if ! command -v module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/module/init/bash; do
        if [ -f "$module_init" ]; then
            # shellcheck disable=SC1090
            source "$module_init"
            break
        fi
    done
fi

if command -v module >/dev/null 2>&1; then
    module load {cuda_module}
else
    echo "Warning: module command not available after shell initialization; continuing without module load."
fi
"""


def _build_slurm_script(prefs, blender_cmd: str, context):
    scene = context.scene
    ntasks = getattr(scene, "hpcrender_ntasks", 1)
    cpus = getattr(scene, "hpcrender_cpus", 1)
    gpus = getattr(scene, "hpcrender_gpus",
                   prefs.gpus if hasattr(prefs, "gpus") else 1)
    mem = getattr(scene, "hpcrender_mem", 64)
    time_limit = getattr(scene, "hpcrender_time_limit", 21600)
    remote_logs_dir = str(PurePosixPath(prefs.remote_dir) / "slurm-logs")

    return f"""#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}G
#SBATCH --time={_format_slurm_time(time_limit)}
#SBATCH --partition={prefs.partition}
#SBATCH --output={remote_logs_dir}/%j-hpcrender.out
#SBATCH --error={remote_logs_dir}/%j-hpcrender.err

mkdir -p "{remote_logs_dir}"
{_module_load_snippet(prefs.cuda_module)}
{blender_cmd}
"""


def _build_distributed_animation_slurm_script(prefs, blender_cmd: str, context, nodes: int):
    scene = context.scene
    cpus = getattr(scene, "hpcrender_cpus", 1)
    gpus = getattr(scene, "hpcrender_gpus",
                   prefs.gpus if hasattr(prefs, "gpus") else 1)
    mem = getattr(scene, "hpcrender_mem", 64)
    time_limit = getattr(scene, "hpcrender_time_limit", 21600)
    remote_logs_dir = str(PurePosixPath(prefs.remote_dir) / "slurm-logs")
    frame_start = int(getattr(scene, "frame_start", 1))
    frame_end = int(getattr(scene, "frame_end", frame_start))
    frame_step = max(1, int(getattr(scene, "frame_step", 1)))
    frame_list = "\n".join(
        str(frame) for frame in range(frame_start, frame_end + 1, frame_step)
    )
    batch_size = max(1, int(getattr(scene, "hpcrender_batch_size", 8)))

    gpu_setup = ""
    if prefs.use_gpu:
        gpu_setup = """
pref = bpy.context.preferences.addons['cycles'].preferences
pref.get_devices()
pref.compute_device_type = 'CUDA'
[setattr(d, 'use', True) for d in pref.devices if d.type == 'CUDA']
bpy.context.scene.cycles.device = 'GPU'
"""

    return f"""#!/bin/bash -l
#SBATCH --nodes={nodes}
#SBATCH --ntasks={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}G
#SBATCH --time={_format_slurm_time(time_limit)}
#SBATCH --partition={prefs.partition}
#SBATCH --output={remote_logs_dir}/%j-hpcrender.out
#SBATCH --error={remote_logs_dir}/%j-hpcrender.err

mkdir -p "{remote_logs_dir}"
{_module_load_snippet(prefs.cuda_module)}
python_bin="$(command -v python3 || command -v python)"
if [ -z "$python_bin" ]; then
    echo "python3/python not found on PATH."
    exit 1
fi
frame_queue="{remote_logs_dir}/${{SLURM_JOB_ID}}-hpcrender-frames.txt"
cat > "$frame_queue" <<'EOF'
{frame_list}
EOF

render_batch() {{
    local batch_script
    local frame_args
    local status

    frame_args="$(IFS=,; echo "$*")"
    batch_script="$(mktemp "{remote_logs_dir}/$SLURM_JOB_ID-batch-XXXXXX.py")"

    cat > "$batch_script" <<PY
import bpy

frames = [${{frame_args}}]
{gpu_setup}for frame in frames:
    print(f"Distributed render | Frame:{{frame}} | Node:${{SLURM_NODEID:-unknown}}")
    bpy.context.scene.frame_set(frame)
    bpy.ops.render.render(write_still=True)
PY

    {blender_cmd} --python "$batch_script"
    status=$?
    rm -f "$batch_script"
    return $status
}}

pop_frame() {{
    "$python_bin" - "$1" <<'PY'
import fcntl
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open("r+", encoding="utf-8") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    frames = [line.strip() for line in handle if line.strip()]
    if not frames:
        raise SystemExit(0)
    frame = frames[0]
    handle.seek(0)
    handle.truncate(0)
    handle.write("\\n".join(frames[1:]))
    if frames[1:]:
        handle.write("\\n")
print(frame)
PY
}}

worker() {{
    while true; do
        batch_frames=()
        while [ "${{#batch_frames[@]}}" -lt {batch_size} ]; do
            frame="$(pop_frame "$frame_queue")"
            if [ -z "$frame" ]; then
                break
            fi
            batch_frames+=("$frame")
        done

        if [ "${{#batch_frames[@]}}" -eq 0 ]; then
            break
        fi

        render_batch "${{batch_frames[@]}}"
    done
}}

export -f pop_frame
export -f worker
export python_bin
export frame_queue

srun --kill-on-bad-exit=0 --ntasks={nodes} --ntasks-per-node=1 bash -lc "worker"
"""
