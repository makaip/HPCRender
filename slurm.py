import re
import subprocess
import threading
import time
from pathlib import Path
from pathlib import PurePosixPath

from .helpers import _read_remote_log_tail

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
}

# threading lock to prevent race conditions for the latest-queue cache
_LATEST_QUEUE_INFO_LOCK = threading.Lock()
_LATEST_QUEUE_INFO = {
    "job_id": "",
    "status": "Idle",
    "elapsed": "0s",
    "render_status": "",
    "frame": "",
    "sample": "",
    "sample_total": "",
}


def _set_latest_queue_info(
    job_id: str,
    status: str,
    elapsed: str,
    render_status: str | None = None,
    frame: str | None = None,
    sample: str | None = None,
    sample_total: str | None = None,
):
    with _LATEST_QUEUE_INFO_LOCK:
        _LATEST_QUEUE_INFO["job_id"] = job_id
        _LATEST_QUEUE_INFO["status"] = status
        _LATEST_QUEUE_INFO["elapsed"] = elapsed
        if render_status is not None:
            _LATEST_QUEUE_INFO["render_status"] = render_status
        if frame is not None:
            _LATEST_QUEUE_INFO["frame"] = frame
        if sample is not None:
            _LATEST_QUEUE_INFO["sample"] = sample
        if sample_total is not None:
            _LATEST_QUEUE_INFO["sample_total"] = sample_total


def get_latest_queue_info():
    with _LATEST_QUEUE_INFO_LOCK:
        return dict(_LATEST_QUEUE_INFO)


def _parse_render_log_line(line: str):
    stripped = line.strip()
    if not stripped:
        return None

    parts = [field.strip() for field in stripped.split("|")]
    if len(parts) >= 4:
        render_status = " | ".join(parts[3:]).strip()
    else:
        render_status = stripped

    frame_match = re.search(r"Fra:(\d+)", stripped)
    sample_match = re.search(r"Sample (\d+)/(\d+)", stripped)
    tile_match = re.search(r"Rendered (\d+)/(\d+) Tiles", stripped)

    sample = ""
    sample_total = ""
    if sample_match:
        sample = sample_match.group(1)
        sample_total = sample_match.group(2)
    elif tile_match:
        sample = tile_match.group(1)
        sample_total = tile_match.group(2)

    return {
        "render_status": render_status,
        "frame": frame_match.group(1) if frame_match else "",
        "sample": sample,
        "sample_total": sample_total,
    }


def _read_latest_render_status(host: str, remote_log_path: str):
    tail = _read_remote_log_tail(host, remote_log_path, max_lines=120)
    if not tail:
        return None

    for line in reversed(tail.splitlines()):
        parsed = _parse_render_log_line(line)
        if parsed:
            return parsed

    return None


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


def _submit_job(prefs, slurm_script: str, operator):
    # escape single quotes in the script so we can wrap it in $'...'
    escaped = slurm_script.replace("\\", "\\\\").replace("'", "\\'")
    ssh_cmd = [
        "ssh", prefs.host,
        f"echo $'{escaped}' | sbatch",
    ]

    operator.report({'INFO'}, "Submitting SLURM job...")
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        operator.report({'ERROR'}, f"sbatch failed:\n{result.stderr}")
        return None

    match = re.search(r"(\d+)", result.stdout)

    if not match:
        operator.report(
            {'ERROR'}, f"Could not parse SLURM job id from:\n{result.stdout}")
        return None

    job_id = match.group(1)
    operator.report(
        {'INFO'}, f"Job submitted: {job_id}  |  track with: squeue -j {job_id}")
    _set_latest_queue_info(
        job_id,
        "PENDING",
        "0s",
        render_status="",
        frame="",
        sample="",
        sample_total="",
    )
    return job_id


def _parse_queue_row(row: str):
    fields = [field.strip() for field in row.split("|")]
    if len(fields) >= 3:
        return fields[0], fields[1], fields[2]
    return "", "", ""


def _query_job_info(host: str, job_id: str):
    squeue_cmd = ["ssh", host, f'squeue -h -j {job_id} -o "%i|%T|%M"']
    squeue = subprocess.run(squeue_cmd, capture_output=True, text=True)

    if squeue.returncode == 0:
        state_lines = squeue.stdout.strip().splitlines()
        if state_lines:
            parsed_job_id, parsed_state, parsed_elapsed = _parse_queue_row(
                state_lines[0])
            if parsed_job_id:
                return parsed_job_id, parsed_state.upper(), parsed_elapsed, None

    sacct_cmd = [
        "ssh",
        host,
        f"sacct -n -X -j {job_id} --parsable2 --format=JobID,State,Elapsed | head -n 1",
    ]

    sacct = subprocess.run(sacct_cmd, capture_output=True, text=True)

    if sacct.returncode == 0:
        state_lines = sacct.stdout.strip().splitlines()
        if state_lines:
            parsed_job_id, parsed_state, parsed_elapsed = _parse_queue_row(
                state_lines[0])
            if parsed_job_id:
                return parsed_job_id, parsed_state.upper(), parsed_elapsed, None

    error_blob = (squeue.stderr or "") + \
        ("\n" + sacct.stderr if sacct.stderr else "")
    return job_id, "UNKNOWN", "", error_blob.strip() or "Could not determine job state from squeue/sacct."


def _monitor_job_and_download(
    host: str,
    remote_dir: str,
    job_id: str,
    local_dir: Path,
    start_time: float,
    poll_interval_seconds: int,
):
    query_errors = 0
    remote_logs_dir = str(PurePosixPath(remote_dir) / "slurm-logs")
    remote_out = f"{remote_logs_dir}/{job_id}-hpcrender.out"

    while True:
        parsed_job_id, state, elapsed, state_error = _query_job_info(
            host, job_id)
        render_info = _read_latest_render_status(host, remote_out)
        if parsed_job_id:
            _set_latest_queue_info(
                parsed_job_id,
                state,
                elapsed or "0s",
                **(render_info or {}),
            )
        if state in TERMINAL_STATES:
            break
        if state == "UNKNOWN":
            query_errors += 1
            if query_errors >= 3:
                # import helpers lazily to avoid import cycles
                from .helpers import _format_duration
                from .helpers import _notify_user

                elapsed = _format_duration(time.monotonic() - start_time)
                _notify_user(
                    "HPC Render Monitor Error",
                    [
                        f"Job {job_id}: unable to query status after {elapsed}.",
                        state_error,
                    ],
                    icon="ERROR",
                )
                return
        else:
            query_errors = 0
        time.sleep(max(1, int(poll_interval_seconds)))

    # job finished
    from .helpers import _format_duration
    from .helpers import _read_remote_log_tail
    from .helpers import _download_remote_renders
    from .helpers import _notify_user

    elapsed = _format_duration(time.monotonic() - start_time)
    remote_out = f"{remote_logs_dir}/{job_id}-hpcrender.out"
    remote_err = f"{remote_logs_dir}/{job_id}-hpcrender.err"

    if state == "COMPLETED":
        dl_ok, dl_error = _download_remote_renders(host, remote_dir, local_dir)
        if dl_ok:
            _notify_user(
                "HPC Render Complete",
                [
                    f"Job {job_id} finished in {elapsed}.",
                    f"Renders downloaded to {local_dir}",
                ],
                icon="CHECKMARK",
            )
            return

        _notify_user(
            "HPC Render Download Failed",
            [
                f"Job {job_id} finished in {elapsed}, but auto-download failed.",
                f"Remote stdout: {remote_out}",
                f"Remote stderr: {remote_err}",
                dl_error or "Unknown scp error.",
            ],
            icon="ERROR",
        )
        return

    snippet = _read_remote_log_tail(host, remote_err)
    if not snippet:
        snippet = _read_remote_log_tail(host, remote_out)

    lines = [
        f"Job {job_id} ended with state: {state}",
        f"Elapsed: {elapsed}",
        f"Remote logs: {remote_logs_dir}",
        f"Remote stdout: {remote_out}",
        f"Remote stderr: {remote_err}",
    ]
    if snippet:
        lines.append("Error tail:")
        lines.append(snippet)

    _notify_user("HPC Render Failed", lines, icon="ERROR")


def _start_async_monitor(
    host: str,
    remote_dir: str,
    job_id: str,
    local_dir: Path,
    poll_interval_seconds: int,
):
    thread = threading.Thread(
        target=_monitor_job_and_download,
        args=(
            host,
            remote_dir,
            job_id,
            local_dir,
            time.monotonic(),
            poll_interval_seconds,
        ),
        daemon=True,
        name=f"HPCRenderMonitor-{job_id}",
    )

    thread.start()
