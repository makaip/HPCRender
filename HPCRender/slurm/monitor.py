import re
import subprocess
import threading
import time
from pathlib import Path
from pathlib import PurePosixPath

from .state import _read_latest_render_status
from .state import _set_latest_queue_info

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


def _submit_job(host: str, slurm_script: str, operator):
    # escape single quotes in the script so we can wrap it in $'...'
    escaped = slurm_script.replace("\\", "\\\\").replace("'", "\\'")
    ssh_cmd = [
        "ssh", host,
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
                from ..helpers import _format_duration
                from ..helpers import _notify_user

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

    from ..helpers import _download_remote_renders
    from ..helpers import _format_duration
    from ..helpers import _notify_user
    from ..helpers import _read_remote_log_tail

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
