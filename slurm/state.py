import re
import threading


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
    from ..helpers import _read_remote_log_tail

    tail = _read_remote_log_tail(host, remote_log_path, max_lines=120)
    if not tail:
        return None

    for line in reversed(tail.splitlines()):
        parsed = _parse_render_log_line(line)
        if parsed:
            return parsed

    return None
