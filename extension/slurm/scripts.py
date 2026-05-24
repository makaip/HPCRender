from pathlib import Path, PurePosixPath


_TEMPLATE_DIR = Path(__file__).resolve().parent


def _render_template(template_name: str, **kwargs):
    template = (_TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    return template.format(**kwargs)


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

    return _render_template("templates/module_load.txt", cuda_module=cuda_module)


def _distributed_gpu_setup_snippet(prefs):
    if not prefs.use_gpu:
        return ""

    return _render_template("templates/dist_gpu.txt")


def _build_slurm_script(prefs, blender_cmd: str, context):
    scene = context.scene
    ntasks = getattr(scene, "hpcrender_ntasks", 1)
    cpus = getattr(scene, "hpcrender_cpus", 1)
    gpus = getattr(scene, "hpcrender_gpus",
                   prefs.gpus if hasattr(prefs, "gpus") else 1)
    mem = getattr(scene, "hpcrender_mem", 64)
    time_limit = getattr(scene, "hpcrender_time_limit", 21600)
    remote_logs_dir = str(PurePosixPath(prefs.remote_dir) / "slurm-logs")

    return _render_template(
        "slurm.txt",
        ntasks=ntasks,
        cpus=cpus,
        gpus=gpus,
        mem=mem,
        time_limit=_format_slurm_time(time_limit),
        partition=prefs.partition,
        remote_logs_dir=remote_logs_dir,
        module_load_snippet=_module_load_snippet(prefs.cuda_module),
        blender_cmd=blender_cmd,
    )


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
    batch_python_script = _render_template(
        "templates/dist_batch.txt",
        gpu_setup=_distributed_gpu_setup_snippet(prefs),
    )

    return _render_template(
        "templates/dist_anim.txt",
        nodes=nodes,
        cpus=cpus,
        gpus=gpus,
        mem=mem,
        time_limit=_format_slurm_time(time_limit),
        partition=prefs.partition,
        remote_logs_dir=remote_logs_dir,
        module_load_snippet=_module_load_snippet(prefs.cuda_module),
        frame_list=frame_list,
        batch_size=batch_size,
        batch_python_script=batch_python_script,
        blender_cmd=blender_cmd,
    )
