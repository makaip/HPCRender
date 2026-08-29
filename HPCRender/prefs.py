import bpy

ADDON_ID = __package__ or __name__.partition(".")[0]

# pyright: ignore[reportInvalidTypeForm] shushes pylance

class HPCRenderPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    default_destination: bpy.props.StringProperty(
        name="Default Destination",
        description=(
            "Default SSH destination for new scenes, in the form "
            "user@host:/remote/project/dir."
        ),
        default="userid@login.hpc.university.edu:/mnt/beegfs/home/youruser/etc.",
    )  # pyright: ignore[reportInvalidTypeForm]

    remote_blender: bpy.props.StringProperty(
        name="Blender path (cluster)",
        description="Absolute path to the blender binary on the cluster",
        default="/mnt/beegfs/home/youruser/Desktop/blender-5.0.0-linux-x64/blender",
    )  # pyright: ignore[reportInvalidTypeForm]

    cuda_module: bpy.props.StringProperty(
        name="CUDA module",
        description="CUDA module to load before rendering",
        default="cuda/12.4.0-gcc-13.2.0-shyinv2",
    )  # pyright: ignore[reportInvalidTypeForm]

    partition: bpy.props.StringProperty(
        name="SLURM partition",
        default="shortq7-gpu",
    )  # pyright: ignore[reportInvalidTypeForm]

    use_gpu: bpy.props.BoolProperty(
        name="GPU rendering (Cycles/CUDA)",
        default=True,
    )  # pyright: ignore[reportInvalidTypeForm]

    poll_interval_seconds: bpy.props.IntProperty(
        name="Poll interval (seconds)",
        description="How often the add-on refreshes SLURM and render log status",
        default=10,
        min=1,
        max=3600,
    )  # pyright: ignore[reportInvalidTypeForm]

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "default_destination")
        layout.prop(self, "remote_blender")
        layout.prop(self, "cuda_module")
        layout.prop(self, "partition")

        layout.separator()

        layout.prop(self, "use_gpu")
        layout.prop(self, "poll_interval_seconds")


def get_prefs(context):
    return context.preferences.addons[ADDON_ID].preferences


def _destination_get(self):
    raw = self.get("hpcrender_destination_raw", "")
    if raw:
        return raw
    try:
        return bpy.context.preferences.addons[ADDON_ID].preferences.default_destination
    except Exception:
        return ""


def _destination_set(self, value):
    self["hpcrender_destination_raw"] = value


def _parse_destination(destination: str):
    """Split a 'user@host:/remote/dir' string into (host, remote_dir)."""
    destination = (destination or "").strip()
    if ":" not in destination:
        raise ValueError(
            f"Destination must be in the form user@host:/remote/dir, got: {destination!r}"
        )
    host, remote_dir = destination.split(":", 1)
    host = host.strip()
    remote_dir = remote_dir.strip()
    if not host or not remote_dir:
        raise ValueError(
            f"Destination must be in the form user@host:/remote/dir, got: {destination!r}"
        )
    return host, remote_dir


def get_destination(context):
    scene = context.scene
    destination = getattr(scene, "hpcrender_destination", "") or get_prefs(context).default_destination
    return _parse_destination(destination)

