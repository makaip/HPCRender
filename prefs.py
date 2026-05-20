import bpy

ADDON_ID = __package__ or __name__.partition(".")[0]

# pyright: ignore[reportInvalidTypeForm] shushes pylance

class HPCRenderPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    host: bpy.props.StringProperty(
        name="SSH Host",
        description="user@hostname for the cluster login node",
        default="userid@login.hpc.university.edu",
    )  # pyright: ignore[reportInvalidTypeForm]

    remote_blender: bpy.props.StringProperty(
        name="Blender path (cluster)",
        description="Absolute path to the blender binary on the cluster",
        default="/mnt/beegfs/home/youruser/Desktop/blender-4.5.5-linux-x64/blender",
    )  # pyright: ignore[reportInvalidTypeForm]

    remote_dir: bpy.props.StringProperty(
        name="Remote project dir",
        description="Directory on the cluster where the .blend file will be uploaded",
        default="/mnt/beegfs/home/youruser/etc.",
    )  # pyright: ignore[reportInvalidTypeForm]

    cuda_module: bpy.props.StringProperty(
        name="CUDA module",
        description="CUDA module to load before rendering. Run `module avail cuda` to find available versions",
        default="cuda/12.4.0-gcc-13.2.0-shyinv2",
    )  # pyright: ignore[reportInvalidTypeForm]

    partition: bpy.props.StringProperty(
        name="SLURM partition. Check available partitions with `sinfo -s`",
        default="shortq7-gpu",
    )  # pyright: ignore[reportInvalidTypeForm]

    use_gpu: bpy.props.BoolProperty(
        name="GPU rendering (Cycles/CUDA)",
        default=True,
    )  # pyright: ignore[reportInvalidTypeForm]

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "host")
        layout.prop(self, "remote_blender")
        layout.prop(self, "remote_dir")
        layout.prop(self, "cuda_module")
        layout.prop(self, "partition")

        layout.separator()

        layout.prop(self, "use_gpu")


def get_prefs(context):
    return context.preferences.addons[ADDON_ID].preferences

