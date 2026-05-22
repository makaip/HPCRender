# HPCRender Blender Extension

A Blender extension to render frames and animations on a high performance compute cluster using SLURM.

Tips:

- Reduce tile size to improve load balancing performance
- Check available partitions with `sinfo -s`
- Run `module avail cuda` to find available versions

Test Scene
- 30 frames, 3 nodes, 4 GPUs per node: 55 seconds. GPU unknown (probably A5500)
- 30 frames, local: 1 hr 57 mins
