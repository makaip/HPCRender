# HPCRender Blender Extension

A Blender extension to render frames and animations on a high performance compute cluster using SLURM.

Tips:

- Reduce tile size to improve load balancing performance
- Check available partitions with `sinfo -s`
- Run `module avail cuda` to find available versions

Test Scene (100 frames, 1 node): 49m 22s

Test Scene (25 frames, 2 nodes): 