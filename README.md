# HPCRender

A Blender extension to render frames and animations on a high performance compute cluster using SLURM.

## Setup

TODO: write this later

## Tips

- Reduce tile size to improve load balancing performance (e.g. 512 or 1024)
- Check available partitions with `sinfo -s`
- Check available configs with `sinfo -N -l`
- Run `module avail cuda` to find available versions of CUDA

### Benchmark

Test Scene:

- 30 frames, 3 nodes, 4 GPUs per node: 55 seconds. GPU unknown (probably A5500s)
- 30 frames, local: 1 hr 57 mins. Single RTX2060 Super
