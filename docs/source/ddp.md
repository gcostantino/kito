# Distributed training (DDP)

Set `TrainingConfig(distributed_training=True)`. There are two ways to start.

## 1. Launched with torchrun

If `RANK`, `LOCAL_RANK` and `WORLD_SIZE` are set, the Engine initializes the process group
itself, and `engine.fit(...)` runs in each process.

```bash
torchrun --nproc_per_node=2 train.py
```

## 2. Automatic spawn

Without those variables, `engine.fit(data_pipeline=pipeline, ...)` spawns one worker per GPU
with `torch.multiprocessing.spawn`.

- The number of workers is `KITO_NPROCS` if set, otherwise `torch.cuda.device_count()`. It must be at least 2.
- `MASTER_ADDR` defaults to `localhost`; `MASTER_PORT` defaults to a free port.
- You must pass `data_pipeline`; pre-built DataLoaders raise `ValueError`.
- Each worker creates `type(module)(config)` and an Engine, then runs `fit`. A module whose
  constructor needs extra arguments cannot be spawned this way. The callbacks and the pipeline are pickled and
  sent to the workers.

## Behavior

- Rank 0 handles callbacks. Each callback is wrapped in `DDPAwareCallback(master_rank=0)`.
- Only rank 0 logs messages below error level, and progress bars use DDP-aware handlers.
- `stop_training` is synchronized across ranks after each epoch.
- `ddp_find_unused_parameters` is forwarded to `DistributedDataParallel`.
