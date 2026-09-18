"""Small, explicit process-group helpers for the native Trainer path.

The native CLI is also used as a regular single-process Python program.  A
process group is therefore created only when the request explicitly asks for
multiple local processes (or when torchrun has supplied a multi-process
environment).  Failing on a mismatch is intentional: silently using one GPU
would produce a run with a different training contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Runtime rank/device information for one native worker."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None
    owns_process_group: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def _visible_cuda_count() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return torch.cuda.device_count()
    return len([item for item in raw.split(",") if item.strip()])


def initialize(config: object) -> DistributedContext:
    """Initialise the requested process group and select this worker's device."""

    requested = int(getattr(config, "nproc_per_node", 1))
    if requested < 1:
        raise ValueError("nproc_per_node must be positive")

    env_world_size = _env_int("WORLD_SIZE", 1)
    env_rank = _env_int("RANK", 0)
    env_local_rank = _env_int("LOCAL_RANK", 0)
    requested_distributed = bool(getattr(config, "distributed", False))
    distributed = requested_distributed or requested > 1 or env_world_size > 1

    if not distributed:
        if env_world_size != 1:
            raise RuntimeError(
                "torchrun supplied WORLD_SIZE>1 but native distributed mode is disabled"
            )
        device = torch.device(str(getattr(config, "device", "cuda")))
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "native device=cuda was requested but CUDA is unavailable"
                )
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"requested CUDA device {device.index} is unavailable; "
                    f"visible device count is {torch.cuda.device_count()}"
                )
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
            backend=None,
            owns_process_group=False,
        )

    if env_world_size <= 1:
        raise RuntimeError(
            "native distributed mode requires torchrun with WORLD_SIZE>1; "
            f"received nproc_per_node={requested}"
        )
    if env_world_size != requested:
        raise ValueError(
            "nproc_per_node does not match torchrun WORLD_SIZE: "
            f"config={requested}, environment={env_world_size}"
        )
    if not 0 <= env_rank < env_world_size:
        raise RuntimeError(f"RANK={env_rank} is outside WORLD_SIZE={env_world_size}")
    if not 0 <= env_local_rank < env_world_size:
        raise RuntimeError(
            f"LOCAL_RANK={env_local_rank} is outside nproc_per_node={env_world_size}"
        )

    configured_device = str(getattr(config, "device", "cuda"))
    if configured_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "native distributed CUDA mode requested but CUDA is unavailable"
            )
        visible = _visible_cuda_count()
        if visible != env_world_size:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES count does not match nproc_per_node: "
                f"visible={visible}, nproc_per_node={env_world_size}; "
                "set both explicitly"
            )
        if env_local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={env_local_rank} exceeds visible CUDA devices "
                f"({torch.cuda.device_count()})"
            )
        torch.cuda.set_device(env_local_rank)
        device = torch.device("cuda", env_local_rank)
    else:
        device = torch.device(configured_device)

    backend = getattr(config, "distributed_backend", None)
    if backend is None:
        backend = "nccl" if device.type == "cuda" else "gloo"
    if backend == "nccl" and device.type != "cuda":
        raise ValueError("distributed_backend=nccl requires device=cuda")
    if backend == "gloo" and device.type == "cuda":
        raise ValueError("distributed_backend=gloo requires a CPU device")
    if backend not in {"nccl", "gloo"}:
        raise ValueError(f"unsupported distributed backend: {backend!r}")

    owns_process_group = False
    if dist.is_initialized():
        actual_world_size = dist.get_world_size()
        actual_rank = dist.get_rank()
        if actual_world_size != env_world_size or actual_rank != env_rank:
            raise RuntimeError(
                "existing process group does not match torchrun environment: "
                f"group=rank{actual_rank}/{actual_world_size}, "
                f"environment=rank{env_rank}/{env_world_size}"
            )
    else:
        dist.init_process_group(backend=backend, init_method="env://")
        owns_process_group = True

    return DistributedContext(
        enabled=True,
        rank=env_rank,
        local_rank=env_local_rank,
        world_size=env_world_size,
        device=device,
        backend=backend,
        owns_process_group=owns_process_group,
    )


def wrap_model(model: nn.Module, context: DistributedContext) -> nn.Module:
    """Wrap a model for DDP while leaving single-process calls unchanged."""

    if not context.enabled:
        return model
    # ESM2 builds a pooler that plm_output=cls never calls, so those parameters
    # take no gradient and DDP's reduction waits for an all-reduce that never
    # arrives: "Expected to have finished reduction in the prior iteration".
    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=True,
        )
    return DistributedDataParallel(model, find_unused_parameters=True)


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def broadcast_stop(context: DistributedContext, stop: bool) -> bool:
    """Broadcast the rank-0 early-stop decision to every worker."""

    if not context.enabled:
        return stop
    value = torch.tensor(
        [int(stop)],
        dtype=torch.int64,
        device=context.device if context.backend == "nccl" else torch.device("cpu"),
    )
    dist.broadcast(value, src=0)
    return bool(value.item())


def gather_predictions(
    context: DistributedContext,
    y_true: list[int],
    y_prob: list[float],
    loss_sum: float,
    sample_count: int,
) -> tuple[list[int], list[float], float, int]:
    """Gather variable-length predictions and sample-weighted loss globally."""

    if not context.enabled:
        return y_true, y_prob, loss_sum, sample_count

    stats = torch.tensor(
        [loss_sum, float(sample_count)], dtype=torch.float64, device=context.device
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    local_true = torch.tensor(y_true, dtype=torch.int64, device=context.device)
    local_prob = torch.tensor(y_prob, dtype=torch.float64, device=context.device)
    local_size = torch.tensor(
        [local_true.numel()], dtype=torch.int64, device=context.device
    )
    sizes = [torch.zeros_like(local_size) for _ in range(context.world_size)]
    dist.all_gather(sizes, local_size)
    max_size = max(int(item.item()) for item in sizes)
    if max_size == 0:
        raise RuntimeError("distributed evaluation produced no samples")

    padded_true = torch.zeros(max_size, dtype=torch.int64, device=context.device)
    padded_prob = torch.zeros(max_size, dtype=torch.float64, device=context.device)
    if local_true.numel():
        padded_true[: local_true.numel()] = local_true
        padded_prob[: local_prob.numel()] = local_prob
    gathered_true = [torch.empty_like(padded_true) for _ in range(context.world_size)]
    gathered_prob = [torch.empty_like(padded_prob) for _ in range(context.world_size)]
    dist.all_gather(gathered_true, padded_true)
    dist.all_gather(gathered_prob, padded_prob)

    true: list[int] = []
    prob: list[float] = []
    for size, values, probabilities in zip(sizes, gathered_true, gathered_prob):
        count = int(size.item())
        true.extend(values[:count].cpu().tolist())
        prob.extend(probabilities[:count].cpu().tolist())
    return true, prob, float(stats[0].item()), int(stats[1].item())


def finalize(context: DistributedContext) -> None:
    """Destroy a group owned by this invocation without a failure-prone barrier."""

    if context.enabled and context.owns_process_group and dist.is_initialized():
        dist.destroy_process_group()
