"""Optional execution-only PyTorch GPU allocator budget for shared devices."""
from __future__ import annotations

import os


def gpu_memory_budget_mib() -> int | None:
    raw=os.environ.get("BENCHMARK_GPU_MEMORY_BUDGET_MIB", "").strip()
    if not raw:
        return None
    if not raw.isdecimal() or int(raw)<1:
        raise ValueError("BENCHMARK_GPU_MEMORY_BUDGET_MIB must be a positive integer")
    return int(raw)


def apply_gpu_memory_budget(torch_module, device_index: int = 0) -> int | None:
    budget=gpu_memory_budget_mib()
    if budget is None:
        return None
    total=torch_module.cuda.get_device_properties(device_index).total_memory
    fraction=budget*1024**2/total
    if not 0<fraction<1:
        raise ValueError("GPU budget must be smaller than the physical device capacity")
    torch_module.cuda.set_per_process_memory_fraction(fraction,device=device_index)
    return budget
