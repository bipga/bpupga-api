"""Exact CUDA assignment and CUDA ordered greedy decoders.

The shortest augmenting path algorithm executes in one persistent CUDA block:
cost validation, search, tie decisions, dual updates and augmentations never
leave the device. A single status transfer at the API boundary reports invalid
or infeasible input. There is no SciPy solver or CPU algorithm fallback.

This is an exact float64 solver, not a claim that assignment parallelizes well:
the persistent block uses one SM, and worst-case complexity is still cubic.
NumPy callers incur input/output transfers; tensor callers retain CUDA outputs.
Native exact objectives, rectangular/dummy padding and SciPy's JV tie rule are
preserved without cost perturbations. Greedy helpers are separate APIs and must
not replace exact matching in a method whose native decoder is exact.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np
import torch

ASSIGNMENT_IMPLEMENTATION = "cuda_float64_exact_sap_device_control_v1"
ASSIGNMENT_CPU_WORK = "CUDA launch, optional NumPy transfers, final exception status transfer"
GREEDY_IMPLEMENTATION = "cuda_stable_flat_sort_device_ordered_greedy_v1"

_LIBRARIES = {}
_BUILD_LOCK = threading.Lock()


def _selected_device(device, value=None):
    selected = torch.device(device if device is not None else
                            (value.device if isinstance(value, torch.Tensor) else "cuda:0"))
    if selected.type != "cuda":
        raise ValueError("This implementation requires CUDA; no CPU fallback is available")
    return selected


def _library(device):
    """Compile the small standalone CUDA library once per source/architecture.

    No torch C++ headers or compiler extensions are needed. The build cache is
    in /tmp and protected across processes. Source bytes, compiler path and
    compute capability determine the cache key; sealed source is never edited.
    """
    capability = torch.cuda.get_device_capability(device)
    architecture = f"{capability[0]}{capability[1]}"
    source = Path(__file__).with_name("assignment_cuda.cu")
    compiler = os.environ.get("BPUPGA_NVCC") or shutil.which("nvcc")
    if not compiler:
        raise RuntimeError("nvcc is required to build the CUDA assignment decoder")
    digest = hashlib.sha256(source.read_bytes() + os.fsencode(os.path.realpath(compiler)) +
                            architecture.encode()).hexdigest()[:20]
    key = (digest, architecture)
    with _BUILD_LOCK:
        if key in _LIBRARIES:
            return _LIBRARIES[key]
        cache = Path(tempfile.gettempdir()) / f"bpupga_cuda_assignment_{os.getuid()}"
        cache.mkdir(mode=0o700, exist_ok=True)
        target = cache / f"assignment_{digest}_sm{architecture}.so"
        with (cache / f"{digest}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                temporary = target.with_suffix(f".{os.getpid()}.so")
                command = [compiler, "-std=c++17", "-O3", "--fmad=false", "--shared",
                           "-Xcompiler=-fPIC", "-gencode",
                           f"arch=compute_{architecture},code=sm_{architecture}",
                           str(source), "-o", str(temporary)]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError("CUDA assignment compilation failed:\n" +
                                       result.stdout[-4000:] + result.stderr[-6000:])
                os.replace(temporary, target)
        library = ctypes.CDLL(str(target))
        pointer = ctypes.c_void_p
        index = ctypes.c_int64
        library.bpupga_assignment.argtypes = [
            pointer, index, index, pointer, pointer, pointer, pointer, pointer,
            pointer, pointer, pointer, pointer, pointer, pointer]
        library.bpupga_assignment.restype = ctypes.c_int
        library.bpupga_assignment_private.argtypes = [
            pointer, index, index, ctypes.c_double, pointer, pointer, pointer,
            pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer]
        library.bpupga_assignment_private.restype = ctypes.c_int
        library.bpupga_ordered_greedy.argtypes = [
            pointer, index, index, index, pointer, pointer, index, pointer,
            pointer, pointer, pointer, pointer, pointer]
        library.bpupga_ordered_greedy.restype = ctypes.c_int
        library.bpupga_cuda_error.argtypes = [ctypes.c_int]
        library.bpupga_cuda_error.restype = ctypes.c_char_p
        _LIBRARIES[key] = library
        return library


def _check_launch(library, result):
    if result:
        message = library.bpupga_cuda_error(result).decode("utf-8", errors="replace")
        raise RuntimeError(f"CUDA assignment launch failed: {message}")


def _return(value, as_numpy):
    return value.detach().cpu().numpy() if as_numpy else value


def _solve_cuda(matrix, dummy_cost=None):
    """Return row->column LAP result; optional private columns stay implicit."""
    n, real_m = matrix.shape
    m = real_m + n if dummy_cost is not None else real_m
    selected = matrix.device
    with torch.cuda.device(selected):
        library = _library(selected)
        floating = torch.empty(n + 2 * m, device=selected, dtype=torch.float64)
        u, v, distance = floating[:n], floating[n:n + m], floating[n + m:]
        indices = torch.empty(2 * n + 4 * m, device=selected, dtype=torch.int64)
        row_match = indices[:n]
        col_match = indices[n:n + m]
        predecessor = indices[n + m:n + 2 * m]
        remaining = indices[n + 2 * m:n + 3 * m]
        seen_row = indices[n + 3 * m:2 * n + 3 * m]
        seen_col = indices[2 * n + 3 * m:]
        status = torch.empty((), device=selected, dtype=torch.int32)
        tail = (u.data_ptr(), v.data_ptr(), distance.data_ptr(), row_match.data_ptr(),
                col_match.data_ptr(), predecessor.data_ptr(), remaining.data_ptr(),
                seen_row.data_ptr(), seen_col.data_ptr(), status.data_ptr(),
                torch.cuda.current_stream(selected).cuda_stream)
        if dummy_cost is None:
            result = library.bpupga_assignment(matrix.data_ptr(), n, m, *tail)
        else:
            result = library.bpupga_assignment_private(matrix.data_ptr(), n, real_m,
                                                        float(dummy_cost), *tail)
        _check_launch(library, result)
        # This is an API error boundary, not search/path/termination control.
        code = int(status.cpu())
        if code == 1:
            raise ValueError("matrix contains invalid numeric entries")
        if code == 2:
            raise ValueError("cost matrix is infeasible")
        return row_match.clone()


@torch.no_grad()
def private_dummy_assignment(cost, *, dummy_cost=0.0, device="cuda:0"):
    """Exact optional LAP with one private dummy column per real source row.

    Returns CUDA (source, target) indices including dummy targets >=cost.shape[1].
    Off-diagonal dummy costs are forbidden and are generated in the CUDA kernel;
    no augmented dense matrix is allocated. This is not rectangular full LAP.
    """
    selected = _selected_device(device, cost)
    matrix = torch.as_tensor(cost, device=selected).to(torch.float64).contiguous()
    if matrix.ndim != 2:
        raise ValueError("expected a matrix (2-D array)")
    if not np.isfinite(dummy_cost):
        raise ValueError("private dummy cost must be finite")
    n, _ = matrix.shape
    if not n:
        empty = torch.empty(0, device=selected, dtype=torch.int64)
        return empty, empty.clone()
    return torch.arange(n, device=selected), _solve_cuda(matrix, dummy_cost)


@torch.no_grad()
def linear_sum_assignment(cost, maximize=False, *, device="cuda:0"):
    """Exact rectangular LAP, solved entirely on CUDA in float64.

    The smaller side is fully matched. Positive infinity is forbidden for
    minimization; negative infinity is forbidden for maximization. NaN, opposite
    infinity and infeasible input raise ValueError. Constant-square ties select
    identity; other ties follow the same remaining-column scan as SciPy JV.
    NumPy input returns NumPy indices; tensor input returns CUDA int64 indices.
    Empty inputs and dimension/type checks use shape metadata, not CPU solving.
    """
    as_numpy = not isinstance(cost, torch.Tensor)
    selected = _selected_device(device, cost)
    if as_numpy:
        array = np.asarray(cost)
        if np.iscomplexobj(array):
            raise ValueError("cost matrix must contain real numbers")
        # Preserve input dtype in the host serialization. Numeric conversion
        # to the solver's float64 precision takes place after CUDA transfer.
        matrix = torch.as_tensor(np.ascontiguousarray(array), device=selected)
        matrix = matrix.to(dtype=torch.float64)
    else:
        if cost.is_complex():
            raise ValueError("cost matrix must contain real numbers")
        matrix = cost.detach().to(device=selected, dtype=torch.float64)
    if matrix.ndim != 2:
        raise ValueError("expected a matrix (2-D array)")
    original_rows, original_columns = matrix.shape
    if not original_rows or not original_columns:
        empty = torch.empty(0, device=selected, dtype=torch.int64)
        return _return(empty, as_numpy), _return(empty.clone(), as_numpy)
    if maximize:
        matrix = -matrix
    transposed = original_rows > original_columns
    matrix = (matrix.T if transposed else matrix).contiguous()
    n, _ = matrix.shape
    row_match = _solve_cuda(matrix)
    row_indices = torch.arange(n, device=selected, dtype=torch.int64)
    if transposed:
        order = torch.argsort(row_match, stable=True)
        output_rows, output_columns = row_match[order], row_indices[order]
    else:
        output_rows, output_columns = row_indices, row_match
    return _return(output_rows, as_numpy), _return(output_columns, as_numpy)


@torch.no_grad()
def greedy_from_order(flat_indices, rows, columns, *, device="cuda:0",
                      row_active=None, col_active=None, budget=None,
                      return_count=False):
    """Consume a CUDA-ordered flat list with one-to-one greedy acceptance.

    The device loop consumes candidates in the supplied order. Optional active
    masks live on CUDA. ``return_count=True`` returns (rows, cols, CUDA count)
    with fixed ``budget`` length and -1 padding, avoiding dynamic output shape.
    Otherwise the two output tensors are compacted on CUDA. Invalid candidate
    flat indices are skipped. This API is deliberately separate from exact LAP.
    """
    selected = _selected_device(device, flat_indices)
    rows, columns = int(rows), int(columns)
    if rows < 0 or columns < 0:
        raise ValueError("row/column counts must be nonnegative")
    maximum = min(rows, columns)
    budget = maximum if budget is None else min(maximum, int(budget))
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    order = torch.as_tensor(flat_indices, device=selected, dtype=torch.int64).flatten().contiguous()
    masks = []
    for name, mask, length in (("row_active", row_active, rows),
                               ("col_active", col_active, columns)):
        if mask is not None:
            mask = torch.as_tensor(mask, device=selected, dtype=torch.bool).contiguous()
            if mask.shape != (length,):
                raise ValueError(f"{name} must have shape ({length},)")
        masks.append(mask)
    output_rows = torch.empty(budget, device=selected, dtype=torch.int64)
    output_columns = torch.empty_like(output_rows)
    count = torch.zeros((), device=selected, dtype=torch.int64)
    if budget and order.numel():
        with torch.cuda.device(selected):
            library = _library(selected)
            row_used = torch.empty(rows, device=selected, dtype=torch.uint8)
            col_used = torch.empty(columns, device=selected, dtype=torch.uint8)
            result = library.bpupga_ordered_greedy(
                order.data_ptr(), order.numel(), rows, columns,
                masks[0].data_ptr() if masks[0] is not None else None,
                masks[1].data_ptr() if masks[1] is not None else None,
                budget, row_used.data_ptr(), col_used.data_ptr(),
                output_rows.data_ptr(), output_columns.data_ptr(), count.data_ptr(),
                torch.cuda.current_stream(selected).cuda_stream)
            _check_launch(library, result)
    elif budget:
        output_rows.fill_(-1)
        output_columns.fill_(-1)
    if return_count:
        return output_rows, output_columns, count
    keep = output_rows >= 0
    return output_rows[keep], output_columns[keep]


@torch.no_grad()
def flat_greedy(scores, maximize=True, *, device="cuda:0", row_active=None,
                col_active=None, budget=None, return_count=False):
    """CUDA stable score ordering followed by CUDA ordered greedy matching.

    Equal scores are processed by ascending original flat index. This explicit
    tie convention may differ from a native NumPy *unstable* quicksort; callers
    must declare the new GPU decoder version when that affects their method.
    """
    selected = _selected_device(device, scores)
    matrix = torch.as_tensor(scores, device=selected)
    if matrix.ndim != 2:
        raise ValueError("scores must be a matrix")
    if matrix.is_complex():
        raise ValueError("scores must contain real numbers")
    order = torch.argsort(matrix.flatten(), descending=maximize, stable=True)
    return greedy_from_order(order, *matrix.shape, device=selected,
                             row_active=row_active, col_active=col_active,
                             budget=budget, return_count=return_count)
