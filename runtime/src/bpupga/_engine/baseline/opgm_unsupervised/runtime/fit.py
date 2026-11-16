"""Graph-only child; model fitting completes and is sealed before GT scoring."""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import baseline  # Activates the bundled common result contract.
from evaluation.runtime.graph_packet import load_graphs
from evaluation.runtime.io import now, read_json, sha256, verify_files, write_json

METHOD = "opgm_unsupervised"
PACKET_FIELDS = {"method", "parameters", "seed", "graphs_sha256", "source_hashes", "gpu_uuid"}


def validate_packet(packet):
    if set(packet) != PACKET_FIELDS or packet["method"] != METHOD:
        raise ValueError("Unexpected graph-only packet fields or method")
    if not isinstance(packet["parameters"], dict):
        raise ValueError("Model parameters must be a fixed configuration dictionary")
    if any(name.lower() in {"gt", "ground_truth", "true_k", "true_overlap", "rho", "data_root", "dataset"}
           for name in packet["parameters"]):
        raise ValueError("Ground truth, overlap and dataset fields are forbidden")


def run(packet_path):
    # Import the model only after the packet's structure and source seals pass.
    packet_path = Path(packet_path)
    packet = read_json(packet_path)
    validate_packet(packet)
    verify_files(ROOT, packet["source_hashes"])
    graph_path = packet_path.parent / "graphs.npz"
    if sha256(graph_path) != packet["graphs_sha256"]:
        raise ValueError("Graph packet SHA256 mismatch")
    import numpy as np
    import torch
    from common.seed import seed_everything
    from baseline.opgm_unsupervised.model import construct

    graphs = load_graphs(graph_path)
    output = packet_path.parent / "fit"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    seed_everything(packet["seed"])
    torch.manual_seed(packet["seed"])
    gpu = packet["gpu_uuid"]
    if gpu:
        if not (torch.cuda.is_available() and torch.cuda.device_count() == 1
                and os.environ.get("CUDA_VISIBLE_DEVICES") == gpu):
            raise RuntimeError("Worker must see exactly the assigned GPU UUID")
        torch.cuda.set_per_process_memory_fraction(.90, 0)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, config = construct(packet["parameters"], packet["seed"], "cuda:0" if gpu else "cpu")
    result = model.fit(*graphs)
    if gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if result.method != METHOD:
        raise ValueError(f"Unexpected result method: {result.method}")
    result.validate_shapes(*map(len, graphs))
    if hasattr(model, "export_checkpoint"):
        torch.save(model.export_checkpoint(), output / "model_state.pt")
    if result.score_matrix is None and result.cost_matrix is None:
        raise ValueError("The same-fit native continuous matrix is required")
    write_json(output / "native_pairs_local.json",
               dict(pairs=result.predicted_pairs, index_space="consecutive_local"))
    write_json(output / "algorithm_diagnostics.json", result.metadata)
    for name in ("score_matrix", "cost_matrix", "source_embeddings", "target_embeddings"):
        values = getattr(result, name)
        if values is not None:
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite " + name)
            np.save(output / (name + ".npy"), values, allow_pickle=False)
    for name, values in (result.partial_decoder_data or {}).items():
        if not name.isidentifier() or not np.isfinite(values).all():
            raise FloatingPointError("Invalid native-state artifact")
        np.save(output / ("native_state_" + name + ".npy"), values, allow_pickle=False)
    for name, values in result.stage_score_matrices.items():
        if not name.isidentifier() or not np.isfinite(values).all():
            raise FloatingPointError("Invalid same-fit stage artifact")
        np.save(output / ("stage_" + name + ".npy"), values, allow_pickle=False)
    write_json(output / "fit_metadata.json", dict(
        method=METHOD, parameters=config, seed=packet["seed"], larger_is_better=result.larger_is_better,
        method_metadata=result.metadata, stage_diagnostics_metadata=result.stage_diagnostics_metadata,
        runtime=dict(native_reported_seconds=result.runtime_sec, fit_wall_seconds=elapsed,
                     fit_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                     torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2 if gpu else None,
                     torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2 if gpu else None),
        device=dict(gpu_uuid=gpu, name=torch.cuda.get_device_name() if gpu else "CPU",
                    allocator_fraction=.90 if gpu else None),
        inference_contract=dict(GT_in_packet=False, true_overlap_in_packet=False, true_K_in_packet=False,
                                dataset_path_in_packet=False, graph_only_child=True,
                                model_identity=METHOD, supervised_checkpoint_used=False)))
    verify_files(ROOT, packet["source_hashes"])
    write_json(output / "fit_seal.json", {path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    print("Independent unsupervised fit sealed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.packet)
    except BaseException as error:
        message = str(error).lower()
        # CUDA OOM text and MemoryError are evidence; an unexplained SIGKILL is not.
        status = ("OOM" if isinstance(error, MemoryError) or "out of memory" in message or "unable to allocate" in message
                  else "NUMERICAL_ERROR" if isinstance(error, FloatingPointError) or "non-finite" in message
                  else "ERROR")
        write_json(args.packet.parent / "fit_failure.json", dict(
            status=status, error=str(error), type=type(error).__name__, traceback=traceback.format_exc(), at=now()))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
