"""Independent 115-input OPGM adaptation campaign. Preparation starts no fit.

The formal scheduler waits for every original and Arxiv GPU slot to become
terminal before claiming the same four cards. Existing attempts are immutable;
interrupted attempts require an audit rather than an automatic retry.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Coordinator and CPU prechecks have a single BLAS/OMP thread. Formal workers
# use the same conservative setting; they never claim an extra CPU fit slot.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import baseline
from evaluation.runtime.io import DATA, GPU_UUIDS, now, read_json, sha256, verify_files, write_json, write_text
from evaluation.runtime.report import task_markdown
from baseline.opgm_unsupervised.runtime.report import generate

METHOD = "opgm_unsupervised"
PACKAGE = ROOT / "baseline/opgm_unsupervised"
OLD_OUTPUT = ROOT / "evaluation/output"
ARXIV = OLD_OUTPUT / "_extensions/arxiv_261029"
FORMAL_OUTPUT = OLD_OUTPUT / "_extensions/opgm_unsupervised_261029"
TEST_OUTPUT = ROOT / "evaluation/test/opgm_unsupervised_261029"
SOURCE_PLANS = (OLD_OUTPUT / "plan.json", ARXIV / "plan.json")
TERMINAL = {"SUCCESS", "OOM", "TIMEOUT", "NUMERICAL_ERROR", "ERROR", "EVALUATION_ERROR", "PROTOCOL_INCOMPATIBLE"}
UPSTREAM_GPU_METHODS = {"alpine", "gradalign_plus", "slotalign", "ulot"}
PARAMETERS = read_json(PACKAGE / "DEFAULT_PARAMETERS.json")


def allowed_output(path):
    path = Path(path).resolve()
    if not any(path == root.resolve() or root.resolve() in path.parents for root in (FORMAL_OUTPUT, TEST_OUTPUT)):
        raise ValueError("Output must stay inside this adaptation's formal or test directory")
    return path


def results_root(metadata_root):
    metadata_root = allowed_output(metadata_root)
    return OLD_OUTPUT if metadata_root == FORMAL_OUTPUT.resolve() else metadata_root


def source_plans():
    plans = [read_json(path) for path in SOURCE_PLANS]
    if [len(plan["tasks"]) for plan in plans] != [1080, 300]:
        raise ValueError("Expected unchanged original1080 and Arxiv300 source plans")
    frozen = plans[0]["source_hashes"]
    if len(frozen) != 161 or plans[1]["source_hashes"] != frozen:
        raise ValueError("Original and Arxiv must share the unchanged161-source manifest")
    verify_files(ROOT, frozen)
    expected_uuids = {str(key): value for key, value in GPU_UUIDS.items()}
    if len(expected_uuids) != 4 or set(GPU_UUIDS) != {4, 5, 6, 7}:
        raise ValueError("Expected physical GPUs4–7 only")
    if any(plan["gpu_uuids"] != expected_uuids for plan in plans):
        raise ValueError("Source-plan GPU UUID allowlist changed")
    return plans


def source_hashes(plans):
    # The frozen161 files are reused read-only. Only this package adds code.
    hashes = dict(plans[0]["source_hashes"])
    paths = [path for path in PACKAGE.rglob("*") if path.is_file()
             and not {"evidence", "__pycache__", "dependency_dist", "tests"}.intersection(path.relative_to(PACKAGE).parts)
             and (path.suffix in {".py", ".so", ".json"} or path.name == "LICENSE")]
    if not (PACKAGE / "model.py").exists():
        raise FileNotFoundError("Independent model.py is not ready for source freezing")
    for path in sorted(paths):
        if "__pycache__" not in path.parts:
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


def make_plan(metadata_root):
    plans = source_plans()
    config = copy.deepcopy(plans[0]["baseline_configs"])
    # Verified implementation: frozen decoders.method_pairs explicitly supports
    # method_native_partial and checks that the native pair set is injective.
    if config["partial_adapters"]["opgm"].get("strategy") != "method_native_partial":
        raise ValueError("Frozen OPGM partial adapter strategy changed")
    config["partial_adapters"][METHOD] = copy.deepcopy(config["partial_adapters"]["opgm"])
    tasks, inputs = [], []
    for source_plan, expected in zip(plans, (90, 25)):
        source_tasks = [task for task in source_plan["tasks"] if task["method"] == "opgm"]
        if len(source_tasks) != expected:
            raise ValueError("Original supervised OPGM slots changed")
        input_lookup = {(item["dataset"], item["condition"], item["seed"]): item for item in source_plan["inputs"]}
        for old in source_tasks:
            dataset, condition, seed = old["dataset"], old["condition"], old["seed"]
            tasks.append(dict(id=f"{dataset}__{condition}__{METHOD}__{seed}", dataset=dataset,
                              condition=condition, method=METHOD, seed=seed,
                              instance=copy.deepcopy(old["instance"]), parameters=copy.deepcopy(PARAMETERS),
                              profile="fixed_unsupervised_adaptation_261029_r1",
                              relative_dir=f"{dataset}/{METHOD}/{condition}/seed_{seed}",
                              original_supervised_task_id=old["id"]))
            inputs.append(copy.deepcopy(input_lookup[(dataset, condition, seed)]))
    groups = {}
    for task in tasks:
        groups.setdefault((task["dataset"], task["condition"]), []).append(task["seed"])
    if len(tasks) != 115 or len({task["id"] for task in tasks}) != 115 or len(groups) != 23:
        raise ValueError("Adaptation must have exactly115 inputs across23 conditions")
    if any(sorted(seeds) != list(range(2026, 2031)) for seeds in groups.values()):
        raise ValueError("Every condition must retain seeds2026–2030")
    return dict(schema="opgm_unsupervised_flat_layout_261029_r2", created_at=now(), tasks=tasks, inputs=inputs,
                output_root=str(results_root(metadata_root)), metadata_root=str(Path(metadata_root).resolve()),
                datasets=sorted({task["dataset"] for task in tasks}), methods=[METHOD],
                source_hashes=source_hashes(plans), frozen_baseline_source_hashes=plans[0]["source_hashes"],
                source_plan_hashes={str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PLANS},
                baseline_configs=config, gpu_uuids=plans[0]["gpu_uuids"],
                limits=dict(fit_timeout_seconds=7200, fit_tree_rss_gib=64, gpu_allocator_fraction=.90,
                            max_total_gpus=4, max_fits_per_gpu=1, threads_per_fit=1, cpu_fit_slots=0),
                protocol=dict(fully_unsupervised=True, independent_model=True, ground_truth_only_after_fit_seal=True,
                              original_1380_slots_modified=False, original_115_NA_replaced=False,
                              old_V1_results_reused=False, failed_fits_are_NA=True,
                              source_input_checks="Inherited source-plan input hashes; verified again before and after every fit"),
                scheduling=dict(upstream_gpu_queues_must_be_terminal=True,
                                upstream_gpu_slot_counts=[365, 100],
                                gpu_idle_rule="No compute PID, memory<512MiB, no upstream pending/running slot, compatible lease",
                                no_external_process_termination=True, failed_attempts_automatically_retried=False))


def check_plan(plan, output_root):
    verify_files(output_root, read_json(output_root / "plan_seal.json"))
    plans = source_plans()
    if plan["source_hashes"] != source_hashes(plans):
        raise ValueError("Independent source binding changed; use a separately audited version")
    verify_files(ROOT, plan["source_hashes"])
    verify_files(ROOT, plan["source_plan_hashes"])
    if len(plan["tasks"]) != 115 or len({task["id"] for task in plan["tasks"]}) != 115:
        raise ValueError("Invalid independent task cardinality")
    for task in plan["tasks"]:
        if task["method"] != METHOD or task["seed"] not in range(2026, 2031):
            raise ValueError("Unexpected independent method or seed")
        target = (results_root(output_root) / task["relative_dir"]).resolve()
        target.relative_to(results_root(output_root))
        if Path(task["relative_dir"]).parts[:2] != (task["dataset"], METHOD):
            raise ValueError("Only this adaptation's dataset/method directories are writable")
        (DATA / task["instance"]["root"]).resolve().relative_to(DATA.resolve())


def task_dir(output_root, task):
    return results_root(output_root) / task["relative_dir"] / "attempt_0001"


def terminal_entry(record):
    if record["status"] not in TERMINAL:
        raise ValueError("Nonterminal fit result")
    return dict(status=record["status"], attempt=1, completed_at=record["completed_at"])


def reconcile(plan, output_root):
    path = output_root / "state.json"
    state = read_json(path) if path.exists() else dict(
        schema="opgm_unsupervised_state_261029_r1", status="prepared", created_at=now(),
        tasks={task["id"]: dict(status="PENDING", attempt=1) for task in plan["tasks"]})
    if set(state["tasks"]) != {task["id"] for task in plan["tasks"]}:
        raise ValueError("Independent state task IDs differ from frozen plan")
    for task in plan["tasks"]:
        attempt = task_dir(output_root, task)
        if (attempt / "result.json").exists():
            record = read_json(attempt / "result.json")
            if record["task"] != task:
                raise ValueError("Existing result differs from the frozen independent task")
            verify_files(attempt, read_json(attempt / "artifact_hashes.json"))
            if record["status"] == "SUCCESS":
                verify_files(attempt / "fit", read_json(attempt / "fit/fit_seal.json"))
            state["tasks"][task["id"]] = terminal_entry(record)
        elif attempt.exists() or state["tasks"][task["id"]]["status"] != "PENDING":
            raise RuntimeError("Interrupted attempt requires process/attempt audit; refusing retry: " + task["id"])
    state.update(updated_at=now(), counts=dict(Counter(entry["status"] for entry in state["tasks"].values())))
    return state


def prepare(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "plan.json"
    if path.exists():
        plan = read_json(path)
        check_plan(plan, output_root)
    else:
        plan = make_plan(output_root)
        write_json(path, plan)
        write_json(output_root / "source_hashes.json", plan["source_hashes"])
        write_json(output_root / "plan_seal.json", {"plan.json": sha256(path)})
    state = reconcile(plan, output_root)
    write_json(output_root / "state.json", state)
    generate(plan, state, output_root)
    return plan, state


def uses_upstream_gpu(task):
    return task["method"] in UPSTREAM_GPU_METHODS or (task["method"] == "fugal" and task["dataset"] == "Oregon-1")


def upstream_observation(plans, audit_results=False):
    """Pending GPU work also blocks dispatch, preventing an idle-card race."""
    observations = []
    for source_path, plan, expected in zip(SOURCE_PLANS, plans, (365, 100)):
        state = read_json(source_path.parent / "state.json")
        tasks = [task for task in plan["tasks"] if uses_upstream_gpu(task)]
        if len(tasks) != expected:
            raise ValueError("Unexpected upstream GPU task count")
        if set(state["tasks"]) != {task["id"] for task in plan["tasks"]}:
            raise ValueError("Upstream state does not match its own plan")
        counts, running_uuids = Counter(), []
        for task in tasks:
            entry = state["tasks"][task["id"]]
            counts[entry["status"]] += 1
            if entry["status"] == "RUNNING":
                running_uuids.append(entry.get("gpu_uuid", GPU_UUIDS.get(entry.get("gpu"))))
            if entry["status"] in TERMINAL and audit_results:
                result = read_json(OLD_OUTPUT / task["relative_dir"] / "attempt_0001/result.json")
                if result["task"] != task or result["status"] != entry["status"]:
                    raise ValueError("Upstream terminal result does not prove queue completion")
        observations.append(dict(plan=str(source_path.relative_to(ROOT)), expected_gpu_slots=expected,
                                 counts=dict(counts), running_gpu_uuids=running_uuids,
                                 all_gpu_slots_terminal=sum(counts[status] for status in TERMINAL) == expected))
    return dict(at=now(), queues=observations,
                all_upstream_gpu_slots_terminal=all(item["all_gpu_slots_terminal"] for item in observations))


def gpu_snapshot():
    memory_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15)
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15)
    if memory_query.returncode or process_query.returncode:
        raise RuntimeError("nvidia-smi query failed; no dispatch permitted")
    memory, processes = {}, {}
    for line in memory_query.stdout.splitlines():
        gpu, used = [part.strip() for part in line.split(",")]
        memory[gpu] = int(used)
    for line in process_query.stdout.splitlines():
        if line.strip():
            gpu, pid = [part.strip() for part in line.split(",")]
            processes.setdefault(gpu, []).append(int(pid))
    if not set(GPU_UUIDS.values()).issubset(memory):
        raise RuntimeError("Allowed GPU UUID missing from query")
    return memory, processes


def claim_gpu(output_root, gpu):
    """Honor an existing Arxiv lease without creating or changing its files."""
    handles = []
    own = output_root / "_gpu_leases" / f"gpu_{GPU_UUIDS[gpu]}.lock"
    own.parent.mkdir(exist_ok=True)
    previous = ARXIV / own.name
    try:
        if previous.exists():
            handles.append(previous.open("r"))
            fcntl.flock(handles[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(own.open("a"))
        fcntl.flock(handles[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except BlockingIOError:
        for handle in handles:
            handle.close()
        return None


def worker_env(seed, gpu):
    environment = dict(os.environ)
    environment.update(PYTHONHASHSEED=str(seed), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUDA_VISIBLE_DEVICES=GPU_UUIDS[gpu] if gpu is not None else "",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1", MPLBACKEND="Agg", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    return environment


def kill_own_fit(process):
    # Popen creates a fresh session. No process outside that fit group is killed.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_task(task, plan, gpu, output_root):
    import psutil
    from common.dataset import load_dataset
    from evaluation.runtime.graph_packet import save_graphs
    from evaluation.runtime.evaluate import evaluate_outputs

    output_root = allowed_output(output_root)
    output = task_dir(output_root, task)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(task=task, attempt=1, started_at=now(), status="RUNNING", gpu=gpu)
    write_json(output / "started.json", record)
    neutral = output_root / "_inference" / uuid.uuid4().hex
    neutral.mkdir(parents=True, exist_ok=False)
    try:
        verify_files(ROOT, plan["source_hashes"])
        data_root = DATA / task["instance"]["root"]
        verify_files(data_root, task["instance"]["sha256"])
        dataset = load_dataset(data_root)
        save_graphs(neutral / "graphs.npz", (dataset.graph1, dataset.graph2))
        packet = dict(method=METHOD, parameters=task["parameters"], seed=task["seed"],
                      graphs_sha256=sha256(neutral / "graphs.npz"), source_hashes=plan["source_hashes"],
                      gpu_uuid=GPU_UUIDS[gpu] if gpu is not None else None)
        write_json(neutral / "packet.json", packet)
        write_json(output / "inference_contract.json", dict(packet=str(neutral / "packet.json"),
                   GT_in_packet=False, true_K_in_packet=False, true_overlap_in_packet=False,
                   dataset_path_in_packet=False, graph_packet_sha256=packet["graphs_sha256"],
                   packet_sha256=sha256(neutral / "packet.json"), model_identity=METHOD))
        peak_rss, reason = 0, None
        with (output / "fit.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "baseline.opgm_unsupervised.runtime.fit", "--packet", str(neutral / "packet.json")],
                cwd=ROOT, env=worker_env(task["seed"], gpu), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        rss = sum(child.memory_info().rss for child in [parent] + parent.children(recursive=True) if child.is_running())
                        peak_rss = max(peak_rss, rss)
                    except psutil.NoSuchProcess:
                        pass
                    if peak_rss > plan["limits"]["fit_tree_rss_gib"] * 1024**3:
                        reason = "OOM"
                        kill_own_fit(process)
                        break
                    if time.monotonic() - started > plan["limits"]["fit_timeout_seconds"]:
                        reason = "TIMEOUT"
                        kill_own_fit(process)
                        break
                    time.sleep(1)
            except BaseException:
                if process.poll() is None:
                    kill_own_fit(process)
                raise
        if (neutral / "fit").exists():
            (neutral / "fit").rename(output / "fit")
        write_json(output / "resource_monitor.json", dict(
            observed_fit_tree_peak_rss_mib=peak_rss / 1024**2, returncode=process.returncode,
            termination=reason, poll_seconds=1, tree_rss_limit_gib=64, timeout_seconds=7200))
        if reason:
            record.update(status=reason, error="Explicit resource budget exceeded; no approximate fallback")
        elif process.returncode:
            failure = read_json(neutral / "fit_failure.json") if (neutral / "fit_failure.json").exists() else dict(
                status="ERROR", error=f"Child exit{process.returncode}; no OOM evidence")
            write_json(output / "fit_failure.json", failure)
            record.update(status=failure["status"], error=failure["error"])
        else:
            fit_seal = read_json(output / "fit/fit_seal.json")
            verify_files(output / "fit", fit_seal)
            evaluation_started = time.monotonic()
            variants, metadata = evaluate_outputs(METHOD, output / "fit", dataset, plan["baseline_configs"])
            variants["native"]["version"] = "native_opgm_unsupervised_261029_r1"
            native_path = output / "native_predicted_pairs.json"
            native_record = read_json(native_path)
            native_record["version"] = variants["native"]["version"]
            write_json(native_path, native_record)
            # The generic evaluator must not mutate this adaptation's sealed fit.
            verify_files(output / "fit", fit_seal)
            verify_files(data_root, task["instance"]["sha256"])
            verify_files(ROOT, plan["source_hashes"])
            record.update(status="SUCCESS", variants=variants, dataset_summary=dataset.summary(),
                          algorithm_diagnostics=metadata["method_metadata"], device=metadata["device"],
                          runtime=dict(metadata["runtime"], evaluation_wall_seconds=time.monotonic() - evaluation_started,
                                       end_to_end_seconds=time.monotonic() - started,
                                       observed_fit_tree_peak_rss_mib=peak_rss / 1024**2),
                          GT_only_evaluation_after_fit_seal=True)
    except BaseException as error:
        record.update(status="EVALUATION_ERROR" if (output / "fit/fit_seal.json").exists() else "ERROR",
                      error=str(error), traceback=traceback.format_exc())
        write_text(output / "exception.log", record["traceback"])
    record.update(completed_at=now(), end_to_end_seconds=time.monotonic() - started)
    if "algorithm_diagnostics" not in record and (output / "fit/algorithm_diagnostics.json").exists():
        record["algorithm_diagnostics"] = read_json(output / "fit/algorithm_diagnostics.json")
    write_text(output / "metrics.md", "OPGM with an unsupervised objective and inputs.\n\n" + task_markdown(record))
    write_json(output / "artifact_hashes.json", {str(path.relative_to(output)): sha256(path)
                                               for path in output.rglob("*") if path.is_file()})
    # result.json is the terminal commit marker, written after all other artifacts.
    write_json(output / "result.json", record)
    return record


def event(output_root, kind, **fields):
    with (output_root / "resource_journal.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(at=now(), event=kind, **fields), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(plan, state, output_root, poll_seconds, report_seconds):
    check_plan(plan, output_root)
    plans = source_plans()
    state.update(scheduler_pid=os.getpid(), status="waiting_for_upstream_gpu_queues", updated_at=now())
    write_json(output_root / "state.json", state)
    stop = {"requested": False, "signal": None}

    def request_stop(signum, _frame):
        stop.update(requested=True, signal=signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    active, last_report, upstream_audited = {}, 0, False
    tasks = sorted(plan["tasks"], key=lambda task: (task["seed"], task["dataset"], task["condition"]))
    pool = ThreadPoolExecutor(max_workers=4)
    event(output_root, "COORDINATOR_START", pid=os.getpid(), gpu_allowlist=GPU_UUIDS)
    try:
        while True:
            for future, (task, gpu, handles) in list(active.items()):
                if not future.done():
                    continue
                try:
                    state["tasks"][task["id"]] = terminal_entry(future.result())
                except BaseException:
                    state["tasks"][task["id"]] = dict(status="COORDINATOR_ERROR", attempt=1, error=traceback.format_exc())
                for handle in handles:
                    handle.close()
                del active[future]
                event(output_root, "RELEASE", task=task["id"], gpu=gpu, status=state["tasks"][task["id"]]["status"])
            observation = upstream_observation(plans)
            upstream_ready = observation["all_upstream_gpu_slots_terminal"]
            if upstream_ready and not upstream_audited:
                observation = upstream_observation(plans, audit_results=True)
                write_json(output_root / "upstream_gpu_terminal_proof.json", observation)
                upstream_audited = True
            pending = [task for task in tasks if state["tasks"][task["id"]]["status"] == "PENDING"]
            busy = {gpu for task, gpu, handles in active.values()}
            if upstream_ready and pending and not stop["requested"]:
                memory, processes = {}, {}
                try:
                    memory, processes = gpu_snapshot()
                    observation.update(gpu_memory_mib=memory, gpu_compute_pids=processes)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    observation["gpu_query_error"] = str(error)
                for task in pending:
                    handles, assigned = None, None
                    for gpu, gpu_uuid in GPU_UUIDS.items():
                        if gpu in busy or memory.get(gpu_uuid, 10**9) >= 512 or processes.get(gpu_uuid):
                            continue
                        handles = claim_gpu(output_root, gpu)
                        if handles is None:
                            continue
                        # Recheck after the lease, including upstream pending work.
                        try:
                            current = upstream_observation(plans)
                            latest_memory, latest_processes = gpu_snapshot()
                            ready = (current["all_upstream_gpu_slots_terminal"]
                                     and latest_memory.get(gpu_uuid, 10**9) < 512 and not latest_processes.get(gpu_uuid))
                        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
                            ready = False
                        if not ready:
                            for handle in handles:
                                handle.close()
                            handles = None
                            continue
                        assigned = gpu
                        break
                    if assigned is None:
                        break
                    if task_dir(output_root, task).exists():
                        for handle in handles:
                            handle.close()
                        raise RuntimeError("Refusing existing independent attempt: " + task["id"])
                    state["tasks"][task["id"]] = dict(status="RUNNING", attempt=1, started_at=now(),
                                                     gpu=assigned, gpu_uuid=GPU_UUIDS[assigned])
                    write_json(output_root / "state.json", state)
                    event(output_root, "CLAIM", task=task["id"], gpu=assigned, gpu_uuid=GPU_UUIDS[assigned],
                          total_adaptation_gpu_fits=len(busy) + 1, all_upstream_gpu_slots_terminal=True)
                    try:
                        future = pool.submit(run_task, task, plan, assigned, output_root)
                    except BaseException:
                        for handle in handles:
                            handle.close()
                        raise
                    active[future] = (task, assigned, handles)
                    busy.add(assigned)
            pending = [task for task in tasks if state["tasks"][task["id"]]["status"] == "PENDING"]
            complete = not active and not pending and all(entry["status"] in TERMINAL for entry in state["tasks"].values())
            state.update(updated_at=now(), counts=dict(Counter(entry["status"] for entry in state["tasks"].values())),
                         status="complete" if complete else "draining" if stop["requested"] else
                         "running" if upstream_ready else "waiting_for_upstream_gpu_queues",
                         resource_observation=observation)
            write_json(output_root / "state.json", state)
            if time.monotonic() - last_report >= report_seconds or complete:
                generate(plan, state, output_root)
                last_report = time.monotonic()
            if complete:
                check_plan(plan, output_root)
                write_json(output_root / "COMPLETED.json", dict(
                    at=now(), counts=state["counts"], all_115_slots_terminal=True, source_hashes_verified=True,
                    old_results_reused=False, original_supervised_NA_replaced=False,
                    independent_acceptance_required=True))
                event(output_root, "COORDINATOR_COMPLETE", counts=state["counts"])
                return 0
            if not active and not pending:
                raise RuntimeError("Unresolved coordinator error; audit required")
            if stop["requested"] and not active:
                state.update(status="interrupted", updated_at=now(), received_signal=stop["signal"])
                write_json(output_root / "state.json", state)
                generate(plan, state, output_root)
                return 128 + stop["signal"]
            time.sleep(poll_seconds)
    finally:
        # Monitor only our fits to their existing limits; leave upstream jobs alone.
        pool.shutdown(wait=True)
        for task, gpu, handles in active.values():
            for handle in handles:
                handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=FORMAL_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resource-check-only", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true", help="Test outputs only; one CPU thread, no GPU")
    parser.add_argument("--datasets", nargs="+", default=["fbtw"])
    parser.add_argument("--conditions", nargs="+", default=["rho_s1_rho_t1", "rho_s0p6_rho_t0p6"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--report-seconds", type=float, default=20.)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60 or args.report_seconds <= 0:
        parser.error("Poll must be0–60 seconds and report interval positive")
    output_root = allowed_output(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, state = prepare(output_root)
        if args.cpu_smoke:
            if not (output_root == TEST_OUTPUT.resolve() or TEST_OUTPUT.resolve() in output_root.parents):
                raise ValueError("CPU smoke must stay in the isolated test directory")
            selected = [task for task in plan["tasks"] if task["dataset"] in args.datasets
                        and task["condition"] in args.conditions and task["seed"] == args.seed]
            if not selected:
                raise ValueError("No requested smoke task exists")
            state.update(status="testing", updated_at=now())
            for task in selected:
                if state["tasks"][task["id"]]["status"] != "PENDING":
                    raise ValueError("Refusing to repeat an existing smoke attempt")
                state["tasks"][task["id"]] = dict(status="RUNNING", attempt=1, gpu=None, started_at=now())
                write_json(output_root / "state.json", state)
                record = run_task(task, plan, None, output_root)
                state["tasks"][task["id"]] = terminal_entry(record)
                state.update(updated_at=now(), status="smoke_complete",
                             counts=dict(Counter(entry["status"] for entry in state["tasks"].values())))
                write_json(output_root / "state.json", state)
                generate(plan, state, output_root)
                print(json.dumps(dict(task=task["id"], status=record["status"],
                                      native_pair=record.get("variants", {}).get("native", {}).get("pair")), ensure_ascii=False), flush=True)
            return 0 if all(state["tasks"][task["id"]]["status"] == "SUCCESS" for task in selected) else 1
        if args.resource_check_only:
            observation = upstream_observation(source_plans())
            memory, processes = gpu_snapshot()
            observation.update(gpu_memory_mib=memory, gpu_compute_pids=processes, fits_started=0)
            write_json(output_root / "resource_precheck.json", observation)
            print(json.dumps(observation, ensure_ascii=False), flush=True)
            return 0
        if args.prepare_only or args.report_only:
            print(json.dumps(dict(status=state["status"], task_slots=len(plan["tasks"]), fits_started=0,
                                  plan=str(output_root / "plan.json"), source_files=len(plan["source_hashes"])),
                             ensure_ascii=False), flush=True)
            return 0
        return run(plan, state, output_root, args.poll_seconds, args.report_seconds)


if __name__ == "__main__":
    sys.exit(main())
