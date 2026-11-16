"""Reports for this adaptation only; failed fits remain N/A."""
from __future__ import annotations

import csv
import io
import math
from collections import Counter
from pathlib import Path

from evaluation.runtime.io import now, read_json, write_text
from evaluation.runtime.report import flat, table


def generate(plan, state, output_root):
    output_root = Path(output_root)
    result_root = Path(plan.get("output_root", output_root))
    is_formal = result_root.resolve() != output_root.resolve()
    counts = Counter(entry["status"] for entry in state["tasks"].values())
    top = ("# OPGM unsupervised adaptation\n\n"
           f"Method: opgm_unsupervised; generated: {now()}; task status: {dict(counts)}.\n\n"
           "The adaptation uses an unsupervised objective and inputs. Native Pair F1 is the primary metric. "
           "Failed fits are N/A; means and sample standard deviations use successful runs.\n\n")
    csv_rows = []
    for dataset in plan["datasets"]:
        selected = [task for task in plan["tasks"] if task["dataset"] == dataset]
        dataset_rows, method_rows = [], []
        for condition in sorted({task["condition"] for task in selected}):
            values, statuses = [], Counter()
            for task in sorted((task for task in selected if task["condition"] == condition), key=lambda task: task["seed"]):
                entry = state["tasks"][task["id"]]
                result_path = result_root / task["relative_dir"] / "attempt_0001/result.json"
                # A worker can commit its result just before its future is reaped.
                # Until the coordinator publishes the terminal state, report no score.
                record = (read_json(result_path) if entry["status"] not in {"PENDING", "RUNNING"}
                          and result_path.exists() else None)
                if record and (record["task"] != task or record["status"] != entry["status"]):
                    raise ValueError("Result and independent state disagree: " + task["id"])
                value = record["variants"]["native"]["pair"]["f1"] if record and record["status"] == "SUCCESS" else None
                statuses[entry["status"]] += 1
                if value is not None:
                    values.append(value)
                method_rows.append([condition, task["seed"], entry["status"],
                                    f"{value:.6f}" if value is not None else None,
                                    f'[Details]({condition}/seed_{task["seed"]}/attempt_0001/metrics.md)'])
                row = dict(dataset=dataset, method=task["method"], condition=condition, seed=task["seed"],
                           status=entry["status"], result=str(result_path.relative_to(result_root)))
                if value is not None:
                    for name, variant in record["variants"].items():
                        row.update(flat({name: variant}))
                    row.update(flat(dict(runtime=record["runtime"], device=record["device"])))
                csv_rows.append(row)
            mean = sum(values) / len(values) if values else None
            sd = math.sqrt(sum((value - mean)**2 for value in values) / (len(values) - 1)) if len(values) > 1 else None
            dataset_rows.append([condition, f"{len(values)}/5", f"{mean:.6f}" if mean is not None else None,
                                 f"{sd:.6f}" if sd is not None else None, dict(statuses)])
        method_path = result_root / dataset / "opgm_unsupervised/README.md"
        write_text(method_path, f"# {dataset} / opgm_unsupervised\n\n" +
                   "Unsupervised adaptation. Scores are fractions; failed fits are N/A. Per-fit outputs are in the corresponding attempt directory.\n\n" +
                   table(["Condition", "seed", "Status", "Native PairF1", "Details"], method_rows))
        if not is_formal:
            write_text(output_root / dataset / "README.md", f"# {dataset} / OPGM unsupervised adaptation\n\n" +
                       table(["Condition", "Successful runs", "Native PairF1 mean", "sample SD", "Status"], dataset_rows) +
                       "[Per-seed results](opgm_unsupervised/README.md)\n")
        top += f"- [{dataset}]({dataset}/README.md)\n"
    top += "\n[Experiment plan](plan.json) · [Task state](state.json) · [Results CSV](native_metrics.csv)\n"
    if not is_formal:
        write_text(output_root / "README.md", top)
        write_text(output_root / "PROGRESS.md", top)
    buffer = io.StringIO()
    fields = sorted(set().union(*(row.keys() for row in csv_rows)))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)
    write_text(output_root / "native_metrics.csv", buffer.getvalue())
