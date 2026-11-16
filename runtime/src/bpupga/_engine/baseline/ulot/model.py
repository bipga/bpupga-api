"""Paper ULOT network with a leakage-safe plain-graph input adapter."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from common.result import AlignmentResult
from ..vendor.ulot.ulot.architectures import ULOT_net


OFFICIAL_COMMIT = "2bd7f5bed4dd3cbaff07eb17f39630d094ec6a37"
CHECKPOINT_SHA256 = "c31a8f2f5ec12e62ff34d9785a6817754423c41079d109c6f2814c42a26265b8"
DEFAULT_CHECKPOINT = Path(__file__).with_name("checkpoints") / "ulot_sbm_50000.pt"


@dataclass(frozen=True)
class ULOTConfig:
    """Published SBM architecture plus project-specific inference controls."""

    alpha: float = 0.5
    rho: float = 1.0
    checkpoint: str = "bundled_sbm_50000"
    verify_checkpoint: bool = True
    feature_adapter: str = "degree_quantile_onehot3"
    hidden_channels: int = 64
    hidden_channels_message: int = 16
    out_channels: int = 256
    num_layers: int = 5
    temperature: float = 3.0
    alpha_encoding_dim: int = 10
    dtype: str = "float32"
    device: str = "auto"
    seed: int = 2026
    show_progress: bool = True
    max_dense_entries: int = 25_000_000

    def validate(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must lie in (0, 1], the published training range")
        if self.feature_adapter not in {
            "degree_quantile_onehot3",
            "constant_onehot3",
        }:
            raise ValueError("unsupported topology feature adapter")
        if self.dtype != "float32":
            raise ValueError("the released ULOT checkpoint uses float32")
        if self.max_dense_entries < 1:
            raise ValueError("max_dense_entries must be positive")
        published = (64, 16, 256, 5, 10)
        actual = (
            self.hidden_channels,
            self.hidden_channels_message,
            self.out_channels,
            self.num_layers,
            self.alpha_encoding_dim,
        )
        if self.checkpoint == "bundled_sbm_50000" and actual != published:
            raise ValueError(
                f"bundled checkpoint requires architecture {published}, got {actual}"
            )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
    return torch.device(value)


def _directed_edges(graph: nx.Graph, device: torch.device) -> torch.Tensor:
    edges: list[tuple[int, int]] = []
    for source, target in graph.edges():
        edges.append((int(source), int(target)))
        if source != target:
            edges.append((int(target), int(source)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.as_tensor(edges, dtype=torch.long, device=device).T.contiguous()


def _topology_features(
    graph1: nx.Graph,
    graph2: nx.Graph,
    adapter: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert plain graphs to the released checkpoint's three channels.

    The released model was trained on three noisy SBM community indicators.
    Plain graphs do not provide them, so BPUPGA uses an explicit,
    permutation-invariant degree-tertile proxy with thresholds shared by both
    graphs. This adapter is not part of the ULOT paper.
    """

    n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
    if adapter == "constant_onehot3":
        first = np.zeros((n1, 3), dtype=np.float32)
        second = np.zeros((n2, 3), dtype=np.float32)
        first[:, 0] = 1.0
        second[:, 0] = 1.0
        return first, second
    degree1 = np.asarray([graph1.degree(i) for i in range(n1)], dtype=np.float64)
    degree2 = np.asarray([graph2.degree(i) for i in range(n2)], dtype=np.float64)
    thresholds = np.quantile(np.concatenate([degree1, degree2]), [1 / 3, 2 / 3])
    bins1 = np.searchsorted(thresholds, degree1, side="right")
    bins2 = np.searchsorted(thresholds, degree2, side="right")
    return np.eye(3, dtype=np.float32)[bins1], np.eye(3, dtype=np.float32)[bins2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ULOT:
    """Neural ULOT plan predictor for unbalanced graph optimal transport."""

    def __init__(self, config: ULOTConfig | None = None) -> None:
        self.config = config or ULOTConfig()
        self.config.validate()

    def _checkpoint_path(self) -> Path:
        path = (
            DEFAULT_CHECKPOINT
            if self.config.checkpoint == "bundled_sbm_50000"
            else Path(self.config.checkpoint).expanduser().resolve()
        )
        if not path.is_file():
            raise FileNotFoundError(f"ULOT checkpoint not found: {path}")
        if (
            self.config.verify_checkpoint
            and self.config.checkpoint == "bundled_sbm_50000"
        ):
            actual = _sha256(path)
            if actual != CHECKPOINT_SHA256:
                raise RuntimeError(f"ULOT checkpoint checksum mismatch: {actual}")
        return path

    def fit(self, graph1: nx.Graph, graph2: nx.Graph) -> AlignmentResult:
        features = _topology_features(graph1, graph2, self.config.feature_adapter)
        return self.fit_with_features(graph1, graph2, *features)

    def fit_with_features(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph,
        features1: np.ndarray,
        features2: np.ndarray,
    ) -> AlignmentResult:
        """Run the paper model with explicit three-channel node features."""

        if graph1.number_of_nodes() == 0 or graph2.number_of_nodes() == 0:
            raise ValueError("ULOT cannot align empty graphs")
        for graph, label in ((graph1, "G1"), (graph2, "G2")):
            if set(graph.nodes()) != set(range(graph.number_of_nodes())):
                raise ValueError(f"{label} must use consecutive integer node ids")
        n1, n2 = graph1.number_of_nodes(), graph2.number_of_nodes()
        if n1 * n2 > self.config.max_dense_entries:
            raise ValueError(
                f"ULOT cross-attention needs {n1*n2:,} entries, exceeding "
                f"max_dense_entries={self.config.max_dense_entries:,}"
            )
        features1 = np.asarray(features1, dtype=np.float32)
        features2 = np.asarray(features2, dtype=np.float32)
        if features1.shape != (n1, 3) or features2.shape != (n2, 3):
            raise ValueError(
                "released SBM checkpoint requires feature shapes (n1, 3), (n2, 3)"
            )

        start = time.perf_counter()
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        device = _resolve_device(self.config.device)
        model = ULOT_net(
            3,
            self.config.hidden_channels,
            self.config.hidden_channels_message,
            self.config.out_channels,
            self.config.num_layers,
            self.config.temperature,
            self.config.alpha_encoding_dim,
        ).to(device)
        checkpoint_path = self._checkpoint_path()
        checkpoint = torch.load(
            checkpoint_path, weights_only=True, map_location=device
        )
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
        model.eval()

        x1 = torch.as_tensor(features1, device=device)
        x2 = torch.as_tensor(features2, device=device)
        edge1 = _directed_edges(graph1, device)
        edge2 = _directed_edges(graph2, device)
        batch1 = torch.zeros(n1, dtype=torch.long, device=device)
        batch2 = torch.zeros(n2, dtype=torch.long, device=device)
        rho1 = torch.full((n1, 1), self.config.rho, device=device)
        rho2 = torch.full((n2, 1), self.config.rho, device=device)
        alpha1 = torch.full((n1, 1), self.config.alpha, device=device)
        alpha2 = torch.full((n2, 1), self.config.alpha, device=device)
        with torch.inference_mode():
            plan, mask1, mask2 = model(
                x1, x2, edge1, edge2, batch1, batch2,
                rho1, rho2, alpha1, alpha2,
            )
        score = plan[0][mask1[0]][:, mask2[0]].cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(score)) or np.any(score < 0):
            raise RuntimeError("ULOT produced an invalid transport plan")
        sources, targets = linear_sum_assignment(-score)
        pairs = [(int(i), int(j)) for i, j in zip(sources, targets)]
        return AlignmentResult(
            method="ULOT",
            predicted_pairs=pairs,
            runtime_sec=float(time.perf_counter() - start),
            score_matrix=score,
            larger_is_better=True,
            metadata={
                "config": asdict(self.config),
                "device": str(device),
                "core_method": "published neural ULOT cross-attention plan predictor",
                "paper": "NeurIPS 2025, arXiv:2506.12025v3",
                "official_repository": "https://github.com/smazelet/ULOT",
                "official_commit": OFFICIAL_COMMIT,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "checkpoint_git_blob": "3a1c784afa19cf77c7e77bab03bb163e1b98db91",
                "training_split_defined_by_author": {"train": 0.6, "validation": 0.2, "test": 0.2},
                "evaluation_graphs_used_for_training": False,
                "checkpoint_training_domain": (
                    "released 50k random-size three-community SBM model"
                ),
                "feature_adapter": self.config.feature_adapter,
                "feature_adapter_scope": (
                    "BPUPGA plain-graph adaptation; not in ULOT paper"
                ),
                "prediction_rule": (
                    "maximum-weight assignment on predicted transport"
                ),
                "partial_adaptation": (
                    "supplementary method-aware decoder only; native uses Hungarian"
                ),
                "ground_truth_used_during_inference": False,
                "ground_truth_cardinality_used_during_inference": False,
            },
        )
