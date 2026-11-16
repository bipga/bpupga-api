"""CPU adapters backed exclusively by bundled sources."""
from dataclasses import asdict, fields
import importlib


def construct(method, parameters, seed=2026, device="cpu", dataset_family=""):
    values = dict(parameters)
    if method == "alpine":
        from baseline.alpine_author_cpu import AuthorCPUAlpine
        defaults = dict(mu=1., niter=10, inner_iterations=10, weight=2,
                        sinkhorn_reg=1., sinkhorn_iterations=500,
                        sinkhorn_tolerance=1e-5, dtype="float64", device="cpu")
        if values.keys() - defaults.keys():
            raise ValueError("Unknown ALPINE parameter: " + ", ".join(sorted(values.keys() - defaults.keys())))
        if values.get("niter", 10) != 10 or values.get("device", "cpu") not in {"cpu", "auto"}:
            raise ValueError("The CPU author implementation fixes ten iterations and a CPU device")
        config = dict(defaults, **values)
        config.update(device="cpu", implementation="original_author_CPU")
        return AuthorCPUAlpine(config), config
    if method == "opgm_unsupervised":
        from baseline.opgm_unsupervised.model import construct as build
        return build(values, seed, device)
    special = {
        "cena": ("SingleCENAConfig", "SingleCENA"),
        "gradalign_plus": ("AuthorGradConfig", "AuthorGradAlign"),
        "s_gwl": ("AuthorSGWLConfig", "AuthorSGWL"),
    }
    ordinary = {
        "fugal": ("fugal", "FUGALConfig", "FUGAL"),
        "grampa": ("grampa", "GRAMPAConfig", "GRAMPA"),
        "immnc": ("immnc", "IMMNCConfig", "IMMNC"),
        "SparkAlign": ("SparkAlign", "SparkAlignConfig", "SparkAlign"),
        "regal": ("regal", "REGALConfig", "REGAL"),
        "slotalign": ("slotalign", "SLOTAlignConfig", "SLOTAlign"),
        "ulot": ("ulot", "ULOTConfig", "ULOT"),
    }
    if method in special:
        module = importlib.import_module("baseline.author_adapters")
        config_name, model_name = special[method]
    else:
        try:
            module_name, config_name, model_name = ordinary[method]
        except KeyError:
            raise ValueError("Unknown method: " + method) from None
        module = importlib.import_module("baseline." + module_name)
    config_type = getattr(module, config_name)
    accepted = {field.name for field in fields(config_type)}
    runtime = dict(seed=seed, show_progress=False, max_dense_entries=150_000_000,
                   compute_cost_matrix=True, device=device)
    values.update({name: value for name, value in runtime.items() if name in accepted})
    config = config_type(**values)
    if hasattr(config, "validate"):
        config.validate()
    model = getattr(module, model_name)(config)
    if method == "fugal" and dataset_family == "Oregon-1":
        from baseline.fugal_numerics.fugal_converged_fit import fit_full
        model.fit = lambda first, second: fit_full(model, first, second)
    return model, asdict(config)
