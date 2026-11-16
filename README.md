# BiPGA

Standalone Python interfaces and twelve graph alignment methods, with construction, evaluation, diagnostics and paper-result reproduction.

- [Documentation](https://bipga.github.io/)
- [Install and download](https://bipga.github.io/installation.html)
- [Runtime source and command reference](runtime/README.md)
- [GPU Dockerfile](runtime/docker/Dockerfile)
- [Paper artifact](https://bipga.github.io/supplementary/)

## Install from source

```bash
python -m pip install "./runtime[all]"
bipga doctor --all
bipga demo --method all --device cpu --output demo_cpu
```

The existing `bpupga` import and the `bipga` import expose the same API. The runtime includes method code, weights, constructor, metrics and diagnostics; it runs independently of a research checkout. Full GPU execution requires an NVIDIA GPU and CUDA development tools.

The `runtime/` tree is the installable package. `guide/`, `src/`, `docs/` and `examples/` contain browsable documentation exports. The main documentation provides the installed CLI reference and current downloads.

[Release administration](runtime/RELEASING.md) describes Python trusted publishing and container registry setup.
